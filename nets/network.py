from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import (_safe_logit,_valid_num_groups, DEVICE)
from .encoders import WaveletEncoder, FullResolutionDetailEncoder, _HAS_PYTORCH_WAVELETS
from .ctm import LocalClusterCTMFusionCell, CrossScaleSyncConduction
from .decision import FocusPriorEstimator, ProgressiveRegionDecoder, FullResolutionDecisionHead, EdgeAnchoredDecisionRefiner


# -------------------- 推理后处理工具 --------------------
@torch.no_grad()
def fuse_by_decision(A: torch.Tensor, B: torch.Tensor, D: torch.Tensor,
                     hard: bool = False, threshold: float = 0.5) -> torch.Tensor:
    if hard:
        D = (D > threshold).float()
    return D * A + (1.0 - D) * B


@torch.no_grad()
def hard_decision_postprocess(D: torch.Tensor, kernel_size: int = 3, threshold: float = 0.5,
                              smooth_before_threshold: bool = True) -> torch.Tensor:
    D_work = D.clamp(0.0, 1.0)
    if smooth_before_threshold:
        band = torch.exp(-((D_work - 0.5).pow(2)) / (2.0 * 0.10 ** 2)).clamp(0.0, 1.0)
        D_avg = F.avg_pool2d(D_work, kernel_size=3, stride=1, padding=1)
        D_work = D_work * (1.0 - 0.35 * band) + D_avg * (0.35 * band)
    D_bin = (D_work > threshold).float()
    pad = kernel_size // 2
    dilate = F.max_pool2d(D_bin, kernel_size, stride=1, padding=pad)
    close = -F.max_pool2d(-dilate, kernel_size, stride=1, padding=pad)
    erode = -F.max_pool2d(-close, kernel_size, stride=1, padding=pad)
    opened = F.max_pool2d(erode, kernel_size, stride=1, padding=pad)
    return opened


class MyNetwork(nn.Module):
    def __init__(self, img_channels: int = 3, hidden_channels: int = 32, dropout: float = 0.0,
                 num_ticks: int = 3, history_len: int = 8, cluster_size: int = 8,
                 graph_top_k: int = 8, graph_hops: int = 1,
                 num_out_pairs: int = 96, num_query_pairs: int = 48, nlm_hidden: int = 16,
                 wavelet_J: int = 2, wavelet_backend: str = "auto",
                 return_details: bool = False, use_feedback: bool = True,
                 refiner_iter: int = 4, **kwargs):
        super().__init__()
        self.num_ticks = num_ticks
        self.return_details = return_details
        self.use_feedback = use_feedback
        base_dim = hidden_channels

        self.focus_prior = FocusPriorEstimator(temp=4.0)
        self.encoder = WaveletEncoder(in_channels=img_channels, base_dim=base_dim, J=wavelet_J, backend=wavelet_backend)
        self.detail_encoder = FullResolutionDetailEncoder(in_channels=img_channels, dim=base_dim)

        dims = (base_dim * 2, base_dim * 4, base_dim * 4)

        c1 = max(2, cluster_size // 2)
        c2 = max(2, cluster_size // 2)
        c3 = max(1, cluster_size // 4)

        self.ctm_level1 = LocalClusterCTMFusionCell(
            dim=dims[0], num_ticks=num_ticks, history_len=history_len,
            num_out_pairs=max(16, num_out_pairs // 2), num_query_pairs=max(16, num_query_pairs // 2),
            nlm_hidden=nlm_hidden, attn_heads=2, cluster_size=c1,
            graph_top_k=graph_top_k, graph_hops=graph_hops,
        )
        self.ctm_level2 = LocalClusterCTMFusionCell(
            dim=dims[1], num_ticks=num_ticks, history_len=history_len,
            num_out_pairs=num_out_pairs, num_query_pairs=num_query_pairs,
            nlm_hidden=nlm_hidden, attn_heads=4, cluster_size=c2,
            graph_top_k=graph_top_k, graph_hops=graph_hops,
        )
        self.ctm_level3 = LocalClusterCTMFusionCell(
            dim=dims[2], num_ticks=num_ticks, history_len=history_len,
            num_out_pairs=num_out_pairs, num_query_pairs=num_query_pairs,
            nlm_hidden=nlm_hidden, attn_heads=4, cluster_size=c3,
            graph_top_k=graph_top_k, graph_hops=graph_hops,
        )

        self.cross_sync = CrossScaleSyncConduction(dims=dims)
        self.sync_skip_gate1 = nn.Sequential(
            nn.Conv2d(1, dims[0], 3, padding=1),
            nn.GroupNorm(_valid_num_groups(8, dims[0]), dims[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(dims[0], dims[0], 3, padding=1),
            nn.Sigmoid(),
        )
        self.sync_skip_gate2 = nn.Sequential(
            nn.Conv2d(1, dims[1], 3, padding=1),
            nn.GroupNorm(_valid_num_groups(16, dims[1]), dims[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(dims[1], dims[1], 3, padding=1),
            nn.Sigmoid(),
        )
        self.region_decoder = ProgressiveRegionDecoder(dim_deep=dims[2], skip_dims=(dims[0], dims[1]), dropout=dropout)
        self.detail_head = FullResolutionDecisionHead(img_channels=img_channels, detail_dim=base_dim, hidden=base_dim)
        self.refiner = EdgeAnchoredDecisionRefiner(inner_iter=refiner_iter)

        self.prior_logit_scale = nn.Parameter(torch.tensor(1.25))
        self.ctm_logit_scale = nn.Parameter(torch.tensor(0.35))
        self.region_logit_scale = nn.Parameter(torch.tensor(0.45))
        self.detail_logit_scale = nn.Parameter(torch.tensor(0.55))
        self.tick_fusion = nn.Parameter(torch.linspace(0.35, 1.0, steps=num_ticks))

    @staticmethod
    def _make_kv(a_feat: torch.Tensor, b_feat: torch.Tensor) -> torch.Tensor:
        return torch.cat([a_feat.flatten(2).transpose(1, 2), b_feat.flatten(2).transpose(1, 2)], dim=1)

    @staticmethod
    def _cluster_sync_to_map(sync_clu: torch.Tensor, n_h: int, n_w: int, target_shape: Tuple[int, int]) -> torch.Tensor:
        B, N = sync_clu.shape
        if N != n_h * n_w:
            raise RuntimeError(f"sync cluster shape mismatch: N={N}, n_h*n_w={n_h*n_w}")
        grid = sync_clu.view(B, 1, n_h, n_w)
        return F.interpolate(grid, size=target_shape, mode="bilinear", align_corners=False)

    @staticmethod
    def _channel_mean_gate(y: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(y).mean(dim=1, keepdim=True).clamp(1e-4, 1.0 - 1e-4)

    def forward_full(self, A: torch.Tensor, B: torch.Tensor,
                     max_ticks: Optional[int] = None) -> Tuple[List[torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
        T = int(max_ticks) if max_ticks is not None else self.num_ticks
        T = max(1, min(T, self.num_ticks))

        prior_decision, focus_conf, signed_focus, eA, eB = self.focus_prior(A, B)

        a1, a2, a3, prior_A_low, energy_A_low = self.encoder(A)
        b1, b2, b3, prior_B_low, energy_B_low = self.encoder(B)
        da = self.detail_encoder(A)
        db = self.detail_encoder(B)

        low_signed = (energy_A_low - energy_B_low) / (energy_A_low + energy_B_low + 1e-6)
        low_conf = F.avg_pool2d(low_signed.abs(), kernel_size=5, stride=1, padding=2).clamp(0.0, 1.0)

        state1 = self.ctm_level1.init_state(a1, b1, low_conf)
        state2 = self.ctm_level2.init_state(a2, b2, low_conf)
        state3 = self.ctm_level3.init_state(a3, b3, low_conf)
        kv1 = self._make_kv(a1, b1)
        kv2 = self._make_kv(a2, b2)
        kv3 = self._make_kv(a3, b3)
        n_hw_list = [(state1["n_h"], state1["n_w"]), (state2["n_h"], state2["n_w"]), (state3["n_h"], state3["n_w"])]

        outputs_raw: List[torch.Tensor] = []
        outputs_refined: List[torch.Tensor] = []
        anchor_weights: List[torch.Tensor] = []
        sync_maps: List[torch.Tensor] = []
        ctm_decision_maps: List[torch.Tensor] = []
        modulations: List[torch.Tensor] = []

        for t in range(T):
            for state, ctm in [(state1, self.ctm_level1), (state2, self.ctm_level2), (state3, self.ctm_level3)]:
                p = F.adaptive_avg_pool2d(low_conf, (state["n_h"], state["n_w"]))
                state["sync_feat"] = state["sync_feat"] + ctm.sync_feat_proj(p.view(p.shape[0], -1, 1))

            guides = self.cross_sync([state1["sync_feat"], state2["sync_feat"], state3["sync_feat"]], n_hw_list)
            state1, y1, _, _ = self.ctm_level1.step(state1, kv1, t, guides[0])
            state2, y2, _, _ = self.ctm_level2.step(state2, kv2, t, guides[1])
            state3, y3, g3, b3_mod = self.ctm_level3.step(state3, kv3, t, guides[2])
            modulations.append(torch.stack([g3, b3_mod], dim=-1))

            sync1 = self._cluster_sync_to_map(state1["cluster_sync"], state1["n_h"], state1["n_w"], a1.shape[2:])
            sync2 = self._cluster_sync_to_map(state2["cluster_sync"], state2["n_h"], state2["n_w"], a2.shape[2:])
            sync3 = self._cluster_sync_to_map(state3["cluster_sync"], state3["n_h"], state3["n_w"], a3.shape[2:])

            gate1_ch = 0.5 * self.sync_skip_gate1(sync1) + 0.5 * torch.sigmoid(y1)
            gate2_ch = 0.5 * self.sync_skip_gate2(sync2) + 0.5 * torch.sigmoid(y2)
            gate3_ch = torch.sigmoid(y3)
            skip1 = gate1_ch * a1 + (1.0 - gate1_ch) * b1
            skip2 = gate2_ch * a2 + (1.0 - gate2_ch) * b2
            deep = gate3_ch * a3 + (1.0 - gate3_ch) * b3

            region_logit = self.region_decoder(deep, skip2, skip1)
            region_logit = F.interpolate(region_logit, size=A.shape[2:], mode="bilinear", align_corners=False)

            ctm_dec1 = F.interpolate(self._channel_mean_gate(y1), size=A.shape[2:], mode="bilinear", align_corners=False)
            ctm_dec2 = F.interpolate(self._channel_mean_gate(y2), size=A.shape[2:], mode="bilinear", align_corners=False)
            ctm_dec3 = F.interpolate(self._channel_mean_gate(y3), size=A.shape[2:], mode="bilinear", align_corners=False)
            ctm_decision = (ctm_dec1 + ctm_dec2 + ctm_dec3) / 3.0

            sync_full = (
                F.interpolate(sync1, size=A.shape[2:], mode="bilinear", align_corners=False) +
                F.interpolate(sync2, size=A.shape[2:], mode="bilinear", align_corners=False) +
                F.interpolate(sync3, size=A.shape[2:], mode="bilinear", align_corners=False)
            ) / 3.0

            detail_logit = self.detail_head(A, B, da, db, prior_decision, focus_conf, signed_focus, ctm_decision, sync_full)

            decision_logit = (
                self.prior_logit_scale * _safe_logit(prior_decision) +
                self.ctm_logit_scale * _safe_logit(ctm_decision) +
                self.region_logit_scale * region_logit +
                self.detail_logit_scale * detail_logit
            )
            raw_decision = torch.sigmoid(decision_logit).clamp(1e-4, 1.0 - 1e-4)
            refined_decision, anchor = self.refiner(raw_decision, focus_conf, sync_full, A, B)

            outputs_raw.append(raw_decision)
            outputs_refined.append(refined_decision)
            anchor_weights.append(anchor)
            sync_maps.append(sync_full)
            ctm_decision_maps.append(ctm_decision)

            if self.use_feedback and t < T - 1:
                d3 = F.adaptive_avg_pool2d(refined_decision.detach(), a3.shape[2:])
                d2 = F.adaptive_avg_pool2d(refined_decision.detach(), a2.shape[2:])
                d1 = F.adaptive_avg_pool2d(refined_decision.detach(), a1.shape[2:])
                a3 = a3 * (0.78 + 0.22 * d3)
                b3 = b3 * (0.78 + 0.22 * (1.0 - d3))
                a2 = a2 * (0.78 + 0.22 * d2)
                b2 = b2 * (0.78 + 0.22 * (1.0 - d2))
                a1 = a1 * (0.78 + 0.22 * d1)
                b1 = b1 * (0.78 + 0.22 * (1.0 - d1))
                kv1 = self._make_kv(a1, b1)
                kv2 = self._make_kv(a2, b2)
                kv3 = self._make_kv(a3, b3)

        weights = F.softmax(self.tick_fusion[:T], dim=0)
        raw_final = sum(w * o for w, o in zip(weights, outputs_raw))
        refined_final = sum(w * o for w, o in zip(weights, outputs_refined))
        sync_final = sum(w * s for w, s in zip(weights, sync_maps))
        ctm_final = sum(w * c for w, c in zip(weights, ctm_decision_maps))
        refined_final, final_anchor = self.refiner(refined_final, focus_conf, sync_final, A, B)
        fused_img = refined_final * A + (1.0 - refined_final) * B

        details = {
            "decision": refined_final,
            "decision_map": raw_final,
            "refined_decision": refined_final,
            "fused_img": fused_img,
            "fused_direct": raw_final * A + (1.0 - raw_final) * B,
            "prior_decision": prior_decision,
            "focus_conf": focus_conf,
            "signed_focus": signed_focus,
            "focus_energy_A": eA,
            "focus_energy_B": eB,
            "ctm_decision_map": ctm_final,
            "ctm_sync_map": sync_final,
            "outputs_raw": outputs_raw,
            "outputs_refined": outputs_refined,
            "anchor_weights": anchor_weights,
            "anchor_mask": final_anchor,
            "final_anchor": final_anchor,
            "modulations": modulations,
            "sync_strength_clusters": state3["cluster_sync"],
            "sync_strength_clusters_level1": state1["cluster_sync"],
            "sync_strength_clusters_level2": state2["cluster_sync"],
            "sync_matrix": self.ctm_level3._calc_full_sync(state3["Z_hist"]).mean(dim=1),
            "architecture": "Detail-preserving signed-prior WaveletEncoder + scale-aware LocalClusterCTM + edge-anchored refiner",
            "ctm_enabled": True,
        }
        return outputs_refined, low_conf, details

    def forward(self, A: torch.Tensor, B: torch.Tensor,
                return_details: Optional[bool] = None,
                max_ticks: Optional[int] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if return_details is None:
            return_details = self.return_details
        _, _, details = self.forward_full(A, B, max_ticks=max_ticks)
        D = details["refined_decision"]
        if return_details:
            return D, details
        return D


# -------------------- 冒烟测试 --------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    if DEVICE.type == "cpu":
        torch.set_num_threads(1)

    A = torch.rand(1, 3, 64, 64).to(DEVICE)
    B = torch.rand(1, 3, 64, 64).to(DEVICE)
    model = MyNetwork(
        img_channels=3,
        hidden_channels=16,
        dropout=0.0,
        num_ticks=1,
        history_len=4,
        cluster_size=8,
        graph_top_k=4,
        graph_hops=1,
        wavelet_backend="auto",
        return_details=True,
    ).to(DEVICE)

    with torch.no_grad():
        D, details = model(A, B)
        Fused = fuse_by_decision(A, B, D)
        D_hard = hard_decision_postprocess(D)

    print("=" * 80)
    print("Detail-preserving CTM MFF smoke test passed")
    print(f"pytorch_wavelets available: {_HAS_PYTORCH_WAVELETS}")
    print(f"Decision map shape: {tuple(D.shape)}")
    print(f"Hard map shape: {tuple(D_hard.shape)}")
    print(f"Fused image shape: {tuple(Fused.shape)}")
    print(f"CTM decision map shape: {tuple(details['ctm_decision_map'].shape)}")
    print(f"CTM sync map shape: {tuple(details['ctm_sync_map'].shape)}")
    print(f"Params/M: {sum(p.numel() for p in model.parameters()) / 1e6:.3f}")
    print("=" * 80)
