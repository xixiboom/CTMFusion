from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicSyncGraphConv(nn.Module):
    def __init__(self, dim: int, top_k: int = 8, spatial_sigma: float = 0.25, num_hops: int = 1):
        super().__init__()
        self.top_k = top_k
        self.spatial_sigma = spatial_sigma
        self.num_hops = num_hops
        self.out_proj = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))
        self._spatial_cache: Dict[Tuple[int, int, str], torch.Tensor] = {}

    def _build_spatial_affinity(self, n_h: int, n_w: int, device: torch.device) -> torch.Tensor:
        key = (n_h, n_w, str(device))
        cached = self._spatial_cache.get(key)
        if cached is not None and cached.device == device:
            return cached
        y = torch.arange(n_h, device=device).float() / max(n_h - 1, 1)
        x = torch.arange(n_w, device=device).float() / max(n_w - 1, 1)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack([yy.flatten(), xx.flatten()], dim=-1)
        dist = torch.cdist(coords, coords)
        aff = torch.exp(-dist.pow(2) / (2 * self.spatial_sigma ** 2))
        self._spatial_cache[key] = aff
        return aff

    def forward(self, x: torch.Tensor, sync_strength: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        B, N, D = x.shape
        if N <= 1 or self.top_k <= 0:
            return self.out_proj(x)
        sync_strength = sync_strength.clamp(0.0, 1.0)
        sync_adj = torch.bmm(sync_strength.unsqueeze(-1), sync_strength.unsqueeze(1))
        spatial_aff = self._build_spatial_affinity(n_h, n_w, x.device)
        adj = sync_adj * spatial_aff.unsqueeze(0)
        eye_mask = torch.eye(N, device=x.device).bool().unsqueeze(0)
        adj = adj.masked_fill(eye_mask, -float("inf"))
        k = min(self.top_k, N - 1)
        topk_vals, topk_idx = torch.topk(adj, k=k, dim=-1)
        sparse_adj = torch.full_like(adj, -float("inf"))
        sparse_adj.scatter_(-1, topk_idx, topk_vals)
        adj_norm = torch.nan_to_num(F.softmax(sparse_adj, dim=-1), nan=0.0, posinf=0.0, neginf=0.0)
        agg = x
        for _ in range(self.num_hops):
            agg = torch.bmm(adj_norm, agg)
        return self.out_proj(x + agg)


class LocalClusterCTMFusionCell(nn.Module):
    def __init__(self, dim: int, num_ticks: int = 3, history_len: int = 8,
                 num_out_pairs: int = 96, num_query_pairs: int = 48,
                 nlm_hidden: int = 16, attn_heads: int = 4, cluster_size: int = 4,
                 use_graph_agg: bool = True, graph_top_k: int = 8, graph_hops: int = 1):
        super().__init__()
        self.dim = dim
        self.num_ticks = num_ticks
        self.history_len = history_len
        self.cluster_size = cluster_size
        self.use_graph_agg = use_graph_agg
        if use_graph_agg:
            self.graph_agg = DynamicSyncGraphConv(dim, graph_top_k, num_hops=graph_hops)

        self.nlm_W1 = nn.Parameter(torch.randn(dim, history_len, nlm_hidden) * 0.08)
        self.nlm_b1 = nn.Parameter(torch.zeros(dim, nlm_hidden))
        self.nlm_W2 = nn.Parameter(torch.randn(dim, nlm_hidden, 2) * 0.08)
        self.nlm_b2 = nn.Parameter(torch.zeros(dim, 2))

        self.query_proj_left = nn.Linear(dim, num_query_pairs)
        self.query_proj_right = nn.Linear(dim, num_query_pairs)
        self.query_decay_near = nn.Parameter(torch.ones(num_query_pairs) * 0.8)
        self.query_decay_mid = nn.Parameter(torch.ones(num_query_pairs) * 0.3)
        self.query_decay_far = nn.Parameter(torch.ones(num_query_pairs) * 0.05)
        self.query_proj = nn.Linear(num_query_pairs, dim)

        self.out_proj_left = nn.Linear(dim, num_out_pairs)
        self.out_proj_right = nn.Linear(dim, num_out_pairs)
        self.out_decay_near = nn.Parameter(torch.ones(num_out_pairs) * 0.8)
        self.out_decay_mid = nn.Parameter(torch.ones(num_out_pairs) * 0.3)
        self.out_decay_far = nn.Parameter(torch.ones(num_out_pairs) * 0.05)
        self.out_proj = nn.Linear(num_out_pairs, dim)

        self.temporal_scale_weights = nn.Parameter(torch.randn(num_ticks, 3) * 0.1)
        self.sync_feat_proj = nn.Sequential(nn.Linear(1, dim), nn.GELU())
        self.step_mixing_weights = nn.Parameter(torch.randn(num_ticks, 3) * 0.1)
        self.cross_proj = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.synapse = nn.Sequential(nn.Linear(dim * 3, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.cross_attn = nn.MultiheadAttention(dim, attn_heads, batch_first=True)

    def _clusterify(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        B, C, H, W = x.shape
        pad_h = (self.cluster_size - H % self.cluster_size) % self.cluster_size
        pad_w = (self.cluster_size - W % self.cluster_size) % self.cluster_size
        if pad_h or pad_w:
            can_reflect_h = H > 1 and pad_h < H
            can_reflect_w = W > 1 and pad_w < W
            mode = "reflect" if can_reflect_h and can_reflect_w else "replicate"
            x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
        x = F.unfold(x, kernel_size=self.cluster_size, stride=self.cluster_size)
        x = x.view(B, C, self.cluster_size ** 2, -1).mean(dim=2)
        return x.permute(0, 2, 1), (H, W)

    def _declusterify(self, x_cluster: torch.Tensor, orig_size: Tuple[int, int]) -> Tuple[torch.Tensor, int, int]:
        B, N, C = x_cluster.shape
        H, W = orig_size
        n_h = (H + self.cluster_size - 1) // self.cluster_size
        n_w = (W + self.cluster_size - 1) // self.cluster_size
        expected = n_h * n_w
        if N != expected:
            raise RuntimeError(f"Cluster mismatch: got {N}, expected {expected} for {(H, W)}.")
        grid = x_cluster.permute(0, 2, 1).contiguous().view(B, C, n_h, n_w)
        return F.interpolate(grid, size=(H, W), mode="bilinear", align_corners=False), n_h, n_w

    @staticmethod
    def _update_history(hist: torch.Tensor, new_val: torch.Tensor) -> torch.Tensor:
        return torch.cat([hist[..., 1:], new_val.unsqueeze(-1)], dim=-1)

    def _calc_sync(self, Z_hist: torch.Tensor, proj_left: nn.Linear, proj_right: nn.Linear,
                   decay_near: torch.Tensor, decay_mid: torch.Tensor, decay_far: torch.Tensor,
                   step_idx: int) -> torch.Tensor:
        _, _, _, T = Z_hist.shape
        z_left = proj_left(Z_hist.transpose(2, 3)).transpose(2, 3)
        z_right = proj_right(Z_hist.transpose(2, 3)).transpose(2, 3)
        scale_w = F.softmax(self.temporal_scale_weights[step_idx], dim=0)
        decay = scale_w[0] * decay_near + scale_w[1] * decay_mid + scale_w[2] * decay_far
        t_idx = torch.arange(T, device=Z_hist.device).float()
        weights = torch.exp(-F.softplus(decay).unsqueeze(1) * t_idx.unsqueeze(0))
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)
        return (z_left * z_right * weights.unsqueeze(0).unsqueeze(0)).sum(dim=-1)

    def _calc_full_sync(self, Z_hist: torch.Tensor) -> torch.Tensor:
        return (Z_hist.unsqueeze(3) * Z_hist.unsqueeze(2)).mean(dim=-1)

    def init_state(self, f_a: torch.Tensor, f_b: Optional[torch.Tensor] = None,
                   init_prior: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        B, C, _, _ = f_a.shape
        init_feat = 0.5 * (f_a + f_b) if f_b is not None else f_a
        z_cluster, orig_size = self._clusterify(init_feat)
        _, n_h, n_w = self._declusterify(z_cluster, orig_size)
        Z_hist = z_cluster.unsqueeze(-1).repeat(1, 1, 1, self.history_len)
        A_hist = torch.zeros(B, z_cluster.shape[1], C, self.history_len, device=f_a.device, dtype=f_a.dtype)
        if init_prior is not None:
            prior_grid = F.adaptive_avg_pool2d(init_prior, (n_h, n_w))
            sync_feat = self.sync_feat_proj(prior_grid.view(B, -1, 1))
        else:
            sync_feat = torch.zeros(B, z_cluster.shape[1], self.dim, device=f_a.device, dtype=f_a.dtype)
        return {
            "z_cluster": z_cluster, "A_hist": A_hist, "Z_hist": Z_hist,
            "sync_feat": sync_feat, "orig_size": orig_size,
            "n_h": n_h, "n_w": n_w, "N_clu": z_cluster.shape[1],
            "cluster_sync": torch.zeros(B, z_cluster.shape[1], device=f_a.device, dtype=f_a.dtype),
        }

    def step(self, state: Dict[str, torch.Tensor], kv: torch.Tensor, step_idx: int,
             cross_guide: Optional[torch.Tensor] = None) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        z_cluster = state["z_cluster"]
        A_hist = state["A_hist"]
        Z_hist = state["Z_hist"]
        sync_feat = state["sync_feat"]
        orig_size = state["orig_size"]
        n_h, n_w = state["n_h"], state["n_w"]

        if cross_guide is not None:
            sync_feat = sync_feat + self.cross_proj(cross_guide)

        sync_q = self._calc_sync(Z_hist, self.query_proj_left, self.query_proj_right,
                                 self.query_decay_near, self.query_decay_mid, self.query_decay_far, step_idx)
        q = self.query_proj(sync_q)
        attn_out, _ = self.cross_attn(q, kv, kv, need_weights=False)

        sw = F.softmax(self.step_mixing_weights[step_idx], dim=0)
        synapse_in = torch.cat([sw[0] * attn_out, sw[1] * sync_feat, sw[2] * z_cluster], dim=-1)
        a = self.synapse(synapse_in)
        A_hist = self._update_history(A_hist, a)

        h = torch.einsum("bnct,cth->bnch", A_hist, self.nlm_W1) + self.nlm_b1
        h = F.relu(h, inplace=False)
        gamma_beta = torch.einsum("bnch,chk->bnck", h, self.nlm_W2) + self.nlm_b2
        gamma_raw, beta = gamma_beta[..., 0], gamma_beta[..., 1]
        gamma = torch.sigmoid(gamma_raw) * 0.8 + 0.1
        z_cluster = gamma * z_cluster + (1.0 - gamma) * beta

        cluster_sync = self._calc_sync(Z_hist, self.out_proj_left, self.out_proj_right,
                                       self.out_decay_near, self.out_decay_mid, self.out_decay_far,
                                       step_idx).mean(dim=-1)
        cluster_sync = torch.sigmoid(cluster_sync)
        if self.use_graph_agg:
            z_cluster = self.graph_agg(z_cluster, cluster_sync, n_h, n_w)

        Z_hist = self._update_history(Z_hist, z_cluster)
        sync_out = self._calc_sync(Z_hist, self.out_proj_left, self.out_proj_right,
                                   self.out_decay_near, self.out_decay_mid, self.out_decay_far,
                                   step_idx)
        y_t = self.out_proj(sync_out)
        y_map, _, _ = self._declusterify(y_t, orig_size)
        gamma_map, _, _ = self._declusterify(gamma, orig_size)
        beta_map, _, _ = self._declusterify(beta, orig_size)
        sync_feat = self.sync_feat_proj(cluster_sync.unsqueeze(-1))

        new_state = {
            "z_cluster": z_cluster, "A_hist": A_hist, "Z_hist": Z_hist,
            "sync_feat": sync_feat, "orig_size": orig_size,
            "n_h": n_h, "n_w": n_w, "N_clu": state["N_clu"],
            "cluster_sync": cluster_sync,
        }
        return new_state, y_map, gamma_map, beta_map


class CrossScaleSyncConduction(nn.Module):
    def __init__(self, dims: Tuple[int, int, int] = (64, 128, 128)):
        super().__init__()
        d1, d2, d3 = dims
        self.up_proj12 = nn.Conv2d(d1, d2, kernel_size=1)
        self.up_proj23 = nn.Conv2d(d2, d3, kernel_size=1)
        self.down_proj32 = nn.Conv2d(d3, d2, kernel_size=1)
        self.down_proj21 = nn.Conv2d(d2, d1, kernel_size=1)
        self.gate2 = nn.Sequential(nn.Conv2d(d2 * 2, d2, kernel_size=1), nn.Sigmoid())
        self.gate1 = nn.Sequential(nn.Conv2d(d1 * 2, d1, kernel_size=1), nn.Sigmoid())
        self.gate3 = nn.Sequential(nn.Conv2d(d3 * 2, d3, kernel_size=1), nn.Sigmoid())

    def forward(self, sync_feats: List[torch.Tensor], n_hw_list: List[Tuple[int, int]]) -> List[torch.Tensor]:
        f1, f2, f3 = sync_feats
        (nh1, nw1), (nh2, nw2), (nh3, nw3) = n_hw_list
        B = f1.shape[0]
        f1_grid = f1.transpose(1, 2).contiguous().view(B, -1, nh1, nw1)
        f2_grid = f2.transpose(1, 2).contiguous().view(B, -1, nh2, nw2)
        f3_grid = f3.transpose(1, 2).contiguous().view(B, -1, nh3, nw3)
        f1_to_2 = self.up_proj12(F.interpolate(f1_grid, size=(nh2, nw2), mode="bilinear", align_corners=False))
        f2_to_3 = self.up_proj23(F.interpolate(f2_grid, size=(nh3, nw3), mode="bilinear", align_corners=False))
        f3_to_2 = self.down_proj32(F.interpolate(f3_grid, size=(nh2, nw2), mode="bilinear", align_corners=False))
        f2_to_1 = self.down_proj21(F.interpolate(f2_grid, size=(nh1, nw1), mode="bilinear", align_corners=False))
        mix2 = f1_to_2 + f3_to_2
        g2 = self.gate2(torch.cat([f2_grid, mix2], dim=1))
        out2_grid = g2 * f2_grid + (1.0 - g2) * mix2
        g1 = self.gate1(torch.cat([f1_grid, f2_to_1], dim=1))
        out1_grid = g1 * f1_grid + (1.0 - g1) * f2_to_1
        g3 = self.gate3(torch.cat([f3_grid, f2_to_3], dim=1))
        out3_grid = g3 * f3_grid + (1.0 - g3) * f2_to_3
        return [
            out1_grid.flatten(2).transpose(1, 2),
            out2_grid.flatten(2).transpose(1, 2),
            out3_grid.flatten(2).transpose(1, 2),
        ]
