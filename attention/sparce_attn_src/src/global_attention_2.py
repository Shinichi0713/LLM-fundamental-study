import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseGlobalAttention(nn.Module):
    def __init__(self, dim, num_heads=8, window=4):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window = window
        self.kernel_size = 2*window + 1

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, global_mask):
        """
        x: (B,T,D)
        global_mask: (B,T) bool — True の位置がグローバルトークン
        """
        B,T,D = x.shape

        # ---- 1) Project to multi-head ----
        q = self.q_proj(x).view(B,T,self.num_heads,self.head_dim).transpose(1,2)  # (B,H,T,Dh)
        k = self.k_proj(x).view(B,T,self.num_heads,self.head_dim).transpose(1,2)
        v = self.v_proj(x).view(B,T,self.num_heads,self.head_dim).transpose(1,2)

        # ============================================================
        # 2) Local Attention: sliding window (各クエリは近傍のみ参照)
        # ============================================================
        local_scores = torch.zeros(B,self.num_heads,T,self.kernel_size, device=x.device)

        for t in range(T):
            L = max(0, t-self.window)
            R = min(T, t+self.window+1)
            k_local = k[:,:,L:R,:]          # (B,H,win,Dh)
            q_t = q[:,:,t:t+1,:]            # (B,H,1,Dh)
            score = torch.einsum("bhid,bhjd->bhij", q_t, k_local) / (self.head_dim**0.5)
            local_scores[:,:,t,:R-L] = score.squeeze(2)

        # ============================================================
        # 3) Global Attention 部分
        # ============================================================
        # global_mask = True のトークンだけ全部に対して attention！
        global_scores = []

        for b in range(B):
            idx = torch.nonzero(global_mask[b], as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                global_scores.append(None)
                continue

            k_g = k[b,:,idx,:]  # (H, G, Dh)
            v_g = v[b,:,idx,:]

            # Q 全部のトークン → global tokens のスコア
            score = torch.einsum("htd,hgd->htg", q[b], k_g) / (self.head_dim**0.5)
            global_scores.append(score)     # (H,T,G)

        # ============================================================
        # 4) Local + Global を結合して softmax
        # ============================================================
        out_heads = torch.zeros(B,self.num_heads,T,self.head_dim, device=x.device)

        for b in range(B):
            for h in range(self.num_heads):
                for t in range(T):
                    # local
                    L = max(0, t-self.window)
                    R = min(T, t+self.window+1)
                    local_s = local_scores[b,h,t,:R-L]

                    if global_scores[b] is None:
                        # local only
                        attn = F.softmax(local_s, dim=-1)
                        v_loc = v[b,h,L:R,:]
                        ctx = torch.sum(attn.unsqueeze(-1) * v_loc, dim=-2)
                    else:
                        # local + global
                        g_score = global_scores[b][h,t]   # (G)
                        s = torch.cat([local_s, g_score], dim=-1)   # (win+G)
                        attn = F.softmax(s, dim=-1)

                        # 分割
                        attn_loc = attn[:R-L]
                        attn_g   = attn[R-L:]

                        v_loc = v[b,h,L:R,:]
                        v_g   = v[b,h, global_mask[b], :]

                        ctx = (
                            torch.sum(attn_loc.unsqueeze(-1)*v_loc, dim=-2) +
                            torch.sum(attn_g.unsqueeze(-1)*v_g, dim=-2)
                        )

                    out_heads[b,h,t] = ctx

        # ---- merge heads ----
        out = out_heads.transpose(1,2).reshape(B,T,D)
        out = self.out_proj(out)
        return out
