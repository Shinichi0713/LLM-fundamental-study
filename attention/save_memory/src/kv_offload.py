import torch
import numpy as np
import os

class KVCacheOffloader:
    def __init__(self, head_dim, max_tokens, device="cuda",
                 cpu_limit=256, nvme_path="kv_cache"):
        """
        cpu_limit : CPUに保持する最大トークン数
        """
        self.device = device
        self.head_dim = head_dim
        self.cpu_limit = cpu_limit

        self.gpu_k = []
        self.gpu_v = []

        self.cpu_k = []
        self.cpu_v = []

        os.makedirs(nvme_path, exist_ok=True)
        self.nvme_k = np.memmap(
            f"{nvme_path}/k.dat",
            dtype=np.float32,
            mode="w+",
            shape=(max_tokens, head_dim),
        )
        self.nvme_v = np.memmap(
            f"{nvme_path}/v.dat",
            dtype=np.float32,
            mode="w+",
            shape=(max_tokens, head_dim),
        )

        self.nvme_ptr = 0

    def append(self, k, v):
        """
        k, v: (1, head_dim) on GPU
        """
        self.gpu_k.append(k)
        self.gpu_v.append(v)

        # GPU → CPU オフロード
        if len(self.gpu_k) > 1:
            old_k = self.gpu_k.pop(0).cpu()
            old_v = self.gpu_v.pop(0).cpu()
            self.cpu_k.append(old_k)
            self.cpu_v.append(old_v)

        # CPU → NVMe オフロード
        if len(self.cpu_k) > self.cpu_limit:
            ck = self.cpu_k.pop(0).numpy()
            cv = self.cpu_v.pop(0).numpy()
            self.nvme_k[self.nvme_ptr] = ck
            self.nvme_v[self.nvme_ptr] = cv
            self.nvme_ptr += 1

    def get_all_kv(self):
        """
        全KVをGPUに集約（Attention用）
        """
        kv = []

        # NVMe → CPU → GPU
        if self.nvme_ptr > 0:
            nk = torch.from_numpy(self.nvme_k[:self.nvme_ptr]).to(self.device)
            nv = torch.from_numpy(self.nvme_v[:self.nvme_ptr]).to(self.device)
            kv.append((nk, nv))

        if self.cpu_k:
            ck = torch.cat(self.cpu_k).to(self.device)
            cv = torch.cat(self.cpu_v).to(self.device)
            kv.append((ck, cv))

        if self.gpu_k:
            gk = torch.cat(self.gpu_k)
            gv = torch.cat(self.gpu_v)
            kv.append((gk, gv))

        K = torch.cat([x[0] for x in kv], dim=0)
        V = torch.cat([x[1] for x in kv], dim=0)
        return K, V

class SimpleAttention(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = torch.nn.Linear(dim, dim)
        self.k = torch.nn.Linear(dim, dim)
        self.v = torch.nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x, kv_cache: KVCacheOffloader):
        """
        x: (1, dim)
        """
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        kv_cache.append(k, v)

        K, V = kv_cache.get_all_kv()  # 全履歴取得

        attn = (q @ K.T) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = attn @ V
        return out

def generate(model, steps=512, dim=64):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    kv_cache = KVCacheOffloader(
        head_dim=dim,
        max_tokens=steps,
        device=device,
        cpu_limit=64,
    )

    x = torch.zeros(1, dim, device=device)

    for i in range(steps):
        with torch.no_grad():
            x = model(x, kv_cache)

        if i % 50 == 0:
            print(f"step {i}, KV total = {i+1}")

    print("Generation done")


attn = SimpleAttention(dim=64)
generate(attn, steps=300)

