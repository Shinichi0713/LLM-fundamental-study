import torch
import torch.nn as nn
import torch.nn.functional as F

class Module(nn.Module):
    """1つの小さなモジュール（2層FFN）"""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)


class RoutingNetwork(nn.Module):
    """ルーティングネットワーク＋モジュール群"""
    def __init__(self, dim, hidden_dim, num_modules, tau=1.0, hard=False):
        super().__init__()
        self.dim = dim
        self.num_modules = num_modules
        self.tau = tau          # Gumbel-Softmaxの温度パラメータ
        self.hard = hard        # 推論時にone-hotに丸めるかどうか

        # モジュール群
        self.modules_list = nn.ModuleList([
            Module(dim, hidden_dim) for _ in range(num_modules)
        ])

        # ルーティングネットワーク（小さなFFN）
        self.router = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_modules)  # 各モジュールのスコアを出力
        )

    def forward(self, x, num_steps=3):
        """
        x: [batch_size, dim]
        num_steps: 何ステップモジュールを適用するか
        """
        h = x  # 現在の隠れ状態

        # 各ステップでどのモジュールが使われたかを記録（可視化用）
        used_modules = []

        for step in range(num_steps):
            # ルーティングネットワークで各モジュールのスコアを計算
            logits = self.router(h)  # [batch_size, num_modules]

            # Gumbel-Softmaxでモジュール選択（学習時はsoft, 推論時はhardにすることも可能）
            module_weights = F.gumbel_softmax(logits, tau=self.tau, hard=self.hard, dim=-1)
            # module_weights: [batch_size, num_modules]

            # 使用されたモジュールのインデックス（可視化用）
            if self.hard:
                # hard=True のときはargmaxでインデックスを取得
                chosen = torch.argmax(module_weights, dim=-1)  # [batch_size]
            else:
                # softな場合は「一番重みが大きいもの」を代表として記録
                chosen = torch.argmax(module_weights, dim=-1)
            used_modules.append(chosen.detach().cpu().numpy())

            # 各モジュールの出力を重み付きで合成
            # ここでは単純に「バッチ内各サンプルごとに1つのモジュールだけ」を使う実装も可能ですが、
            # 一応「すべてのモジュール出力を重み付きで混ぜる」形にしています。
            outputs = []
            for i, mod in enumerate(self.modules_list):
                out_i = mod(h)  # [batch_size, dim]
                # モジュールiの重みをブロードキャストして掛ける
                w_i = module_weights[:, i].unsqueeze(-1)  # [batch_size, 1]
                outputs.append(out_i * w_i)

            # すべてのモジュール出力を足し合わせる
            new_h = sum(outputs)  # [batch_size, dim]

            # 残差接続的な更新（単純に加算）
            h = h + new_h

        return h, used_modules

# ハイパーパラメータ
batch_size = 4
dim = 64
hidden_dim = 128
num_modules = 8
num_steps = 3

# モデル作成
model = RoutingNetwork(
    dim=dim,
    hidden_dim=hidden_dim,
    num_modules=num_modules,
    tau=1.0,
    hard=False  # 学習時はsoft, 推論時はhard=Trueにしてもよい
)

# ダミー入力
x = torch.randn(batch_size, dim)

# 順伝播
out, used_modules = model(x, num_steps=num_steps)

print("出力形状:", out.shape)  # [batch_size, dim]
print("各ステップで使われたモジュールインデックス（バッチごと）:")
for step, idx in enumerate(used_modules):
    print(f"ステップ {step}: {idx}")