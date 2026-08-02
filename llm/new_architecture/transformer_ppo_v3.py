import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# TransformerEncoderLayer の fast path（融合カーネル）による形状バグを回避
torch.backends.mha.set_fastpath_enabled(False)


# ================================================================
# 1. 安全なモデルロード関数
# ================================================================
def load_model_safely(model, path, device="cpu"):
    try:
        checkpoint_state = torch.load(path, map_location=device)
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return

    if isinstance(checkpoint_state, dict) and "state_dict" in checkpoint_state:
        checkpoint_state = checkpoint_state["state_dict"]

    current_state = model.state_dict()
    filtered_state = {}
    skipped_keys = []

    for name, param in checkpoint_state.items():
        if name in current_state:
            if param.shape == current_state[name].shape:
                filtered_state[name] = param
            else:
                skipped_keys.append(f"{name} (Shape Mismatch: saved {tuple(param.shape)} vs model {tuple(current_state[name].shape)})")
        else:
            skipped_keys.append(f"{name} (Key Not Found in Model)")

    model.load_state_dict(filtered_state, strict=False)
    print(f"Successfully loaded {len(filtered_state)} / {len(current_state)} layers.")
    if skipped_keys:
        print("Skipped layers due to mismatch or missing:")
        for key in skipped_keys:
            print(f" - {key}")


# ================================================================
# 1.5. 行動マスキング（壁・盤外に向かう行動を無効化する）
# ================================================================
def compute_valid_action_mask(x, wall_channel_idx=0, agent_channel_idx=3):
    """
    観測テンソル x: (B, C, H, W) から、現在位置で選択可能な行動(上下左右)の
    マスクを計算する。有効 = 盤内 かつ 壁でない。

    これがないと、方策は壁や盤外に向かう行動もロジット上は選び得るため、
    遠いゴールに向かう長い正解シーケンスほど「1手でも間違えると失敗」の
    確率が積み重なってしまう（近い迷路は解けるが遠い迷路は解けない、
    という症状の主因になりやすい）。

    戻り値: (B, 4) の bool テンソル (True=選択可能)
    action定義は MazeEnv.step に合わせる: 0=上 1=下 2=左 3=右
    """
    B, C, H, W = x.shape
    device = x.device
    wall_map = x[:, wall_channel_idx]          # (B, H, W)
    agent_map = x[:, agent_channel_idx]        # (B, H, W)
    agent_flat = agent_map.reshape(B, -1).argmax(dim=1)  # (B,)
    agent_r = agent_flat // W
    agent_c = agent_flat % W

    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    valid = torch.zeros(B, 4, dtype=torch.bool, device=device)
    batch_idx = torch.arange(B, device=device)

    for a, (dr, dc) in enumerate(deltas):
        nr = agent_r + dr
        nc = agent_c + dc
        in_bounds = (nr >= 0) & (nr < H) & (nc >= 0) & (nc < W)
        nr_clamped = nr.clamp(0, H - 1)
        nc_clamped = nc.clamp(0, W - 1)
        is_wall = wall_map[batch_idx, nr_clamped, nc_clamped] > 0.5
        valid[:, a] = in_bounds & (~is_wall)

    return valid


def apply_action_mask(logits, valid_mask, mask_value=-1e9):
    """
    valid_mask=Falseの行動のロジットを非常に小さい値にして、実質選ばれなく
    する。全行動が無効（迷路生成のバグ等で完全に孤立したマス）という
    異常系のときは、NaN化を避けるためマスクをかけない（フォールバック）。
    """
    all_invalid = ~valid_mask.any(dim=1)
    safe_mask = valid_mask.clone()
    if all_invalid.any():
        safe_mask[all_invalid] = True
    return logits.masked_fill(~safe_mask, mask_value)


class PPORolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def push(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

# ================================================================
# 2. Transformer Actor-Critic Network (形状変換の完全防御)
# ================================================================
class TransformerActorCritic(nn.Module):
    def __init__(
        self,
        in_channels=5,   # 変更: ch4=ゴールまでの正規化BFS距離 を追加（MazeEnv側の変更と対応）
        grid_size=5,
        d_model=64,
        nhead=4,
        num_layers=2,
        action_dim=4,
        hidden_size=128,
        wall_channel_idx=0,   # obs[0] = 壁マップ (1=壁, 0=通行可)
        use_wall_mask=False,  # 変更: 既定でOFF。理由は_build_attn_maskのdocstring参照
    ):
        super().__init__()
        assert d_model % 2 == 0, "d_model は2次元位置エンコーディングのため偶数にしてください"

        self.grid_size = grid_size
        self.in_channels = in_channels
        self.num_tokens = grid_size * grid_size
        self.nhead = nhead
        self.wall_channel_idx = wall_channel_idx
        self.d_model = d_model
        self.use_wall_mask = use_wall_mask

        self.embedding = nn.Linear(in_channels, d_model)

        # --- 2次元位置エンコーディング ---
        # row / col それぞれ d_model/2 次元。concatして d_model 次元にする。
        half_dim = d_model // 2
        self.row_embedding = nn.Parameter(torch.randn(grid_size, half_dim) * 0.02)
        self.col_embedding = nn.Parameter(torch.randn(grid_size, half_dim) * 0.02)

        # CLSトークン用（グリッド外の特別な位置なので独立パラメータ）
        self.cls_pos_embedding = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # 事前計算: 各セルindex -> (row_idx, col_idx) の対応をbufferとして保持
        row_idx, col_idx = self._build_grid_indices(grid_size)
        self.register_buffer("row_idx", row_idx, persistent=False)  # (N,)
        self.register_buffer("col_idx", col_idx, persistent=False)  # (N,)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        self.actor_head = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, action_dim)
        )
        self.critic_head = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1)
        )

        # 隣接ペア（上下左右）を事前計算し、モデルのbufferとして登録
        idx_i, idx_j = self._build_neighbor_indices(grid_size)
        self.register_buffer("neighbor_i", idx_i, persistent=False)
        self.register_buffer("neighbor_j", idx_j, persistent=False)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _build_grid_indices(grid_size):
        """トークン順(row-major: i = r*W + c)に対応する row_idx, col_idx を返す"""
        H = W = grid_size
        rows, cols = [], []
        for r in range(H):
            for c in range(W):
                rows.append(r)
                cols.append(c)
        return torch.tensor(rows, dtype=torch.long), torch.tensor(cols, dtype=torch.long)

    @staticmethod
    def _build_neighbor_indices(grid_size):
        H = W = grid_size
        pairs_i, pairs_j = [], []
        for r in range(H):
            for c in range(W):
                i = r * W + c
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        pairs_i.append(i)
                        pairs_j.append(nr * W + nc)
        return torch.tensor(pairs_i, dtype=torch.long), torch.tensor(pairs_j, dtype=torch.long)

    def _build_2d_pos_embedding(self):
        """
        row_embedding[row_idx] と col_embedding[col_idx] をconcatして
        (1, N, d_model) の位置エンコーディングを作る
        """
        row_pe = self.row_embedding[self.row_idx]  # (N, half_dim)
        col_pe = self.col_embedding[self.col_idx]  # (N, half_dim)
        pos = torch.cat([row_pe, col_pe], dim=-1)  # (N, d_model)
        return pos.unsqueeze(0)  # (1, N, d_model)

    def _build_attn_mask(self, wall_map, agent_pos_flat, B, device):
        """
        wall_map: (B, H, W)
        agent_pos_flat: (B,) 各バッチの現在位置のフラットインデックス

        【重要な制約・use_wall_mask=Falseにした理由】
        このmaskはCLSトークンが「現在位置から1ホップ隣接するマスの情報」しか
        直接受け取れない設計になっている（adjacencyは上下左右1マスのみ）。
        num_layers=2の場合、CLSが集約できる情報は実質2ホップ分に限られ、
        5x5迷路で必要になりうる距離（最大8マス程度）の経路情報を
        表現できない。これが「数マス先の迂回が必要な壁を避けられない」
        原因になっていたため、既定では使わない。

        代わりにMazeEnv側でBFS距離マップをobs[4]として直接与えることで、
        受容野に依存せず各セルがゴール方向の情報を持てるようにしている。

        use_wall_mask=True にすると従来通りこのmaskを使う（アブレーション用）。
        使う場合は num_layers を迷路の直径をカバーできる程度まで
        増やすことを推奨（5x5なら num_layers=4〜6程度）。
        """
        H = W = self.grid_size
        N = self.num_tokens
        wall_flat = wall_map.reshape(B, N)

        adjacency = torch.zeros(B, N, N, dtype=torch.bool, device=device)
        not_wall_i = (wall_flat[:, self.neighbor_i] == 0)
        not_wall_j = (wall_flat[:, self.neighbor_j] == 0)
        connected = not_wall_i & not_wall_j
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, self.neighbor_i.shape[0])
        adjacency[batch_idx, self.neighbor_i.expand(B, -1), self.neighbor_j.expand(B, -1)] = connected
        diag_idx = torch.arange(N, device=device)
        adjacency[:, diag_idx, diag_idx] = True

        full_adj = torch.zeros(B, N + 1, N + 1, dtype=torch.bool, device=device)
        full_adj[:, 1:, 1:] = adjacency

        # CLS(index 0) は「現在位置のマスと同じ接続性」を持たせる
        # -> 現在位置から到達可能なマスの情報だけを、CLSは直接受け取れる
        batch_range = torch.arange(B, device=device)
        cls_connectivity = adjacency[batch_range, agent_pos_flat]  # (B, N) 現在位置の行を流用
        full_adj[:, 0, 1:] = cls_connectivity
        full_adj[:, 1:, 0] = cls_connectivity
        full_adj[:, 0, 0] = True

        attn_mask = ~full_adj
        attn_mask = attn_mask.repeat_interleave(self.nhead, dim=0)
        return attn_mask

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)

        assert x.dim() == 4, f"Expected 4D (B,C,H,W) input, got {tuple(x.shape)}"
        B, C, H, W = x.shape

        wall_map = x[:, self.wall_channel_idx, :, :]  # (B, H, W)  ※permute前に取得
        agent_map = x[:, 3, :, :]  # (B, H, W) エージェント位置チャンネル
        agent_pos_flat = agent_map.reshape(B, -1).argmax(dim=1)  # (B,)
        x = x.permute(0, 2, 3, 1).contiguous().reshape(B, H * W, C)  # (B, 25, 4)

        x = self.embedding(x)  # (B, 25, d_model)

        # --- 2次元位置エンコーディングの加算 ---
        pos_embedding = self._build_2d_pos_embedding()  # (1, 25, d_model)
        x = x + pos_embedding

        cls_tokens = self.cls_token.expand(B, -1, -1) + self.cls_pos_embedding
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 26, d_model)

        # 変更: 既定(use_wall_mask=False)ではフルの自己注意を使う。
        # 壁を挟んだ受容野の制限をなくし、かつobs[4]の距離マップから
        # グローバルな経路情報を各トークンが直接参照できるようにするため。
        if self.use_wall_mask:
            attn_mask = self._build_attn_mask(wall_map, agent_pos_flat, B, x.device)
            x = self.transformer(x, mask=attn_mask)
        else:
            x = self.transformer(x)

        cls_out = x[:, 0]
        logits = self.actor_head(cls_out)
        value = self.critic_head(cls_out)

        return logits, value.squeeze(-1)


# ================================================================
# 3. PPO Agent
# ================================================================
def _init_weights(m):
    """直交初期化（Orthogonal Initialization）の基本関数"""
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class TransformerPPOAgent:

    def __init__(
        self,
        env,
        in_channels=5,   # 変更: MazeEnvの距離チャンネル追加に合わせる
        grid_size=5,
        d_model=64,
        nhead=4,
        num_layers=2,
        action_dim=4,
        hidden_size=128,
        use_wall_mask=False,  # 追加: Trueにすると従来の壁トポロジーmaskを使う（非推奨・アブレーション用）
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        ppo_epochs=10,
        batch_size=32,
        entropy_coef=0.03,
        entropy_coef_final=None,  # 追加: 指定するとentropy_coefから線形に減衰させる（探索→活用）
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        path_save="transformer_ppo.pth",
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        # 追加: アニーリング用に開始値・終了値を保持。entropy_coef_final未指定なら
        # 従来通り定数（アニーリングなし）。
        self.entropy_coef_start = entropy_coef
        self.entropy_coef_final = (
            entropy_coef_final if entropy_coef_final is not None else entropy_coef
        )
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.path_save = path_save
        self.device = device

        # ネットワークの初期化
        self.policy = TransformerActorCritic(
            in_channels=in_channels,
            grid_size=grid_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            action_dim=action_dim,
            hidden_size=hidden_size,
            use_wall_mask=use_wall_mask,
        )
        self.policy.to(device)

        # -------------------------------------------------------------
        # 【コツ1】直交初期化と出力層ゲインの調整
        # -------------------------------------------------------------
        self.policy.apply(_init_weights)
        self._apply_head_gain_initialization()

        # 修正: 以前はここに来る前に無条件で self.load_model(path_save) を呼んでおり、
        # チェックポイントファイルが存在しない初回実行時に torch.load が
        # FileNotFoundError で落ちる可能性があった。os.path.exists 判定後の
        # 1箇所にまとめる。
        if os.path.exists(self.path_save):
            self.load_model(self.path_save)
            print("Successfully loaded model.")

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)
        self.buffer = PPORolloutBuffer()

    def _apply_head_gain_initialization(self):
        """Actorヘッド（0.01）とCriticヘッド（1.0）のゲインを個別に調整"""
        # ネットワーク構造に合わせてヘッド属性名を取得（一般的な命名に対応）
        actor_head = getattr(
            self.policy, "actor", getattr(self.policy, "action_head", None)
        )
        critic_head = getattr(
            self.policy, "critic", getattr(self.policy, "value_head", None)
        )

        if actor_head is not None:
            last_layer = (
                actor_head[-1]
                if isinstance(actor_head, nn.Sequential)
                else actor_head
            )
            if hasattr(last_layer, "weight"):
                nn.init.orthogonal_(last_layer.weight, gain=0.01)

        if critic_head is not None:
            last_layer = (
                critic_head[-1]
                if isinstance(critic_head, nn.Sequential)
                else critic_head
            )
            if hasattr(last_layer, "weight"):
                nn.init.orthogonal_(last_layer.weight, gain=1.0)

    def _preprocess_state(self, state_img):
        if isinstance(state_img, np.ndarray):
            state_img = torch.from_numpy(state_img).float()
        # バッチ次元（Dim 0）がない場合は追加
        if state_img.dim() == 3:
            state_img = state_img.unsqueeze(0)
        return state_img.to(self.device)

    def select_action(self, state_img):
        self.policy.eval()
        with torch.no_grad():
            state_tensor = self._preprocess_state(state_img)
            logits, value = self.policy(state_tensor)

            # ナンバー・オーバーフロー保護
            logits = torch.clamp(logits, min=-20.0, max=20.0)

            # 追加: 壁・盤外に向かう行動をマスクしてから分布を作る
            valid_mask = compute_valid_action_mask(state_tensor)
            logits = apply_action_mask(logits, valid_mask)

            dist = Categorical(logits=logits)

            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.squeeze().item()

    def update(self, next_state_img, done):
        self.policy.train()

        # Bootstrap用の次状態価値の取得
        with torch.no_grad():
            next_state_tensor = self._preprocess_state(next_state_img)
            _, next_value = self.policy(next_state_tensor)
            next_value = next_value.squeeze().item() if not done else 0.0

        rewards = self.buffer.rewards
        dones = self.buffer.dones
        values = self.buffer.values + [next_value]

        # -------------------------------------------------------------
        # GAE (Generalized Advantage Estimation) の過去遡及計算
        # -------------------------------------------------------------
        advantages = []
        gae = 0.0
        for t in reversed(range(len(rewards))):
            non_terminal = 1.0 - float(dones[t])
            delta = (
                rewards[t]
                + self.gamma * values[t + 1] * non_terminal
                - values[t]
            )
            gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, values[:-1])]

        # テンソルへの変換
        states_tensor = torch.cat(
            [self._preprocess_state(s) for s in self.buffer.states], dim=0
        )
        actions_tensor = torch.tensor(
            self.buffer.actions, dtype=torch.long
        ).to(self.device)
        old_log_probs_tensor = torch.tensor(
            self.buffer.log_probs, dtype=torch.float
        ).to(self.device)
        old_values_tensor = torch.tensor(
            self.buffer.values, dtype=torch.float
        ).to(self.device)
        advantages_tensor = torch.tensor(
            advantages, dtype=torch.float
        ).to(self.device)
        returns_tensor = torch.tensor(returns, dtype=torch.float).to(
            self.device
        )

        # -------------------------------------------------------------
        # 【コツ2-2】アドバンテージ標準化（Advantage Normalization）
        # -------------------------------------------------------------
        if len(advantages_tensor) > 1:
            adv_std = advantages_tensor.std()
            if not torch.isnan(adv_std) and adv_std > 1e-8:
                advantages_tensor = (
                    advantages_tensor - advantages_tensor.mean()
                ) / (adv_std + 1e-8)
            else:
                advantages_tensor = (
                    advantages_tensor - advantages_tensor.mean()
                )
        else:
            advantages_tensor = advantages_tensor - advantages_tensor.mean()

        dataset_size = len(self.buffer.states)

        # PPOエポックループ
        for _ in range(self.ppo_epochs):
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)

            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                b_states = states_tensor[batch_idx]
                b_actions = actions_tensor[batch_idx]
                b_old_log_probs = old_log_probs_tensor[batch_idx]
                b_old_values = old_values_tensor[batch_idx]
                b_advantages = advantages_tensor[batch_idx]
                b_returns = returns_tensor[batch_idx]

                logits, new_values = self.policy(b_states)
                new_values = new_values.squeeze(-1)

                # Logits の数値的安定化
                logits = torch.clamp(logits, min=-20.0, max=20.0)

                # 追加: ロールアウト時（select_action）と同じマスクをここでも適用しないと、
                # 保存済みold_log_probs（マスクあり）と学習時のlog_probs（マスクなし）が
                # 食い違い、PPOのratio計算が歪む。
                valid_mask = compute_valid_action_mask(b_states)
                logits = apply_action_mask(logits, valid_mask)

                dist = Categorical(logits=logits)

                new_log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                # Ratio と Clipped Surrogate Loss の計算
                log_ratio = new_log_probs - b_old_log_probs
                log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
                ratios = torch.exp(log_ratio)

                surr1 = ratios * b_advantages
                surr2 = (
                    torch.clamp(
                        ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps
                    )
                    * b_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # -------------------------------------------------------------
                # 【コツ1-1】Value Function（Critic）のクリッピング Loss
                # -------------------------------------------------------------
                v_clipped = b_old_values + torch.clamp(
                    new_values - b_old_values, -self.clip_eps, self.clip_eps
                )
                v_loss_unclipped = (new_values - b_returns) ** 2
                v_loss_clipped = (v_clipped - b_returns) ** 2
                value_loss = 0.5 * torch.max(
                    v_loss_unclipped, v_loss_clipped
                ).mean()

                # トータル Loss
                loss = (
                    policy_loss
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy
                )

                # 数値異常のガード
                if torch.isnan(loss) or torch.isinf(loss):
                    print(
                        "Warning: Loss is NaN/Inf, skipping backward pass."
                    )
                    self.optimizer.zero_grad()
                    continue

                self.optimizer.zero_grad()
                loss.backward()

                # -------------------------------------------------------------
                # 【コツ3-2】勾配クリッピング
                # -------------------------------------------------------------
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), max_norm=self.max_grad_norm
                )
                self.optimizer.step()

        self.buffer.clear()

    def train(
        self,
        num_episodes=1000,
        max_steps_per_episode=100,
        update_horizon=512,
        maze_change=True,
        path=None,
        log_interval=10,
        use_curriculum=False,
        curriculum_start_distance=3,   # 追加: 最初はこの距離までの迷路だけ出す
        curriculum_success_threshold=0.7,  # 追加: 直近window内の成功率がこれを超えたら難易度UP
        curriculum_window=20,
    ):
        save_path = path if path is not None else self.path_save
        episode_rewards = []
        success_flags = []  # 直近のゴール到達可否を記録（診断用・カリキュラム判定用）

        # 追加: カリキュラム学習用の現在の難易度上限（スタート-ゴール間距離）
        current_max_distance = curriculum_start_distance if use_curriculum else None

        for episode in range(num_episodes):
            # 追加: エントロピー係数を線形にアニーリング（探索→活用へ）
            progress = episode / max(1, num_episodes - 1)
            self.entropy_coef = (
                self.entropy_coef_start
                + (self.entropy_coef_final - self.entropy_coef_start) * progress
            )

            if use_curriculum:
                state_pos = self.env.reset(
                    maze_change=maze_change, max_distance=current_max_distance
                )
            else:
                state_pos = self.env.reset(maze_change=maze_change)
            state_img = self.env.get_image_observation()
            episode_reward = 0
            episode_done = False
            next_state_img = state_img  # timeoutでstepが1回も回らない異常系向けの保険

            for step in range(max_steps_per_episode):
                action, log_prob, value = self.select_action(state_img)

                # step の戻り値形式（3要素/5要素）の柔軟な受け取り
                step_result = self.env.step(action)
                if len(step_result) == 5:
                    next_state_pos, reward, terminated, truncated, _ = (
                        step_result
                    )
                    done = terminated or truncated
                else:
                    next_state_pos, reward, done = step_result[:3]

                next_state_img = self.env.get_image_observation()

                self.buffer.push(
                    state_img, action, log_prob, reward, done, value
                )

                state_pos = next_state_pos
                state_img = next_state_img
                episode_reward += reward
                episode_done = done

                # horizon到達 or done(=ゴール到達)の場合はここで更新。
                if len(self.buffer.states) >= update_horizon or done:
                    self.update(next_state_img, done=done)

                if done:
                    break

            # timeoutで終わったエピソードもここで必ずupdateし、bufferを毎エピソードで
            # クリアする（迷路をまたいだGAE汚染を防ぐ）。
            if len(self.buffer.states) > 0:
                self.update(next_state_img, done=episode_done)

            episode_rewards.append(episode_reward)
            success_flags.append(1 if episode_done else 0)

            if episode % log_interval == 0:
                window = success_flags[-log_interval:]
                success_rate = sum(window) / len(window)
                curriculum_info = (
                    f", 難易度(max_distance)={current_max_distance}"
                    if use_curriculum else ""
                )
                print(
                    f"Episode {episode}, Reward: {episode_reward:.2f}, "
                    f"直近{len(window)}エピソードのゴール到達率: {success_rate:.0%}, "
                    f"entropy_coef={self.entropy_coef:.4f}{curriculum_info}"
                )

            # 追加: カリキュラムの難易度更新判定
            if use_curriculum and len(success_flags) >= curriculum_window:
                recent = success_flags[-curriculum_window:]
                recent_success_rate = sum(recent) / len(recent)
                if recent_success_rate >= curriculum_success_threshold:
                    current_max_distance += 1
                    print(
                        f"[curriculum] 直近{curriculum_window}エピソードの成功率"
                        f"{recent_success_rate:.0%} >= 閾値。"
                        f"難易度を引き上げ: max_distance={current_max_distance}"
                    )
                    success_flags = []  # 難易度が変わったら成功率をリセットして再計測

            if episode % 100 == 0 and episode > 0:
                print(f"Episode {episode}, save model to {save_path}")
                self.save_model(save_path)

        print("Training finished.")
        return episode_rewards

    def save_model(self, path):
        self.policy.cpu()
        torch.save(self.policy.state_dict(), path)
        self.policy.to(self.device)

    def load_model(self, path):
        # 修正: self.load_model_safely は存在しないメソッド名だったため
        # hasattr(self, "load_model_safely") は常にFalseとなり、
        # 「形状不一致レイヤーをスキップする安全ロード」が一度も
        # 使われていなかった。モジュール関数を直接呼ぶように修正。
        # これにより、今回のようにモデル構造（in_channels等）を変更した後でも、
        # 形状が一致するレイヤーだけを読み込んで継続学習できる。
        load_model_safely(self.policy, path, device=self.device)
