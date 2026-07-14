import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque



class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, action_dim)
        )

    def forward(self, x):
        return self.net(x)
    

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
    


class DDQNAgent:
    def __init__(self, action_dim, lr=1e-3, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 buffer_capacity=10000, batch_size=32, target_update=100):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.update_count = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Qネットワークとターゲットネットワーク
        self.q_net = TransformerQNetwork(action_dim=action_dim).to(self.device)
        self.target_net = TransformerQNetwork(action_dim=action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)

    def _extract_agent_pos(self, state):
        """
        状態(C, H, W) からエージェントの位置(row, col)を特定するヘルパー関数
        ※ 仮にインデックス0のチャンネルがエージェント位置（エージェントがいるマスが1、他が0）を表現していると想定
        環境の仕様に合わせて必要に応じ変更してください。
        """
        agent_channel = state[0]
        pos = np.argwhere(agent_channel == 1)
        if len(pos) > 0:
            return pos[0]  # [row, col]
        else:
            # 見つからない場合のフォールバック（例: [0, 0]）
            return np.array([0, 0])

    def act(self, state, greedy=False):
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        # state からエージェントの位置を抽出
        agent_pos = self._extract_agent_pos(state)

        # Tensorに変換してバッチ次元を追加
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        pos_tensor = torch.FloatTensor(agent_pos).to(self.device).unsqueeze(0)

        with torch.no_grad():
            # 新しいネットワーク引数（x, agent_pos）に対応
            q_values = self.q_net(state_tensor, pos_tensor)
        return q_values.argmax().item()

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        # バッチサンプリング
        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)

        # 各バッチデータからエージェントの位置を抽出
        agent_pos_batch = np.array([self._extract_agent_pos(s) for s in state])
        next_agent_pos_batch = np.array([self._extract_agent_pos(ns) for ns in next_state])

        # 各種データをTensorに変換
        state = torch.FloatTensor(np.stack(state)).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(np.stack(next_state)).to(self.device)
        done = torch.BoolTensor(done).to(self.device)

        # エージェント位置もテンソル化してデバイスへ送る
        agent_pos = torch.FloatTensor(agent_pos_batch).to(self.device)
        next_agent_pos = torch.FloatTensor(next_agent_pos_batch).to(self.device)

        # 現在のQ値
        q_values = self.q_net(state, agent_pos)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        # Double DQN: 次状態での行動選択は q_net、評価は target_net
        next_actions = self.q_net(next_state, next_agent_pos).argmax(1)
        next_q_values = self.target_net(next_state, next_agent_pos)
        next_q_value = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)

        # ターゲット値
        target = reward + self.gamma * next_q_value * (~done)

        # 損失計算と更新
        loss = nn.MSELoss()(q_value, target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # εの減衰
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # ターゲットネットワークの更新
        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.q_net.state_dict())


class TransformerQNetwork(nn.Module):
    def __init__(self, in_channels=4, grid_size=5, d_model=64, nhead=4, num_layers=2, action_dim=4, hidden_size=128):
        super().__init__()
        self.grid_size = grid_size
        # モデルのトークン数はgridを網羅する
        self.num_tokens = grid_size * grid_size

        # 1. トークン埋め込み
        # ここで各チャネルの意味合いを認識。多分場所情報のみを保持
        self.embedding = nn.Linear(in_channels, d_model)

        # 2. CLSトークンの追加 (空間情報を平均で潰さないため)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # 3. 位置エンコーディング (CLSトークン分を含めて +1)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_tokens + 1, d_model))

        # 4. Transformer Encoder (norm_first=True で Pre-LN に変更)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            activation="gelu", batch_first=True, norm_first=True  # <- 重要！
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 5. 出力MLP
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, action_dim)
        )

        # 最終出力層の初期化を小さくする
        with torch.no_grad():
            self.mlp[-1].weight.fill_(0.0)
            self.mlp[-1].bias.fill_(0.0)



# 変更前 (forward層のイメージ)
# x = self.transformer(x)
# cls_out = x[:, 0, :] # CLSトークンだけを抜く (表現力の限界)
# return self.mlp(cls_out)

# 変更後 (Flatten方式)
class TransformerQNetwork(nn.Module):
    def __init__(self, in_channels=4, grid_size=5, d_model=64, nhead=4, num_layers=2, action_dim=4, hidden_size=128):
        super().__init__()
        self.grid_size = grid_size
        self.num_tokens = grid_size * grid_size

        self.embedding = nn.Linear(in_channels, d_model)
        
        # ★CLSトークンは廃止し、位置エンコーディングは元のトークン数分だけ用意
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_tokens, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ★全トークンを結合するため、入力サイズは (num_tokens * d_model) になる
        self.mlp = nn.Sequential(
            nn.Linear(self.num_tokens * d_model, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, action_dim)
        )
        
        with torch.no_grad():
            self.mlp[-1].weight.fill_(0.0)
            self.mlp[-1].bias.fill_(0.0)

    def forward(self, x):
        # x の形状: (Batch, num_tokens, in_channels)
        x = self.embedding(x)
        x = x + self.pos_embedding
        
        x = self.transformer(x)
        
        # ★全マスの関係性を保持したまま平坦化してMLPへ送る
        x = x.flatten(start_dim=1) 
        return self.mlp(x)