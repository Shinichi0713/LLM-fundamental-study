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

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random

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

# 変更前 (forward層のイメージ)
# x = self.transformer(x)
# cls_out = x[:, 0, :] # CLSトークンだけを抜く (表現力の限界)
# return self.mlp(cls_out)

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

    def act(self):
        if len(self.buffer) < self.batch_size:
            return
        
        # バッチサンプリング
        state, action, reward, state_next, done = self.buffer.sample(self.batch_size)

        # 画像データをTensorに変換
        state = torch.FloatTensor(np.stack(state)).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        state_next = torch.FloatTensor(np.stack(state_next)).to(self.device)
        done = torch.BoolTensor(done).to(self.device)

        # 現在のQ値
        q_value = self.q_net(state)
        q_value = q_value.gather(1, action.unsqueeze(1)).squeeze(1)

        # DDQN 
        action_next = self.q_net(state_next).argmax(1)
        next_q_values = self.target_net(state_next)
        next_q_value = next_q_values.gather(1, action_next.unsqueeze(1)).squeeze(1)

        # ターゲット
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

# 変更後 (Flatten方式)
class TransformerQNetwork(nn.Module):
    def __init__(self, in_channels=4, grid_size=5, d_model=64, nhead=4, num_layers=2, action_dim=4, hidden_size=128):
        super().__init__()
        self.grid_size = grid_size
        self.num_tokens = grid_size * grid_size

        self.embedding = nn.Linear(in_channels, d_model)
        
        # ★CLSトークンは廃止し、位置エンコーディングは元のトークン数分だけ用意
        # self.pos_embedding = nn.Parameter(torch.randn(1, self.num_tokens, d_model))
        pos_emb = self.create_2d_sin_cos_pos_embedding(grid_size, d_model)
        self.register_buffer('pos_embedding', pos_emb)  # これで自動的にデバイス（GPU/CPU）移動が管理されます

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

    def create_2d_sin_cos_pos_embedding(self, grid_size, d_model):
        """
        グリッドに対して、固定の2D Sin-Cos 位置エンコーディングを生成する。
        """
        assert d_model % 4 == 0, "d_model は4の倍数である必要があります"
        pe = torch.zeros(grid_size, grid_size, d_model)
        d_axis = d_model // 2
        div_term = torch.exp(torch.arange(0, d_axis, 2).float() * -(np.log(10000.0) / d_axis))
        
        for y in range(grid_size):
            for x in range(grid_size):
                pe[y, x, 0:d_axis:2] = torch.sin(torch.tensor(x).float() * div_term)
                pe[y, x, 1:d_axis:2] = torch.cos(torch.tensor(x).float() * div_term)
                pe[y, x, d_axis::2]  = torch.sin(torch.tensor(y).float() * div_term)
                pe[y, x, d_axis+1::2] = torch.cos(torch.tensor(y).float() * div_term)
                
        # (1, num_tokens, d_model) に平坦化して返す
        return pe.view(1, grid_size * grid_size, d_model)

    def forward(self, x):
        # x の形状: (Batch, num_tokens, in_channels)
        x = self.embedding(x)
        x = x + self.pos_embedding
        
        x = self.transformer(x)
        
        # ★全マスの関係性を保持したまま平坦化してMLPへ送る
        x = x.flatten(start_dim=1) 
        return self.mlp(x)
    

class MLPExpert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)
    

class MoELayer(nn.Module):
    def __init__(self, d_module, d_ff, num_experts=4):
        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [MLPExpert(d_module, d_ff) for _ in range(num_experts)]
        )
        # 2. どのトークンをどの専門家に送るかを決めるゲート（線形層）
        self.gate = nn.Linear(d_module, num_experts)

    def forward(self, x):
        # x: (Batch, Tokens, d_model)
        batch_size, num_tokens, d_model = x.shape
        
        # トークンごとに独立してルーティングを決定するため、バッチとトークンをフラットにする
        flat_x = x.view(-1, d_model) # (Batch * Tokens, d_model)
        
        # 各トークンに対する各エキスパートの推薦スコアを計算
        gate_logits = self.gate(flat_x) # (Batch * Tokens, num_experts)
        
        # 最もスコアの高いエキスパートのインデックスと、その重み（Softmax値）を取得 (Top-1)
        gate_weights = F.softmax(gate_logits, dim=-1)
        max_weights, expert_indices = torch.max(gate_weights, dim=-1, keepdim=True) # (Batch * Tokens, 1)
        
        # 出力を格納するゼロテンソルを準備
        flat_out = torch.zeros_like(flat_x)
        
        # 各エキスパートごとに、自分にアサインされたトークンだけをまとめて一括処理（効率化）
        for i in range(self.num_experts):
            mask = (expert_indices == i).squeeze(-1) # このエキスパートが担当するトークンのマスク
            if mask.any():
                # 担当トークンを専門家に通し、ゲートの重みを掛け算して出力に加算
                expert_in = flat_x[mask]
                expert_out = self.experts[i](expert_in)
                flat_out[mask] = expert_out * max_weights[mask]
                
        # 元の形状 (Batch, Tokens, d_model) に戻して返す
        return flat_out.view(batch_size, num_tokens, d_model)


class MoETransformerEncoderLayer(nn.Module):
    """FFN部分をMoEに置き換えたTransformerレイヤー"""
    def __init__(self, d_model, nhead, d_ff, num_experts=4):
        super().__init__()
        # Pre-LN 構成
        self.attn_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        
        self.moe_norm = nn.LayerNorm(d_model)
        self.moe = MoELayer(d_model, d_ff, num_experts=num_experts)
        
    def forward(self, x):
        # 1. Multi-Head Attention (Pre-LN & Residual)
        norm_x = self.attn_norm(x)
        attn_out, _ = self.self_attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        
        # 2. MoE FFN (Pre-LN & Residual)
        norm_x2 = self.moe_norm(x)
        moe_out = self.moe(norm_x2)
        x = x + moe_out
        
        return x
    


"""
DDQN（Double DQN）は、DQN が抱える「Q値の過大評価（overestimation）」という問題を軽減するために提案された改良版です。

主な効果は次の2点です。

---

### 1. Q値の過大評価を抑える

DQN では、行動選択（どの行動を取るか）と Q値の評価（その行動の価値をどう見積もるか）に同じネットワークを使います。  
そのため、**最大値を取るステップ（max 操作）** が入ると、たまたま高く推定された行動が選ばれやすくなり、Q値が実際より大きく見積もられてしまう「過大評価バイアス」が生じます。

DDQN では、  
- **行動選択用のネットワーク**（オンライン側）  
- **Q値評価用のネットワーク**（ターゲット側）  

を分け、  
「オンライン側で選んだ行動」に対して「ターゲット側でその行動の Q値を評価」する、という二段構えにします。  
これにより、**行動選択と評価が独立に行われる**ため、特定の行動に対する一時的な過大評価が学習全体に波及しにくくなり、Q値の推定がより正確になります[ダブルDQNとは](https://e-words.jp/w/%E3%83%80%E3%83%96%E3%83%ABDQN.html)。

---

### 2. 学習の安定化と性能向上

Q値の過大評価が抑えられると、  
- 実際には良くない行動でも「良さそうに見える」状態が減る  
- その結果、**学習が安定しやすく、最終的な性能も向上しやすい**  

ことが、Atari などのベンチマークで確認されています[Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)。

DQN に比べると、  
- 計算コストはほぼ同じ（ネットワークは2つあるが、更新頻度や構造はほぼ同等）  
- 実装も DQN のターゲットネットワークを少し使い方を変えるだけ  

なので、**手軽に導入できる割に、性能改善の効果が大きい**のが特徴です。

---

### まとめ：DQN と DDQN の違い

| 項目 | DQN | DDQN |
|------|-----|------|
| ネットワークの使い方 | 行動選択と評価に同じネットワークを使う | 行動選択と評価を別々のネットワークで行う |
| Q値の推定 | 過大評価バイアスが大きい | 過大評価を抑え、より正確 |
| 学習の安定性 | 不安定になりやすい場面がある | 比較的安定しやすい |
| 実装の複雑さ | 標準的な DQN | DQN のターゲットネットワークの使い方を少し変えるだけ |

**結論として**、DDQN は DQN に対して  
- Q値の過大評価を抑える  
- その結果、学習が安定し、性能が向上しやすい  

という効果があります。  
DQN を実装する際には、ほぼ追加コストなしで導入できるため、**ほぼ標準的な改良手法**として使われることが多いです。
"""


