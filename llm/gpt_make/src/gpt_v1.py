import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import random

# ====================================================================
# ハイパーパラメータ設定 (小規模モデル用)
# ====================================================================
VOCAB_SIZE = 50       # 語彙サイズ
SEQ_LEN = 12          # シーケンス長
EMBED_DIM = 64        # 埋め込み次元 (d_model)
NUM_HEADS = 4         # Attention Headの数
NUM_LAYERS = 3        # Transformer Decoderブロックの層数
FFN_HIDDEN_DIM = EMBED_DIM * 2 # FFNの隠れ層の次元

# MoE関連 (オプション)
USE_MOE = False       # MoEを使用するかどうか
NUM_EXPERTS = 4       # MoEのエキスパート数
TOP_K = 2             # MoEで活性化するエキスパート数
MOE_LOSS_COEF = 0.01  # ロードバランシング損失の係数

BATCH_SIZE = 4
LEARNING_RATE = 1e-4
EPOCHS = 100

# 特殊トークンID
PAD_TOKEN_ID = 0
CLS_TOKEN_ID = 1 # GPTでは通常不要だが、例示のため含める
MASK_TOKEN_ID = 2 # GPTでは通常不要だが、MLMのデータ生成を想定
BOS_TOKEN_ID = 3 # Begin Of Sequence (通常はこれが使われる)
EOS_TOKEN_ID = 4 # End Of Sequence

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ====================================================================
# 1. 基本的なTransformer要素の定義
# ====================================================================

# 1.1. Self-Attention (GPTはMasked Self-Attention)
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Q, K, V を計算し、ヘッドごとに分割
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention スコア計算: (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # マスク適用 (Decoder-only Transformerの核)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attention_weights = F.softmax(scores, dim=-1)
        
        # 加重平均 (Context Vector)
        context = torch.matmul(attention_weights, v)
        
        # ヘッドを結合し、最終出力
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_linear(context)
        return output

# 1.2. Feed-Forward Network (FFN)
class FFN(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(), # GELU活性化関数を使用
            nn.Linear(hidden_dim, embed_dim)
        )
    def forward(self, x):
        return self.net(x)

# 1.3. MoE Expert (MoE オプション時)
class Expert(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )
    def forward(self, x):
        return self.net(x)

# 1.4. MoELayer (MoE オプション時)
class MoELayer(nn.Module):
    def __init__(self, dim, num_experts, top_k, expert_hidden_dim=None):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        hidden_dim = expert_hidden_dim if expert_hidden_dim is not None else dim * 2
        
        self.experts = nn.ModuleList([Expert(dim, hidden_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(dim, num_experts)
        
    def forward(self, x):
        original_shape = x.shape
        x = x.view(-1, original_shape[-1])
        N_tokens = x.size(0)
        
        gate_logits = self.gate(x)
        gate_weights = F.softmax(gate_logits, dim=-1)
        
        top_k_weights, top_k_indices = torch.topk(gate_weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        final_output = torch.zeros_like(x)
        
        # ロードバランシング損失
        expert_usage_one_hot = F.one_hot(top_k_indices, num_classes=self.num_experts).sum(dim=1).float()
        expert_router_prob = gate_weights.sum(dim=0) / N_tokens
        expert_fraction_routed = expert_usage_one_hot.sum(dim=0) / N_tokens
        load_balancing_loss = (expert_router_prob * expert_fraction_routed).sum()
        
        for k in range(self.top_k):
            expert_index = top_k_indices[:, k]
            weight = top_k_weights[:, k]       
            
            for i in range(self.num_experts):
                mask = (expert_index == i) 
                if not mask.any():
                    continue
                expert_input = x[mask]
                expert_output = self.experts[i](expert_input)
                weighted_output = expert_output * weight[mask].unsqueeze(1)
                final_output[mask] += weighted_output

        final_output = final_output.view(original_shape)
        return final_output, load_balancing_loss

# 1.5. Decoderブロック (Transformer Layer)
class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_hidden_dim, use_moe=False, num_experts=None, top_k=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = SelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.use_moe = use_moe
        if use_moe:
            assert num_experts is not None and top_k is not None, "MoE config needed"
            self.ffn_or_moe = MoELayer(embed_dim, num_experts, top_k, expert_hidden_dim=ffn_hidden_dim)
        else:
            self.ffn_or_moe = FFN(embed_dim, ffn_hidden_dim)
            
    def forward(self, x, mask):
        # 残差接続とLayer Normalization
        # GPTでは Pre-LN (Attention/FFNの前にLN) が使われることが多い
        
        # Masked Self-Attention
        attn_output = self.attention(self.norm1(x), mask)
        x = x + attn_output # 残差接続
        
        # FFN または MoE Layer
        if self.use_moe:
            ffn_output, moe_aux_loss = self.ffn_or_moe(self.norm2(x))
            x = x + ffn_output # 残差接続
            return x, moe_aux_loss
        else:
            ffn_output = self.ffn_or_moe(self.norm2(x))
            x = x + ffn_output # 残差接続
            return x, None # MoEを使用しない場合は損失はNone

# ====================================================================
# 2. GPTモデル本体
# ====================================================================

class GPT(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_dim, num_heads, num_layers, ffn_hidden_dim,
                 use_moe=False, num_experts=None, top_k=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        
        # トークン埋め込み層
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_TOKEN_ID)
        # 位置埋め込み層
        self.position_embedding = nn.Embedding(seq_len, embed_dim)
        
        # Transformer Decoderブロックのスタック
        self.decoder_layers = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads, ffn_hidden_dim, use_moe, num_experts, top_k)
            for _ in range(num_layers)
        ])
        
        # 最終のLayer Normalization (出力前)
        self.final_norm = nn.LayerNorm(embed_dim)
        
        # 出力層 (語彙サイズへの線形変換)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        
        self.use_moe = use_moe

    def forward(self, input_ids):
        # input_ids: (batch_size, seq_len)
        
        batch_size, seq_len = input_ids.shape
        
        # トークン埋め込み
        token_embeds = self.token_embedding(input_ids)
        
        # 位置埋め込み (torch.arangeで位置IDを生成)
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        position_embeds = self.position_embedding(position_ids)
        
        # 埋め込みの合計
        x = token_embeds + position_embeds
        
        # マスクの作成 (未来のトークンを参照しないようにする)
        # causal_mask: (seq_len, seq_len) の下三角行列
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=input_ids.device)).bool()
        # パディングマスクはここでは考慮しない (MLMデータセットで対応)
        
        total_moe_aux_loss = 0.0
        
        # Decoderブロックを順に適用
        for layer in self.decoder_layers:
            output, moe_aux_loss = layer(x, causal_mask)
            x = output
            if self.use_moe and moe_aux_loss is not None:
                total_moe_aux_loss += moe_aux_loss
        
        # 最終Layer Normalization
        x = self.final_norm(x)
        
        # 言語モデルヘッド (logits)
        logits = self.lm_head(x) # (batch_size, seq_len, vocab_size)
        
        return logits, total_moe_aux_loss

# ====================================================================
# 3. データセットとデータローダーの模擬 (MLMタスクを想定)
# ====================================================================

class MockTokenizer:
    def __init__(self, vocab_size):
        self.vocab = {i: f"token_{i}" for i in range(vocab_size)}
        self.vocab[PAD_TOKEN_ID] = "<PAD>"
        self.vocab[BOS_TOKEN_ID] = "<BOS>" # BOSトークンを使用
        self.vocab[EOS_TOKEN_ID] = "<EOS>"
        self.vocab[MASK_TOKEN_ID] = "<MASK>"

        self.id_to_word = self.vocab
        self.bos_token_id = BOS_TOKEN_ID
        self.pad_token_id = PAD_TOKEN_ID
        self.mask_token_id = MASK_TOKEN_ID

    def decode(self, token_ids):
        return " ".join([self.id_to_word[i] for i in token_ids if i not in [self.pad_token_id, -100]])

class MockDataset(Dataset):
    def __init__(self, num_samples, seq_len, vocab_size, tokenizer):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # ダミーの入力シーケンスを生成 (BOSトークンで開始)
        # BOS, token_id, token_id, ..., PAD, PAD
        input_ids = torch.randint(low=EOS_TOKEN_ID + 1, high=self.vocab_size, size=(self.seq_len - 1,))
        input_ids = torch.cat([torch.tensor([self.tokenizer.bos_token_id]), input_ids])
        
        # GPTは通常、次単語予測 (Causal Language Modeling) のため、
        # 入力そのものがラベルとなる (ずらして比較)
        labels = input_ids.clone()
        
        # 例外: パディングは損失計算から除外
        labels[labels == self.tokenizer.pad_token_id] = -100 
        
        return {"input_ids": input_ids, "labels": labels}

# ====================================================================
# 4. モデルの初期化と訓練ループ
# ====================================================================

def train():
        
    # トークナイザとデータセットの準備
    tokenizer = MockTokenizer(VOCAB_SIZE)
    train_dataset = MockDataset(num_samples=100, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE, tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # GPTモデルの初期化
    gpt_model = GPT(
        vocab_size=VOCAB_SIZE,
        seq_len=SEQ_LEN,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ffn_hidden_dim=FFN_HIDDEN_DIM,
        use_moe=USE_MOE,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K
    ).to(device)

    # オプティマイザと損失関数
    optimizer = torch.optim.Adam(gpt_model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID) # PADトークンは損失計算から除外

    print(f"\nModel initialized with {sum(p.numel() for p in gpt_model.parameters()):,} parameters.")
    print(f"Using MoE: {USE_MOE}")
    print(f"Starting training for {EPOCHS} epochs...")

    for epoch in range(EPOCHS):
        gpt_model.train()
        total_loss = 0
        total_moe_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device) # Causal LMでは input_ids がずれてラベルとなる
            
            optimizer.zero_grad()
            
            logits, moe_aux_loss = gpt_model(input_ids)
            
            # 損失計算 (次単語予測):
            # logits: (batch_size, seq_len, vocab_size) -> (batch_size * seq_len, vocab_size)
            # labels: (batch_size, seq_len)           -> (batch_size * seq_len)
            loss = criterion(logits[:, :-1, :].reshape(-1, VOCAB_SIZE), labels[:, 1:].reshape(-1))
            
            if USE_MOE:
                # MoEのロードバランシング損失を加算
                loss += MOE_LOSS_COEF * moe_aux_loss
                total_moe_loss += moe_aux_loss.item()
                
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        
        if USE_MOE:
            avg_moe_loss = total_moe_loss / len(train_loader)
            print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f} (Main+MoE) | Avg MoE Aux Loss: {avg_moe_loss:.4f}")
        else:
            print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")

    print("\nTraining Finished!")

    # ====================================================================
    # 5. テキスト生成の例 (非常に単純な例)
    # ====================================================================

    gpt_model.eval()
    print("\n--- Text Generation Example ---")

    # シードテキスト (BOSトークンで開始)
    seed_text_ids = [tokenizer.bos_token_id, 5, 8] # 例: "<BOS> token_5 token_8"
    current_ids = torch.tensor([seed_text_ids], dtype=torch.long, device=device)

    max_new_tokens = 5
    generated_tokens = seed_text_ids[:]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # モデルの入力シーケンス長に合わせてトリミング (必要に応じて)
            input_for_model = current_ids[:, -SEQ_LEN:] # 最新のSEQ_LENトークンを使用
            
            logits, _ = gpt_model(input_for_model)
            
            # 最後のトークンのlogitsを取得
            next_token_logits = logits[:, -1, :]
            
            # サンプリングまたはargmaxで次のトークンを選択
            next_token_id = torch.argmax(next_token_logits, dim=-1).item()
            
            if next_token_id == EOS_TOKEN_ID: # EOSトークンが出たら終了
                break
                
            generated_tokens.append(next_token_id)
            current_ids = torch.cat([current_ids, torch.tensor([[next_token_id]], device=device)], dim=-1)

    decoded_generated_text = tokenizer.decode(generated_tokens)
    print(f"Seed: {tokenizer.decode(seed_text_ids)}")
    print(f"Generated: {decoded_generated_text}")

