import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------
# 1. 簡易 LLM (Decoder-Only Causal Transformer) の定義
# ----------------------------------------------------------------------
class MiniLLM(nn.Module):
    def __init__(self, vocab_size=1000, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(512, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            batch_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        # input_ids: [batch_size, seq_len]
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)
        
        x = self.embedding(input_ids) + self.pos_embedding(positions)
        
        # Causal Mask (未来のトークンを見ないためのマスク)
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=input_ids.device), diagonal=1)
        
        # Attention Mask の統合 (1: 有効, 0: パディング)
        padding_mask = (attention_mask == 0) # PyTorch では True がマスク対象
        
        # トランスフォーマーの実行
        out = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        logits = self.lm_head(out) # [batch_size, seq_len, vocab_size]
        return logits


# ----------------------------------------------------------------------
# 2. モデルのインスタンス化と ONNX エクスポート
# ----------------------------------------------------------------------
if __name__ == "__main__":
    vocab_size = 1000
    model = MiniLLM(vocab_size=vocab_size)
    model.eval()

    # ダミー入力の作成 (Batch Size: 1, Sequence Length: 8)
    dummy_input_ids = torch.randint(0, vocab_size, (1, 8), dtype=torch.int64)
    dummy_attention_mask = torch.ones((1, 8), dtype=torch.int64)

    onnx_file_path = "mini_llm.onnx"

    # ONNX へエクスポート
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        onnx_file_path,
        export_params=True,
        opset_version=14, # 最新のオペレータセット
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        # C++側で任意のバッチサイズ・系列長に対応できるよう動的軸を設定
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size", 1: "seq_len"}
        }
    )

    print(f"ONNX モデルの保存に成功しました: {onnx_file_path}")