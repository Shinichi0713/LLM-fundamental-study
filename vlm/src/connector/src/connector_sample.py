# 1. 既存のモデルをロード
vision_model = load_pretrained_vision_model() # 例: CLIP
llm = load_pretrained_llm() # 例: Llama-3

# 2. パラメータを固定（フリーズ）
for param in llm.parameters():
    param.requires_grad = False

# 3. 接続層（Connector）のみを定義
# ここが「新しく作成するビジョンAI部分」の核になります
connector = nn.Sequential(
    nn.Linear(vision_model.config.hidden_size, llm.config.hidden_size),
    nn.GELU(),
    nn.Linear(llm.config.hidden_size, llm.config.hidden_size)
)

# 4. 学習（ConnectorのパラメータだけをOptimizerに渡す）
optimizer = torch.optim.AdamW(connector.parameters(), lr=1e-4)

