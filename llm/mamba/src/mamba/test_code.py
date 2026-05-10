def check_selective_ssm():
    # パラメータ設定
    batch_size = 2
    seq_len = 10
    d_model = 16
    d_state = 8
    
    # モデル初期化
    model = SelectiveSSM(d_model=d_model, d_state=d_state, is_causal=True)
    model.eval()
    
    # 入力データ作成 (Batch, SeqLen, d_model)
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 1. 順伝播テスト
    try:
        with torch.no_grad():
            output = model(x)
        print(f"Success: Forward pass completed.")
        print(f"Input shape:  {x.shape}")
        print(f"Output shape: {output.shape}")
        assert x.shape == output.shape, "Shape mismatch between input and output"
    except Exception as e:
        print(f"Error during forward pass: {e}")
        return

    # 2. 因果性（Causality）テスト
    # 特定のトークン(index=5)を変更したとき、それより前のトークンに影響がないか確認
    x_mod = x.clone()
    x_mod[:, 5, :] += 1.0  # index 5を書き換え
    
    with torch.no_grad():
        out_orig = model(x)
        out_mod = model(x_mod)
    
    # 差分を計算
    diff = (out_orig - out_mod).abs().sum(dim=-1)[0] # 最初のバッチ、各時刻の差分合計
    
    print("\n--- Causality Check ---")
    is_causal = True
    for i in range(seq_len):
        if i < 5 and diff[i] > 1e-5:
            print(f"Step {i}: Affected! (FAILED)")
            is_causal = False
        elif i >= 5:
            print(f"Step {i}: Changed (OK)")
        else:
            print(f"Step {i}: No Change (OK)")
            
    if is_causal:
        print("\nRESULT: Causality is maintained. (Mamba style verified)")
    else:
        print("\nRESULT: Causality broken. Future info leaked to the past.")