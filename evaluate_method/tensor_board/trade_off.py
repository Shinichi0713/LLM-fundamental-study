import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# 1. TensorBoardの準備
writer = SummaryWriter('runs/pr_curve_experiment')

# 2. ダミーデータの作成（例：100個のサンプル）
# 実際には、テストデータに対するモデルの推論結果を使用します
num_samples = 100

# 正解ラベル（0 または 1）
# 不均衡データをシミュレート（1が少なく、0が多い）
true_labels = torch.cat([torch.ones(20), torch.zeros(80)]) 

# モデルの予測スコア（0.0 〜 1.0 の確率値）
# 正解に近い値を予測できているケースを想定
predictions = torch.cat([
    torch.rand(20) * 0.5 + 0.5, # 正解が1のグループには高めのスコア (0.5~1.0)
    torch.rand(80) * 0.7        # 正解が0のグループには低めのスコア (0.0~0.7)
])

# 3. add_pr_curve で記録
# tag: グラフの名前
# labels: 正解ラベル (0 or 1)
# predictions: モデルが出力した確率値
writer.add_pr_curve('PR_Curve/Binary_Classification', true_labels, predictions, global_step=0)

# 4. 複数エポックをシミュレーション（学習が進んで精度が上がる様子を記録）
for epoch in range(1, 5):
    # 学習が進むにつれ、予測精度が向上していくダミーデータ
    improved_preds = torch.cat([
        torch.rand(20) * 0.3 + 0.7, # スコアがより1に寄る
        torch.rand(80) * 0.4        # スコアがより0に寄る
    ])
    writer.add_pr_curve('PR_Curve/Binary_Classification', true_labels, improved_preds, global_step=epoch)

writer.close()
print("PR Curve has been logged. Run 'tensorboard --logdir=runs' to view.")