import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # 特徴抽出部（Convolutional Base）
        # 入力: [Batch, 3, 32, 32] (例: CIFAR-10などのカラー画像)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1) 
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        # プーリング層（解像度を縦横半分にする）
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全結合分類部（Classifier）
        # 32x32 の画像が 2回のプーリングで 8x8 に縮小されるため、特徴量サイズは 32チャネル x 8 x 8
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # 1つ目の畳み込みブロック: Conv -> ReLU -> Pool
        # [3, 32, 32] -> [16, 32, 32] -> [16, 16, 16]
        x = self.pool(F.relu(self.conv1(x)))
        
        # 2つ目の畳み込みブロック: Conv -> ReLU -> Pool
        # [16, 16, 16] -> [32, 16, 16] -> [32, 8, 8]
        x = self.pool(F.relu(self.conv2(x)))
        
        # テンソルを1次元に平坦化（フラット化）
        # [Batch, 32, 8, 8] -> [Batch, 32 * 8 * 8]
        x = x.view(-1, 32 * 8 * 8)
        
        # 全結合層
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # 損失関数に nn.CrossEntropyLoss を使う場合は、最後は激活関数なしで出力
        
        return x

# --- 動作確認 ---
if __name__ == "__main__":
    # モデルのインスタンス化（10クラス分類）
    model = SimpleCNN(num_classes=10)
    print(model)

    # ダミー入力データの作成 (バッチサイズ: 4, チャネル: 3, 縦: 32, 横: 32)
    dummy_input = torch.randn(4, 3, 32, 32)
    
    # 順伝播の実行
    output = model(dummy_input)
    
    print(f"\n入力サイズ: {dummy_input.shape}")
    print(f"出力サイズ: {output.shape} (Batchサイズ, クラス数)")