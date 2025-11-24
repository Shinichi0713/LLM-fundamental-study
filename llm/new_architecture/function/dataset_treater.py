import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# データセットクラス
class LivedoorDataset(Dataset):
    """Livedoorコーパス用のデータセットクラス"""
    def __init__(self, texts, labels, tokenizer=None, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        if self.tokenizer:
            # トークン化済みデータを返す
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,  # 最大長
                padding="max_length",  # 最大長までパディング
                truncation=True,  # 最大長を超える場合は切り詰める
                return_tensors="pt"
            )
            
            # バッチ次元を削除（[1, n]　→　[n]）
            item = {key: val.squeeze(0) for key, val in encoding.items()}
             # ラベルをテンソルに変換
            item['labels'] = torch.tensor(label, dtype=torch.long)
            return item
        else:
            # 生テキストとラベルを返す
            return {"text": text, "label": label}
        

# 評価メトリクスの計算関数
def compute_metrics(eval_pred):
    """評価メトリクスを計算する関数"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1
    }

# 混同行列を可視化する関数
def plot_confusion_matrix(y_true, y_pred, class_names, title, accuracy, f1):
    """混同行列を可視化する関数"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', 
        xticklabels=class_names, 
        yticklabels=class_names
    )
    plt.title(f"{title} (正解率: {accuracy:.4f}, F1: {f1:.4f})")
    plt.ylabel('正解ラベル')
    plt.xlabel('予測ラベル')
    plt.tight_layout()
    plt.show()
    
    return cm