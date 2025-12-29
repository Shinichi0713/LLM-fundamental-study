import torch
import matplotlib.pyplot as plt
import numpy as np

def run_inference_on_test_data(model, loader, classes, num_images=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. モデルを評価モードに設定
    model.eval()
    
    # 2. テストデータから1バッチ分を取得
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)
    
    # 3. 推論実行
    with torch.no_grad():
        outputs = model(images)
        # 確率（信頼度）を算出
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        # 最も確率の高いクラスを取得
        confidences, predicted = torch.max(probabilities, 1)

    # 4. 可視化のためにCPUに移動
    images = images.cpu()
    labels = labels.cpu()
    predicted = predicted.cpu()
    confidences = confidences.cpu()

    # 5. 表示
    plt.figure(figsize=(18, 10))
    for i in range(num_images):
        plt.subplot(2, 5, i + 1)
        
        # 正規化を解除して画像を表示形式に戻す
        img = images[i].numpy().transpose((1, 2, 0))
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2023, 0.1994, 0.2010])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        plt.imshow(img)
        
        # 判定が合っているか
        is_correct = (predicted[i] == labels[i])
        color = 'green' if is_correct else 'red'
        
        # タイトルに予測と正解、信頼度を表示
        title_text = f"Pred: {classes[predicted[i]]}\nActual: {classes[labels[i]]}\nConf: {confidences[i]*100:.1f}%"
        plt.title(title_text, color=color, fontsize=10)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# --- 実行 ---
# 前述のコードで定義した student_model と test_loader, classes を使用
run_inference_on_test_data(student_model, test_loader, classes)