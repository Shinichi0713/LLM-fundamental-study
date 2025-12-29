import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# 1. モデルの定義とロード
def load_student_model(checkpoint_path, num_classes=100):
    # 学習時と同じ構造でモデルを初期化
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    
    # 保存した重みをロード (学習済みの場合)
    # model.load_state_dict(torch.load(checkpoint_path))
    
    model.eval() # 推論モードに設定
    return model

# 2. 画像の前処理 (学習時と同じ設定にする必要があります)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_face(image_path, model, class_names):
    # 画像を開いて前処理
    img = Image.open(image_path).convert('RGB')
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0) # バッチ次元の追加 [1, 3, 224, 224]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    img_tensor = img_tensor.to(device)

    # 推論
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1) # 確率に変換
        
    # 最大確率のクラスを取得
    conf, pred_idx = torch.max(probs, dim=1)
    
    result_label = class_names[pred_idx.item()]
    confidence_score = conf.item() * 100

    return result_label, confidence_score, img

# --- 実行例 ---
# 人物名のリスト（学習時のラベル順）
class_names = [f"Person_{i}" for i in range(100)] 

# モデルの準備 (重みファイルがある場合はパスを指定)
student = load_student_model("student_face_model.pth")

# テスト画像のパス
test_image = "test_face.jpg" 

# 推論実行
label, score, original_img = predict_face(test_image, student, class_names)

# 結果表示
plt.imshow(original_img)
plt.title(f"Prediction: {label} ({score:.2f}%)")
plt.axis('off')
plt.show()

print(f"この人物は {label} です。 (確信度: {score:.2f}%)")

