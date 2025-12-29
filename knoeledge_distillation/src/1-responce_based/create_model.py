import torch
import torch.nn as nn
import torchvision.models as models

def create_models(num_classes=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. 教師モデル (ResNet50) ---
    # ImageNetで事前学習済みの重みをロード
    teacher = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # 最終層をCIFAR-10の10クラスに変更
    teacher.fc = nn.Linear(teacher.fc.in_features, num_classes)
    
    # 【重要】教師モデルはすでにCIFAR-10で学習済みである必要がありますが、
    # 今回は構造の準備としてロードします。
    teacher = teacher.to(device)
    teacher.eval() # 教師は常に評価モード（推論のみ）

    # --- 2. 生徒モデル (MobileNetV2) ---
    # 軽量モデルを選択。重みはNone（一から学習）またはImageNet初期値を選択
    student = models.mobilenet_v2(weights=None) # 蒸留の効果を見るため未学習から開始
    
    # 最終層を10クラスに変更
    student.classifier[1] = nn.Linear(student.last_channel, num_classes)
    student = student.to(device)

    return teacher, student

# モデルの生成
teacher_model, student_model = create_models(num_classes=10)

print("モデルの準備が完了しました。")
print(f"Teacher (ResNet50) params: {sum(p.numel() for p in teacher_model.parameters()):,}")
print(f"Student (MobileNetV2) params: {sum(p.numel() for p in student_model.parameters()):,}")