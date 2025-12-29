import torch.nn as nn
import torchvision.models as models

def create_light_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 教師: ResNet18 (ResNet50より圧倒的に軽い)
    teacher = models.resnet18(weights=None) 
    teacher.fc = nn.Linear(teacher.fc.in_features, 10)
    
    # 生徒: さらに軽い「カスタムCNN」 (MobileNetV2よりもさらにシンプルに)
    student = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 8 * 8, 10)
    )

    return teacher.to(device), student.to(device)

teacher_model, student_model = create_light_models()