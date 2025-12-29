import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def prepare_cifar10_data(batch_size=64):
    # 前処理: CIFAR-10は32x32だが、ResNet等のTeacherに合わせてリサイズ
    transform = transforms.Compose([
        transforms.Resize(224), # Teacherモデルの入力サイズに合わせる
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # ダウンロードと読み込み (download=True で自動取得)
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    
    classes = train_set.classes
    return train_loader, test_loader, classes

# 実行
train_loader, test_loader, classes = prepare_cifar10_data()
print(f"クラス一覧: {classes}")