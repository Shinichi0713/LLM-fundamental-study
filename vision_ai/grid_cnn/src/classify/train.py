import torch.optim as optim

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# モデル・損失関数・最適化手法の設定
model = SimpleGridCNN(grid_size=5, img_size=320).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# データローダーの準備
dataset = GridDataset(
    annotations_file="dataset/annotations.txt",
    img_dir="dataset",
    transform=transform
)

train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 学習ループ
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # 勾配の初期化
        optimizer.zero_grad()

        # 順伝播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 逆伝播
        loss.backward()
        optimizer.step()

        # 統計情報
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")