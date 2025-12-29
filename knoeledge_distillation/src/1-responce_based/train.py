import torch.nn.functional as F

# 知識蒸留用の損失関数（再掲・最適化版）
def distillation_loss_fn(student_logits, teacher_logits, labels, T=3.0, alpha=0.5):
    # 1. Distillation Loss (教師の「振る舞い」を真似る)
    # Tで割ることで確率分布を滑らかにし、小さな差異を強調する
    soft_targets = F.softmax(teacher_logits / T, dim=1)
    student_log_soft = F.log_softmax(student_logits / T, dim=1)
    distill_loss = nn.KLDivLoss(reduction='batchmean')(student_log_soft, soft_targets) * (T * T)

    # 2. Student Loss (「正解」を当てる)
    hard_loss = F.cross_entropy(student_logits, labels)

    # 二つのロスの加重平均
    return alpha * distill_loss + (1. - alpha) * hard_loss

def train_knowledge_distillation(teacher, student, train_loader, test_loader, epochs=10, T=3.0, alpha=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
    
    for epoch in range(epochs):
        student.train()
        teacher.eval() # 教師は常に評価モード
        
        train_loss = 0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # 教師モデルの予測（勾配計算不要）
            with torch.no_grad():
                teacher_logits = teacher(images)
            
            # 生徒モデルの予測
            student_logits = student(images)
            
            # ロスの計算
            loss = distillation_loss_fn(student_logits, teacher_logits, labels, T, alpha)
            
            # バックプロパゲーション
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = student_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # テスト精度検証
        val_acc = evaluate_model(student, test_loader, device)
        
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Loss: {train_loss/len(train_loader):.4f} | "
              f"Train Acc: {100.*correct/total:.2f}% | "
              f"Test Acc: {val_acc:.2f}%")

def evaluate_model(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total

# 実行
train_knowledge_distillation(teacher_model, student_model, train_loader, test_loader)