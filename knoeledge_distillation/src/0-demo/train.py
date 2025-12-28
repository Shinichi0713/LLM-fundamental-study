import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 1. 教師モデル（Teacher）: 少し深めのネットワーク
class TeacherModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    def forward(self, x):
        return self.fc(x.view(-1, 28 * 28))

# 2. 生徒モデル（Student）: かなり軽量なネットワーク
class StudentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(28 * 28, 64), # パラメータ数が圧倒的に少ない
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        return self.fc(x.view(-1, 28 * 28))

# 3. 知識蒸留用の損失関数
def distillation_loss(student_logits, teacher_logits, labels, T=2.0, alpha=0.5):
    """
    T: Temperature (確率分布を滑らかにするパラメータ)
    alpha: Student Loss と Distillation Loss の重みバランス
    """
    # 蒸留ロス: 教師と生徒の確率分布の差 (KLダイバージェンス)
    # T^2 を掛けるのは、勾配のスケールを合わせるため
    soft_loss = nn.KLDivLoss(reduction='batchmean')(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1)
    ) * (T * T)

    # 通常のロス: 正解ラベルとの差
    hard_loss = F.cross_entropy(student_logits, labels)

    return alpha * soft_loss + (1. - alpha) * hard_loss

# 4. 学習のセットアップ
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

teacher = TeacherModel().to(device) # 事前に学習済みであると想定
student = StudentModel().to(device)
optimizer = optim.Adam(student.parameters(), lr=1e-3)

# ダミーデータでの学習ループ例
def train_step(data, target):
    teacher.eval() # 教師は推論モード（勾配不要）
    student.train()
    
    optimizer.zero_grad()
    
    with torch.no_grad():
        teacher_logits = teacher(data)
    
    student_logits = student(data)
    
    # ロス計算
    loss = distillation_loss(student_logits, teacher_logits, target, T=3.0, alpha=0.7)
    
    loss.backward()
    optimizer.step()
    return loss.item()

print("Distillation setup complete.")