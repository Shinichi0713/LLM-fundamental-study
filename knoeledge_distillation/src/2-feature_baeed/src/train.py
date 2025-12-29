import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureDistillationTrainer(nn.Module):
    def __init__(self, teacher_model, student_model, teacher_dim, student_dim):
        super().__init__()
        self.teacher = teacher_model
        self.student = student_model
        
        # 教師を推論モードに（重み固定）
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        # 次元の不一致を埋めるための線形変換 (Student -> Teacherの次元へ)
        # これにより、Studentの中間出力をTeacherのサイズに合わせて比較可能にする
        self.regressor = nn.Linear(student_dim, teacher_dim)
        
        self.criterion = nn.MSELoss()

    def forward(self, input_ids, attention_mask=None):
        # 1. Teacherの中間層出力を取得
        with torch.no_grad():
            # 本来は特定の層を抽出しますが、ここでは最終層の前の出力を想定
            teacher_hidden = self.teacher(input_ids, attention_mask=attention_mask).hidden_states[-1]
        
        # 2. Studentの中間層出力を取得
        student_outputs = self.student(input_ids, attention_mask=attention_mask)
        student_hidden = student_outputs.hidden_states[-1]
        
        # 3. Studentの次元をTeacherに合わせる
        projected_student_hidden = self.regressor(student_hidden)
        
        # 4. 特徴量のMSE損失を計算
        distill_loss = self.criterion(projected_student_hidden, teacher_hidden)
        
        return distill_loss

# --- 使用イメージ ---
# teacher = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
# student = CustomSmallEncoder(dim=256, ...) # あなたが作成中のSparse RoPEモデルなど
# trainer = FeatureDistillationTrainer(teacher, student, teacher_dim=768, student_dim=256)

# optimizer = torch.optim.Adam(list(student.parameters()) + list(trainer.regressor.parameters()), lr=1e-4)