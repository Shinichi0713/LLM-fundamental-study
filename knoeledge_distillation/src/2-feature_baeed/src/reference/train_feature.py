import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureDistillationTrainer(nn.Module):
    def __init__(self, teacher_model, student_model, teacher_dim, student_dim):
        super().__init__()
        self.teacher = teacher_model
        self.student = student_model
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.regressor = nn.Linear(student_dim, teacher_dim)
        self.criterion_mse = nn.MSELoss()
        # MLM用の損失関数を追加
        self.criterion_mlm = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, labels=None):
        # 1. Teacherの特徴量取得
        with torch.no_grad():
            teacher_hidden = self.teacher(input_ids, attention_mask=attention_mask).hidden_states[-1]

        # 2. Studentの出力取得 (MLMヘッドがある前提)
        student_outputs = self.student(input_ids, attention_mask=attention_mask, labels=labels)
        student_hidden = student_outputs.hidden_states[-1]
        
        # 3. 特徴量蒸留損失 (MSE)
        projected_student_hidden = self.regressor(student_hidden)
        distill_loss = self.criterion_mse(projected_student_hidden, teacher_hidden)

        # 4. MLM損失 (単語予測の学習)
        # student_outputs.logits が [Batch, Seq, Vocab] の形状であること
        logits = student_outputs.logits
        
        if labels is not None:
            # -100 (無視対象) 以外でCrossEntropyを計算
            mlm_loss = self.criterion_mlm(logits.view(-1, logits.size(-1)), labels.view(-1))
            # 合計損失 (重み付けは 1:1 などで調整)
            total_loss = distill_loss + mlm_loss
            return total_loss, logits
        
        return distill_loss, logits

# --- 使用イメージ ---
# teacher = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
# student = CustomSmallEncoder(dim=256, ...) # あなたが作成中のSparse RoPEモデルなど
# trainer = FeatureDistillationTrainer(teacher, student, teacher_dim=768, student_dim=256)

# optimizer = torch.optim.Adam(list(student.parameters()) + list(trainer.regressor.parameters()), lr=1e-4)