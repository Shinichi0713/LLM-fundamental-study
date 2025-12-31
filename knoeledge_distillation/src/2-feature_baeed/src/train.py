import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureDistillationTrainer(nn.Module):
    def __init__(self, teacher_model, student_model, teacher_dim, student_dim):
        super().__init__()
        self.teacher = teacher_model
        self.student = student_model
        
        # 教師モデルの固定
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Student -> Teacher の次元変換
        self.regressor = nn.Linear(student_dim, teacher_dim)
        
        # 2つの損失関数
        self.criterion_distill = nn.MSELoss()
        self.criterion_mlm = nn.CrossEntropyLoss() # ignore_index=-100 がデフォルト

    def forward(self, input_ids, attention_mask=None, labels=None):
        # 1. Teacherの隠れ層取得
        with torch.no_grad():
            teacher_out = self.teacher(input_ids, attention_mask=attention_mask)
            teacher_hidden = teacher_out.hidden_states[-1]

        # 2. Studentの順伝播（隠れ層とロジットの両方を取得）
        student_out = self.student(input_ids, attention_mask=attention_mask)
        student_hidden = student_out.last_hidden_state
        logits = student_out.logits

        # 3. 特徴量蒸留損失 (MSE)
        projected_hidden = self.regressor(student_hidden)
        distill_loss = self.criterion_distill(projected_hidden, teacher_hidden)

        # 4. MLM損失 (Cross Entropy)
        # labelsが提供されている場合のみ計算
        if labels is not None:
            # logits: [batch_size, seq_len, vocab_size] -> [N, vocab_size]
            # labels: [batch_size, seq_len] -> [N]
            mlm_loss = self.criterion_mlm(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            # 合計損失 (重み付けを調整可能。ここでは 1:1)
            # 特徴量のスケールが小さい場合は、distill_loss に大きな係数（例: 100）をかけることもあります
            total_loss = distill_loss + mlm_loss
            return total_loss
        
        return distill_loss

# --- 使用イメージ ---
# teacher = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
# student = CustomSmallEncoder(dim=256, ...) # あなたが作成中のSparse RoPEモデルなど
# trainer = FeatureDistillationTrainer(teacher, student, teacher_dim=768, student_dim=256)

# optimizer = torch.optim.Adam(list(student.parameters()) + list(trainer.regressor.parameters()), lr=1e-4)