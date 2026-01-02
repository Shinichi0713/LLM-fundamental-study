import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureDistillationTrainer(nn.Module):
    def __init__(
        self,
        teacher,
        student,
        temperature=4.0,
        alpha_kd=0.5,
        alpha_hidden=1.0
    ):
        super().__init__()
        self.teacher = teacher.eval()
        self.student = student

        self.T = temperature
        self.alpha_kd = alpha_kd
        self.alpha_hidden = alpha_hidden

        # 教師 → 学生 次元射影
        self.proj = nn.Linear(
            teacher.config.hidden_size,
            student.dim
        )

        for p in self.teacher.parameters():
            p.requires_grad = False

    def forward(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        # ---- 教師 ----
        with torch.no_grad():
            t_out = self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

        # ---- 学生 ----
        s_out = self.student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # ======================
        # 1. Hard MLM loss
        # ======================
        loss_mlm = F.cross_entropy(
            s_out["logits"].view(-1, s_out["logits"].size(-1)),
            labels.view(-1),
            ignore_index=-100
        )

        # ======================
        # 2. Logits KD loss
        # ======================
        T = self.T
        kd_loss = F.kl_div(
            F.log_softmax(s_out["logits"] / T, dim=-1),
            F.softmax(t_out.logits / T, dim=-1),
            reduction="batchmean"
        ) * (T * T)

        # ======================
        # 3. Hidden state loss
        # ======================
        loss_hidden = 0.0

        t_hiddens = t_out.hidden_states[1:]  # embedding除外
        s_hiddens = s_out["hidden_states"]

        # 教師層を間引いて対応
        step = len(t_hiddens) // len(s_hiddens)

        for i, s_h in enumerate(s_hiddens):
            t_h = t_hiddens[i * step]
            t_h = self.proj(t_h.detach())
            loss_hidden += F.mse_loss(s_h, t_h)

        loss_hidden /= len(s_hiddens)

        # ======================
        # Total loss
        # ======================
        total_loss = (
            (1 - self.alpha_kd) * loss_mlm
            + self.alpha_kd * kd_loss
            + self.alpha_hidden * loss_hidden
        )

        return {
            "loss": total_loss,
            "loss_mlm": loss_mlm.detach(),
            "loss_kd": kd_loss.detach(),
            "loss_hidden": loss_hidden.detach()
        }


# --- 使用イメージ ---
trainer = FeatureDistillationTrainer(teacher, student, teacher_dim=768, student_dim=256)

optimizer = torch.optim.AdamW(
    trainer.student.parameters(),
    lr=3e-4,
    weight_decay=0.01
)