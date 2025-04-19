import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import os

class ModernBertClassifier(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.dir_current = os.path.dirname(__file__)
        self.dir_nn_parameter = self.dir_current + "/nn_parameters"
        self.tokenizer = AutoTokenizer.from_pretrained(self.dir_nn_parameter)
        num_labels = kwargs['num_labels']
        id2label = kwargs['id2label']
        label2id = kwargs['label2id']
        self.model = AutoModelForSequenceClassification.from_pretrained(self.dir_nn_parameter, num_labels=num_labels,
        id2label=id2label,
        label2id=label2id)
        self.max_length = 2048
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("setting is over")
        self.to(self.device)

    



if __name__ == "__main__":
    modern_bert = ModernBertClassifier()

