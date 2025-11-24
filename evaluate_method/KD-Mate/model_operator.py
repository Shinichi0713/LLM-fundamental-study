from sentence_transformers import models
from transformers import XLMRobertaTokenizer, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, model_card_templates, util, evaluation
from sentence_transformers import ParallelSentencesDataset
import os
from typing import List, Dict, Tuple, Iterable, Type, Union, Callable, Optional
from torch.utils.data import DataLoader
from sentence_transformers import losses
from sentence_transformers import evaluation
import numpy as np
import torch
import json
from transformers import PretrainedConfig
from tqdm.autonotebook import trange

import dataloader_creator

# sentece transformer with GAN
class SentenceTransformerWithGan(SentenceTransformer):
    def __init__(self, model_name_or_path: Optional[str] = None,
                 modules: Optional[Iterable[torch.nn.Module]] = None,
                 device: Optional[str] = None,
                 cache_folder: Optional[str] = None,
                 use_auth_token: Union[bool, str, None] = None
                 ):
        super(SentenceTransformerWithGan, self).__init__(model_name_or_path, modules, device, cache_folder, use_auth_token)
    
    def fit(self,
            train_objectives: Iterable[Tuple[DataLoader, torch.nn.Module]],
            model_teacher,
            generator_adversarial,
            evaluator: evaluation.SentenceEvaluator = None,
            epochs: int = 1,
            steps_per_epoch = None,
            scheduler: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            optimizer_class: Type[torch.optim.Optimizer] = torch.optim.AdamW,
            optimizer_params : Dict[str, object]= {'lr': 2e-5},
            weight_decay: float = 0.01,
            evaluation_steps: int = 0,
            output_path: str = None,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            use_amp: bool = False,
            callback: Callable[[float, int, int], None] = None,
            show_progress_bar: bool = True,
            checkpoint_path: str = None,
            checkpoint_save_total_limit: int = 0,
            ):

        ##Add info to model card
        #info_loss_functions = "\n".join(["- {} with {} training examples".format(str(loss), len(dataloader)) for dataloader, loss in train_objectives])
        info_loss_functions =  []
        for dataloader, loss in train_objectives:
            info_loss_functions.extend(model_card_templates.ModelCardTemplate.get_train_objective_info(dataloader, loss))
        info_loss_functions = "\n\n".join([text for text in info_loss_functions])

        info_fit_parameters = json.dumps({"evaluator": util.fullname(evaluator), "epochs": epochs, "steps_per_epoch": steps_per_epoch, "scheduler": scheduler, "warmup_steps": warmup_steps, "optimizer_class": str(optimizer_class),  "optimizer_params": optimizer_params, "weight_decay": weight_decay, "evaluation_steps": evaluation_steps, "max_grad_norm": max_grad_norm }, indent=4, sort_keys=True)
        self._model_card_text = None
        self._model_card_vars['{TRAINING_SECTION}'] = model_card_templates.ModelCardTemplate.__TRAINING_SECTION__.replace("{LOSS_FUNCTIONS}", info_loss_functions).replace("{FIT_PARAMETERS}", info_fit_parameters)
        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()
        self._target_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # self.device = self._target_device
        self.to(self._target_device)
        dataloaders = [dataloader for dataloader, _ in train_objectives]
        # smart batchingすると、ラベルと勝手に整列されちゃう
        # Use smart batching
        # for dataloader in dataloaders:
        #     dataloader.collate_fn = self.smart_batching_collate
        loss_models = [loss for _, loss in train_objectives]
        for loss_model in loss_models:
            loss_model.to(self._target_device)

        self.best_score = -9999999

        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])

        num_train_steps = int(steps_per_epoch * epochs)
        # 訓練、評価モードにする
        self.train()
        model_teacher.eval()

        # Prepare optimizers
        optimizers = []
        schedulers = []
        for loss_model in loss_models:
            param_optimizer = list(loss_model.named_parameters())

            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
            scheduler_obj = self._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

            optimizers.append(optimizer)
            schedulers.append(scheduler_obj)

        global_step = 0
        data_iterators = [iter(dataloader) for dataloader in dataloaders]
        num_train_objectives = len(train_objectives)

        skip_scheduler = False
        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            training_steps = 0

            for loss_model in loss_models:
                loss_model.zero_grad()
                loss_model.train()

            for _ in trange(steps_per_epoch, desc="Iteration", smoothing=0.05, disable=not show_progress_bar):
                for train_idx in range(num_train_objectives):
                    loss_model = loss_models[train_idx]
                    optimizer = optimizers[train_idx]
                    scheduler = schedulers[train_idx]
                    data_iterator = data_iterators[train_idx]
                    try:
                        data = next(data_iterator)
                    except StopIteration:
                        data_iterator = iter(dataloaders[train_idx])
                        data_iterators[train_idx] = data_iterator
                        # ここで先生modelに入力
                        data = next(data_iterator)

                    features, labels = data
                    labels = labels.to(self._target_device)
                    features = list(map(lambda batch: util.batch_to_device(batch, self._target_device), features))

                    if use_amp:
                        with autocast():
                            loss_value = loss_model(features, labels)

                        scale_before_step = scaler.get_scale()
                        scaler.scale(loss_value).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()

                        skip_scheduler = scaler.get_scale() != scale_before_step
                    else:
                        loss_value = loss_model(features, labels)
                        loss_value.backward()
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        optimizer.step()

                    optimizer.zero_grad()

                    if not skip_scheduler:
                        scheduler.step()

                training_steps += 1
                global_step += 1

                if evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    self._eval_during_training(evaluator, output_path, save_best_model, epoch, training_steps, callback)

                    for loss_model in loss_models:
                        loss_model.zero_grad()
                        loss_model.train()
            self._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)
            self._eval_during_training(evaluator, output_path, save_best_model, epoch, -1, callback)

        if evaluator is None and output_path is not None:   #No evaluator, but output path: save final model version
            self.save(output_path)

        if checkpoint_path is not None:
            self._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)


class ModelOperator():
    def define_model(self):
        # xlmr_tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
        # xlmr = models.Transformer("student_model", tokenizer_name_or_path="student_model")
        dir_student = "./sentencetransformer-distilbert-base-cased"
        with open(os.path.join(dir_student, "tokenizer_config.json"), "r") as f:
            tokenizer_config = json.load(f)
        tokenizer_args = {"config": PretrainedConfig(**tokenizer_config)}
        # xlmr = AutoModel.from_pretrained(dir_student)
        xlmr = models.Transformer(dir_student, tokenizer_args=tokenizer_args,
                         max_seq_length=200, tokenizer_name_or_path=dir_student)
        pooler = models.Pooling(
            xlmr.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True
        )
        student = SentenceTransformerWithGan(modules=[xlmr, pooler])
        print(student)
        dir_teacher = './sentence_bert'
        with open(os.path.join(dir_teacher, "tokenizer_config.json"), "r") as f:
            tokenizer_config_t = json.load(f)
        tokenizer_args_t = {"config": PretrainedConfig(**tokenizer_config_t)}
        teacher_eoncoder = models.Transformer(dir_teacher, tokenizer_args=tokenizer_args_t,
                         max_seq_length=200, tokenizer_name_or_path=dir_teacher)
        # teacher_eoncoder = models.Transformer(dir_teacher, tokenizer_name_or_path=dir_teacher)
        pooler = models.Pooling(
            teacher_eoncoder.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True
        )
        teacher = SentenceTransformerWithGan(modules=[teacher_eoncoder, pooler])
        print(teacher)
        return student, teacher

    def train_model(self):
        student, teacher = self.define_model()
        data = ParallelSentencesDataset(student_model=student, teacher_model=teacher, batch_size=8,
                                        use_embedding_cache=True)
        max_sentences_per_language = 4000
        train_max_sentence_length = 200  # max num of characters per sentence

        train_files = [f for f in os.listdir('./data') if 'train' in f]
        for f in train_files:
            print(f)
            data.load_data('./data/' + f, max_sentences=max_sentences_per_language,
                           max_sentence_length=train_max_sentence_length)
        loader = DataLoader(data, shuffle=True, batch_size=8, collate_fn=dataloader_creator.create_collate_fn(student.tokenizer, 200))
        loss = losses.MSELoss(model=student)

        tokenizer_gen = AutoTokenizer.from_pretrained('bert_mlm')
        generator = AutoModel.from_pretrained("bert_mlm")

        epochs = 2
        warmup_steps = int(len(loader) * epochs * 0.1)
        print("start fine-tuning")
        student.fit(
            train_objectives=[(loader, loss)],
            model_teacher=teacher,
            generator_adversarial=generator,
            epochs=epochs,
            checkpoint_path="./model_trained",
            warmup_steps=warmup_steps,
            output_path='./xlmr-ted',
            optimizer_params={'lr': 2e-6, 'eps': 1e-6},
            save_best_model=True,
            show_progress_bar=True
        )
        student.save("student_fine-tuned")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    model_op = ModelOperator()
    model_op.train_model()




