

import os, gc
import torch
import numpy as np
import function, model
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)


class TrainerModernBert():
    def __init__(self):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def make_preparation(self):
        data_operator = function.DataOperator()
        train_df, val_df, test_df, id2label, label2id = data_operator.make_dataframes()
        # トークナイザーとモデルのロード
        dict_setting = {'num_labels': len(data_operator.gentres), 'id2label': id2label, 'label2id': label2id}
        modern_bert = model.ModernBertClassifier(**dict_setting)
        tokenizer = modern_bert.tokenizer
        max_length = modern_bert.max_length

        # データセット作成(train/val/test)
        train_dataset = function.LivedoorDataset(
            train_df['text_for_input'].values,
            train_df['category_id'].values,
            tokenizer,
            max_length=max_length
        )
        
        val_dataset = function.LivedoorDataset(
            val_df['text_for_input'].values,
            val_df['category_id'].values,
            tokenizer,
            max_length=max_length
        )
        
        test_dataset = function.LivedoorDataset(
            test_df['text_for_input'].values,
            test_df['category_id'].values,
            tokenizer,
            max_length=max_length
        )

        training_args = self.__set_training()
        # データコレータ(各バッチごとに動的パディングで最適なパディングを行いながらデータを処理)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        # Trainerの初期化
        trainer = Trainer(
            model=modern_bert.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=function.compute_metrics,
        )
        return trainer, test_dataset, data_operator.label_encoder

    def __set_training(self):
        is_modernbert = True
        output_dir = os.path.joint(os.path.dirname(__file__), "output")
        if os.path.exists(output_dir) is False:
            os.mkdir(output_dir)
        # トレーニング設定
        training_args_dict = {
            "output_dir" : output_dir,  # 出力dir
            "learning_rate" : 8e-5 if is_modernbert else 5e-5,   # 学習率(modernBERTとtohokuで分けてる)
            "per_device_train_batch_size" : 8 if is_modernbert else 16,   # 学習バッチサイズ(modernBERTとtohokuで分けてる)
            "per_device_eval_batch_size" : 8 if is_modernbert else 16,   # 評価バッチサイズ(modernBERTとtohokuで分けてる)
            "num_train_epochs" : 3,  # epoch数
            "weight_decay" : 8e-6 if is_modernbert else 0.01,  # 重みの減衰率 正則化パラメータ(modernBERTとtohokuで分けてる)
            "adam_beta1" : 0.9,  # Adamオプティマイザ 一次モーメント推定係数
            "adam_beta2" : 0.98 if is_modernbert else 0.999,  # Adamオプティマイザ 二次モーメント推定係数(modernBERTとtohokuで分けてる)
            "adam_epsilon" : 1e-6,  # Adamオプティマイザの数値安定性
            "lr_scheduler_type" : "linear",  # 学習率のスケジューリング方法(線形)
            "warmup_ratio" : 0.1,  # ウォームアップ期間のトレーニング比率(最初の10%のステップで学習率を0から設定値まで徐々に上げる)
            "logging_strategy" : "epoch",  # ログを記録するタイミング(Epoch毎)
            "eval_strategy" : "epoch",  # 検証データでの評価タイミング(Epoch毎)
            "save_strategy" : "epoch",  # モデルの保存(Epoch毎)
            "load_best_model_at_end" : True,  # トレーニング終了時に最良のモデルをロード
            "metric_for_best_model" :"f1",  # 「最良」のモデルを判断するメトリクス(F1スコア)
            "push_to_hub": False,  # Hugging Face Hubにモデルをアップロードしない
            "seed": 42,  # シード値の設定
            "data_seed": 42,  # データシャッフルのシード値
        }
        
        # ModernBERTの場合のみ追加設定を適用（でないと動かなかったので）
        if is_modernbert:
            training_args_dict.update({
                "bf16": True,  # BFloat16(16ビット浮動小数点数)精度を使用
                "bf16_full_eval": True,  # (評価)BFloat16精度を使用
                "gradient_accumulation_steps": 2,  # 勾配累積(8×2=16の実効バッチサイズ)
            })
            
        # TrainingArgumentsインスタンスを作成
        training_args = TrainingArguments(**training_args_dict)
        return training_args

    def train_model(self):
        trainer, test_dataset, label_encoder = self.make_preparation()
        # モデルのトレーニング
        trainer.train()
        
        # 学習未使用のテストデータでの評価
        test_results = trainer.evaluate(test_dataset)
        print(f"\nテスト結果: {test_results}")
        # テストデータの予測と混同行列
        predictions = trainer.predict(test_dataset)
        pred_labels = np.argmax(predictions.predictions, axis=1)
        true_labels = predictions.label_ids
        # 混同行列
        accuracy = test_results['eval_accuracy']
        f1 = test_results['eval_f1']
        cm = function.plot_confusion_matrix(
            true_labels, 
            pred_labels, 
            label_encoder.classes_, 
            f"modern bert 混同行列",
            accuracy,
            f1
        )
        
        # メモリ解放（連続推論をする為）
        del model, trainer
        torch.cuda.empty_cache()
        gc.collect()
        
        return test_results, cm, true_labels, pred_labels


if __name__ == "__main__":
    trainer = TrainerModernBert()
    trainer.train_model()
