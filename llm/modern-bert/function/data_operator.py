
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

dir_dataset = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")
print(dir_dataset)

class DataOperator():
    def __init__(self):
        self.dir_dataset = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")


    def __read_articles_from_directory(self, directory_path):
        """ディレクトリから記事を読み込む関数"""
        # LICENSE.txt以外の全ファイルを取得
        files = [f for f in os.listdir(directory_path) if f != "LICENSE.txt"]

        articles = []
        for file in files:
            with open(os.path.join(directory_path, file), "r", encoding="utf-8") as f:
                lines = f.readlines()
                articles.append({
                    "title": lines[2].strip(),
                    "sentence": ''.join(lines[3:]).strip(),
                    "file": file
                })

        return articles

    def __read_category_from_directory(self, directory_path):
        """LICENSE.txtからカテゴリ情報を取得する関数"""
        with open(os.path.join(directory_path, "LICENSE.txt"), "r", encoding="utf-8") as f:
            lines = f.readlines()
            category = lines[-2].strip()
        return category
    
    def __get_dataframe(self):
        # メインディレクトリを取得
        directories = [d for d in os.listdir(dir_dataset) 
                    if d not in ["CHANGES.txt", "README.txt"]]
        self.gentres = directories
        # 全てのデータフレームを格納するリスト
        all_dfs = []

        # 各ディレクトリからデータを読み込む
        for directory in directories:
            dir_path = os.path.join(self.dir_dataset, directory)
            category = self.__read_category_from_directory(dir_path)
            articles = self.__read_articles_from_directory(dir_path)
            
            df = pd.DataFrame(articles)
            df["category"] = category
            
            all_dfs.append(df)
        # 全てのデータフレームを結合
        full_df = pd.concat(all_dfs, ignore_index=True)

        # カテゴリを数値に変換（ラベルエンコード）
        self.label_encoder = LabelEncoder()
        full_df['category_id'] = self.label_encoder.fit_transform(full_df['category'])

        # カテゴリとIDの対応関係を保存
        category_mapping = dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))
        print("カテゴリとIDのマッピング:")
        # for category, idx in category_mapping.items():
        #     print(f"{category}: {idx}")
        # AIモデルへのinput用にタイトルと本文を結合したテキストを作成
        full_df['text_for_bert'] = full_df['title'] + ' ' + full_df['sentence']
        return full_df
    
    def make_dataframes(self):
        full_df = self.__get_dataframe()
        # カテゴリを数値に変換
        label_encoder = LabelEncoder()
        full_df['category_id'] = label_encoder.fit_transform(full_df['category'])

        # BERTモデル用にタイトルと本文を結合したテキストを作成
        full_df['text_for_input'] = full_df['title'] + ' ' + full_df['sentence']

        # カテゴリ情報
        id2label = {idx: label for idx, label in enumerate(label_encoder.classes_)}
        label2id = {label: idx for idx, label in id2label.items()}
        # カテゴリごとに均等にデータを分割
        # 学習:検証:テスト = 7:2:1
        train_dfs = []
        val_dfs = []
        test_dfs = []

        # バランスよくカテゴリを分けて分割
        for category in full_df['category'].unique():
            category_df = full_df[full_df['category'] == category]
            
            # まず学習データと残りのデータを分割
            train_category, temp_category = train_test_split(
                category_df, train_size=0.7, random_state=42
            )
            
            # 残りのデータを検証データとテストデータに分割（検証:テスト = 2:1）
            val_category, test_category = train_test_split(
                temp_category, train_size=2/3, random_state=42
            )
            
            train_dfs.append(train_category)
            val_dfs.append(val_category)
            test_dfs.append(test_category)

        # 分割したデータを結合
        train_df = pd.concat(train_dfs, ignore_index=True)
        val_df = pd.concat(val_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)

        return train_df, val_df, test_df, id2label, label2id



if __name__ == "__main__":
    data_op = DataOperator()
    df = data_op.make_dataframes()
    print(df.head())
