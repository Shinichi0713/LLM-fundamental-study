dir_base = "/content/images"
df_geo['image_path'] = df_geo['image_path'].apply(lambda x:os.path.join(dir_base, x))


import pandas as pd
import os
import random
import json
import matplotlib.pyplot as plt
from PIL import Image

def display_random_samples(df, num_samples=3):
    # 実際に存在する画像パスを持つ行だけに絞り込む
    df_valid = df[df['saved_image_path'].apply(lambda x: os.path.exists(str(x)))]
    
    if len(df_valid) == 0:
        print("エラー: 'saved_image_path' に指定されたパスに画像ファイルが見つかりません。")
        print("パスのプレフィックス（/content/Temporal など）やファイル名が正しいか再確認してください。")
        return

    # 指定されたサンプル数と有効データ数の小さい方を抽出件数にする
    samples_count = min(num_samples, len(df_valid))
    sampled_df = df_valid.sample(n=samples_count)

    for i, (idx, row) in enumerate(sampled_df.iterrows(), 1):
        print(f"\n{'='*40} サンプル {i} (インデックス: {idx}) {'='*40}")
        
        # 1. 画像の表示
        img_path = row['saved_image_path']
        try:
            img = Image.open(img_path)
            plt.figure(figsize=(6, 6))
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"Sample {i}: {os.path.basename(img_path)}")
            plt.show()
        except Exception as e:
            print(f"画像の読み込みに失敗しました ({img_path}): {e}")

        # 2. 問題文（プロンプト）の表示
        # カラム名が 'prompts' か 'question' の想定（適宜環境に合わせてください）
        prompt_text = row.get('prompts', row.get('question', 'N/A'))
        print(f"\n【問題 (Prompt)】\n{prompt_text}\n")

        # 3. 選択肢（Options）の表示
        options_raw = row.get('options_list', row.get('options', '[]'))
        
        # JSON文字列になっている場合はリストにデコードする
        if isinstance(options_raw, str):
            try:
                options = json.loads(options_raw)
            except json.JSONDecodeError:
                options = [options_raw]
        else:
            options = options_raw

        print("【選択肢 (Options)】")
        if isinstance(options, list):
            for j, opt in enumerate(options, 1):
                print(f"  ({j}) {opt}")
        else:
            print(f"  {options}")
            
        # もし正解ラベル（answer など）のカラムがあれば追加で表示
        if 'answer' in df.columns:
            print(f"\n【正解 (Answer)】: {row['answer']}")
            
        print(f"{'='*95}\n")

# 実行（表示したいサンプル数を num_samples で指定できます）
display_random_samples(df_geo, num_samples=3)


import pandas as pd
import os
import shutil

def pickup_and_save_samples(df, num_samples=20, output_dir="/content/image_problem", output_csv="/content/selected_problems.csv"):
    # 1. 必要なフォルダの作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. 'saved_image_path' カラムが存在し、かつ実際に画像ファイルが存在する行だけに絞り込む
    if 'saved_image_path' not in df.columns:
        print("エラー: 'saved_image_path' カラムが DataFrame に存在しません。")
        return
        
    df_valid = df[df['saved_image_path'].apply(lambda x: os.path.exists(str(x)) if pd.notna(x) else False)]
    
    available_count = len(df_valid)
    print(f"有効な画像付きのデータ件数: {available_count} 件")
    
    if available_count == 0:
        print("エラー: 有効な画像ファイルが見つかりません。パスを確認してください。")
        return
        
    # 3. ランダムサンプリング（指定件数よりデータが少ない場合は全件）
    pickup_count = min(num_samples, available_count)
    df_selected = df_valid.sample(n=pickup_count, random_state=None) # 必要に応じて random_state を固定してください
    
    # 4. 画像のコピーとCSV用のパス書き換え
    new_image_paths = []
    print(f"\n{pickup_count} 個の画像ファイルを {output_dir} にコピーしています...")
    
    for idx, row in df_selected.iterrows():
        src_path = row['saved_image_path']
        filename = os.path.basename(src_path)
        dst_path = os.path.join(output_dir, filename)
        
        try:
            shutil.copy(src_path, dst_path)
            new_image_paths.append(dst_path)
        except Exception as e:
            print(f"インデックス {idx} の画像コピーに失敗しました: {e}")
            new_image_paths.append(None)
            
    # 新しい保存先パスを記録（必要に応じて）
    df_selected['copied_image_path'] = new_image_paths
    
    # 5. ピックアップした問題のCSV保存
    df_selected.to_csv(output_csv, index=False, encoding='utf-8-sig')
    
    print(f"\n処理が完了しました！")
    print(f"- コピーされた画像フォルダ: {output_dir} (枚数: {len(os.listdir(output_dir))}枚)")
    print(f"- 選択された問題のCSV: {output_csv}")

# 実行
pickup_and_save_samples(df_geo, num_samples=20)

import pandas as pd
import os
import shutil

def pickup_and_save_samples(df, num_samples=20, output_dir="/content/image_problem", output_csv="/content/selected_problems.csv"):
    # 1. 必要なフォルダの作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. 'saved_image_path' カラムが存在し、かつ実際に画像ファイルが存在する行だけに絞り込む
    if 'saved_image_path' not in df.columns:
        print("エラー: 'saved_image_path' カラムが DataFrame に存在しません。")
        return
        
    df_valid = df[df['saved_image_path'].apply(lambda x: os.path.exists(str(x)) if pd.notna(x) else False)]
    
    available_count = len(df_valid)
    print(f"有効な画像付きのデータ件数: {available_count} 件")
    
    if available_count == 0:
        print("エラー: 有効な画像ファイルが見つかりません。パスを確認してください。")
        return
        
    # 3. ランダムサンプリング（指定件数よりデータが少ない場合は全件）
    pickup_count = min(num_samples, available_count)
    df_selected = df_valid.sample(n=pickup_count, random_state=None) # 必要に応じて random_state を固定してください
    
    # 4. 画像のコピーとCSV用のパス書き換え
    new_image_paths = []
    print(f"\n{pickup_count} 個の画像ファイルを {output_dir} にコピーしています...")
    
    for idx, row in df_selected.iterrows():
        src_path = row['saved_image_path']
        filename = os.path.basename(src_path)
        dst_path = os.path.join(output_dir, filename)
        
        try:
            shutil.copy(src_path, dst_path)
            new_image_paths.append(dst_path)
        except Exception as e:
            print(f"インデックス {idx} の画像コピーに失敗しました: {e}")
            new_image_paths.append(None)
            
    # 新しい保存先パスを記録（必要に応じて）
    df_selected['copied_image_path'] = new_image_paths
    
    # 5. ピックアップした問題のCSV保存
    df_selected.to_csv(output_csv, index=False, encoding='utf-8-sig')
    
    print(f"\n処理が完了しました！")
    print(f"- コピーされた画像フォルダ: {output_dir} (枚数: {len(os.listdir(output_dir))}枚)")
    print(f"- 選択された問題のCSV: {output_csv}")

# 実行
pickup_and_save_samples(df_geo, num_samples=20)