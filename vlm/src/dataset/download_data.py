import pandas as pd
import json
import os
import io
from PIL import Image

def convert_parquet_to_csv_and_images(parquet_path, csv_path, image_dir):
    if not os.path.exists(parquet_path):
        print(f"エラー: {parquet_path} が見つかりません。")
        return

    # 画像保存用フォルダの作成
    os.makedirs(image_dir, exist_ok=True)

    print(f"Parquetファイルを読み込んでいます: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    print("検出されたカラム一覧:", df.columns.tolist())
    
    # 画像ファイル名を記録する新しい列
    saved_paths = []
    
    print("画像を抽出して保存しています...")
    for idx, row in df.iterrows():
        # 連番のファイル名を生成
        img_filename = f"image_{idx:05d}.png"
        img_path = os.path.join(image_dir, img_filename)
        
        # 'image' カラムからデータを取得
        img_data = row.get('image', None)
        
        success = False
        if img_data is not None:
            try:
                # パターン1: 辞書型で 'bytes' キーを持つ場合（Hugging Face形式）
                if isinstance(img_data, dict) and 'bytes' in img_data:
                    img = Image.open(io.BytesIO(img_data['bytes']))
                    img.save(img_path)
                    success = True
                # パターン2: バイナリ（bytes）データが直接入っている場合
                elif isinstance(img_data, bytes):
                    img = Image.open(io.BytesIO(img_data))
                    img.save(img_path)
                    success = True
                # パターン3: すでに PIL Image オブジェクトとして読み込まれている場合
                elif hasattr(img_data, 'save'):
                    img_data.save(img_path)
                    success = True
            except Exception as e:
                print(f"行 {idx} の画像保存中にエラーが発生しました: {e}")
        
        if success:
            saved_paths.append(img_path)
        else:
            saved_paths.append(None)  # 画像が取得できなかった場合

    # CSVに対応する画像パスを記録
    df['saved_image_path'] = saved_paths
    
    # 既存の 'image' カラム（巨大なバイナリデータ）はCSVを軽量化するため除外
    if 'image' in df.columns:
        df = df.drop(columns=['image'])
    
    # 辞書型やリスト型が含まれるカラムをCSV用にJSON文字列化
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, (dict, list))).any():
            df[col] = df[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else x)
            
    # CSVとして出力
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n処理が完了しました！")
    print(f"- メタデータ（CSV）: {csv_path}")
    print(f"- 画像フォルダ: {image_dir} (総枚数: {len([n for n in saved_paths if n is not None])}枚)")

def geo_filter():
    df = pd.read_csv("/content/Single_metadata.csv")
    df['image_path'] = df['image_path'].apply(lambda x: os.path.basename(str(x)))
    df.task.value_counts()

    df_geo = df[df['task'] == 'Sptatial Relation Classification']

    # 絞り込んだデータだけを別のCSVとして保存
    df_geo.to_csv("/content/Single_geo_only.csv", index=False, encoding='utf-8-sig')
    print("地理関係のみに絞り込んだCSVを保存しました。")

if __name__ == '__main__':
    input_parquet = '/content/Single.parquet?download=true'
    output_csv = '/content/Single_metadata.csv'
    image_directory = '/content/image'
    
    convert_parquet_to_csv_and_images(input_parquet, output_csv, image_directory)
