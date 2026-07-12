import cv2
import numpy as np
import os
import json

# 画像が格納されているディレクトリ
image_dir = "/content/images_det"

print("====================================")
print("   境界グラデーション強度の計算開始")
print("====================================")

for ann in sampled_annotations:
    # 画像情報の取得
    # (image_mapが定義されている前提。ない場合はannからfile_nameを逆引きするか、
    #  事前に格納した情報から取得してください)
    image_id = ann['image_id']
    img_info = image_map.get(image_id)
    if not img_info:
        continue
        
    file_name = img_info['file_name']
    img_path = os.path.join(image_dir, file_name)
    
    if not os.path.exists(img_path):
        continue
        
    # 1. 画像をグレースケールで読み込み
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
        
    h_img, w_img = img.shape
    
    # COCO形式の bbox = [x_min, y_min, box_width, box_height]
    bbox = ann['bbox']
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    
    # 画像範囲外にはみ出さないようクリッピング
    x_min = max(0, x)
    y_min = max(0, y)
    x_max = min(w_img, x + w)
    y_max = min(h_img, y + h)
    
    # 物体が小さすぎる（幅や高さが3ピクセル未満）場合はスキップ
    if (x_max - x_min) < 3 or (y_max - y_min) < 3:
        ann['edge_gradient_intensity'] = 0.0
        continue
        
    # 2. 境界線（エッジ）周辺の領域をクロップ
    # 境界の「外側5ピクセル、内側5ピクセル」程度の細長い帯状の領域をターゲットにする
    pad = 5
    crop_y_min = max(0, y_min - pad)
    crop_y_max = min(h_img, y_max + pad)
    crop_x_min = max(0, x_min - pad)
    crop_x_max = min(w_img, x_max + pad)
    
    roi = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
    
    # 3. Sobelフィルタによる勾配（グラデーション強度）の計算
    # X方向、Y方向の変化量を計算
    sobelx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
    
    # 勾配の大きさ（マグニチュード）を算出
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # 4. 輪郭線が通る「BBoxの境界付近」だけの勾配平均を算出
    # マスクを作成して、BBoxの境界線の全周（padの厚み分）を抽出
    mask = np.zeros_like(roi, dtype=np.uint8)
    # 外側の矩形を白く塗りつぶす
    cv2.rectangle(mask, (0, 0), (roi.shape[1], roi.shape[0]), 255, -1)
    # 内側の矩形（本来のBBoxよりさらに少し内側）を黒くくり抜く
    cv2.rectangle(mask, (pad*2, pad*2), (roi.shape[1] - pad*2, roi.shape[0] - pad*2), 0, -1)
    
    # マスク領域内の勾配の平均値を計算
    edge_gradients = gradient_magnitude[mask == 255]
    
    if len(edge_gradients) > 0:
        mean_gradient = float(np.mean(edge_gradients))
    else:
        mean_gradient = 0.0
        
    # 5. アノテーションデータに新しい特徴量として追加保持
    ann['edge_gradient_intensity'] = mean_gradient

print("- 全データのエッジ勾配強度の計算が完了しました。")
print("====================================\n")

# 1件サンプルを表示して確認
print("■ 特徴量追加後のデータサンプル:")
print(json.dumps(sampled_annotations[0], indent=2, ensure_ascii=False))

import cv2
import numpy as np
import os
import random

# 画像が格納されているディレクトリ
image_dir = "/content/images_det"

print("====================================")
print("  境界のランダムぼかし加工を開始")
print("====================================")

# 画像単位でアノテーションをまとめる
from collections import defaultdict
image_to_anns = defaultdict(list)
for ann in sampled_annotations:
    if 'image_id' in ann:
        image_to_anns[ann['image_id']].append(ann)

success_count = 0

for image_id, anns in image_to_anns.items():
    img_info = image_map.get(image_id)
    if not img_info:
        continue
        
    file_name = img_info['file_name']
    img_path = os.path.join(image_dir, file_name)
    
    if not os.path.exists(img_path):
        continue
        
    # 1. 画像の読み込み
    img = cv2.imread(img_path)
    if img is None:
        continue
        
    h_img, w_img, c_img = img.shape
    
    # 2. 【ランダム設定】画像ごとにぼかしの強さを動的に変更する
    # 弱（3）から強（21）までの奇数をランダムに選択
    k_size = random.choice([3, 5, 7, 9, 11, 13, 15, 17, 19, 21])
    blur_kernel_size = (k_size, k_size)
    
    # ぼかす境界線の太さも、ぼかしの強さに比例させる（2〜8ピクセル）
    border_width = max(2, int(k_size / 2.5))
    
    # 3. ランダムな強さで画像全体をぼかす
    blurred_img = cv2.GaussianBlur(img, blur_kernel_size, 0)
    
    # 4. アルファマスクの初期化
    alpha_mask = np.zeros((h_img, w_img), dtype=np.float32)
    
    for ann in anns:
        bbox = ann['bbox']
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        
        # 動的に決まった太さ（border_width）で境界線を描画
        cv2.rectangle(alpha_mask, (x, y), (x + w, y + h), 1.0, thickness=border_width)
        
    # 5. マスク自体のエッジも、カーネルサイズに合わせて滑らかにする
    mask_blur_size = max(3, k_size // 3)
    if mask_blur_size % 2 == 0:  # 偶数なら奇数に修正
        mask_blur_size += 1
        
    alpha_mask = cv2.GaussianBlur(alpha_mask, (mask_blur_size, mask_blur_size), 0)
    alpha_mask = np.expand_dims(alpha_mask, axis=2)
    
    # 6. 合成
    processed_img = blurred_img * alpha_mask + img * (1.0 - alpha_mask)
    processed_img = np.clip(processed_img, 0, 255).astype(np.uint8)
    
    # 7. 上書き保存
    cv2.imwrite(img_path, processed_img)
    success_count += 1

print(f"- {success_count} 枚の画像に対して、ランダムな強度での境界ぼかし加工が完了しました。")
print("====================================\n")

import pandas as pd
import numpy as np
import os
import glob

# 1. 現在のデータをデータフレーム化して再計算
df_current = pd.DataFrame(sampled_annotations)
df_current['log_area'] = np.log10(df_current['area'])
df_current['range_bin'] = pd.cut(df_current['log_area'], bins=10, labels=False)

# 2. 10個のレンジから各2件ずつ（計20件）をサンプリング
# 万が一特定のビンにデータが足りない場合に備え、replace=True（重複許容）にしています
final_20_df = df_current.groupby('range_bin', group_keys=False).apply(
    lambda x: x.sample(n=2, replace=True, random_state=42)
)

# 3. 20件のリストを確定（不要になった一時カラムは削除）
final_20_annotations = final_20_df.drop(columns=['log_area', 'range_bin']).to_dict(orient='records')

# 4. 20件に残った画像ファイルのセットを作成
keep_image_ids = {ann['image_id'] for ann in final_20_annotations}
keep_file_names = {image_map[img_id]['file_name'] for img_id in keep_image_ids if img_id in image_map}

# 5. images_det フォルダ内の不要な画像を削除してクリーンアップ
image_dir = "/content/images_det"
all_local_files = glob.glob(os.path.join(image_dir, "*"))

removed_count = 0
for file_path in all_local_files:
    file_name = os.path.basename(file_path)
    if file_name not in keep_file_names:
        if os.path.isfile(file_path):
            os.remove(file_path)
            removed_count += 1

print("====================================")
print("       データの厳選（20件）完了")
print("====================================")
print(f"- 厳選後のアノテーション数: {len(final_annotations_20 := final_20_annotations)} 件")
print(f"- 残したユニーク画像枚数 : {len(keep_file_names)} 枚")
print(f"- 削除した不要な画像枚数 : {removed_count} 枚")
print("====================================\n")

# 確認用：各グリッド（A1〜C3）にどう散らばっているか
from collections import Counter
grid_counts = Counter([ann['grid_position'] for ann in final_annotations_20])
print("■ 20件のグリッド配置分布:")
for grid in sorted(grid_counts.keys()):
    print(f"- {grid}: {grid_counts[grid]} 件")

import os
import shutil
import pandas as pd

# 1. パスの設定
src_image_dir = "/content/images_det"  # 先ほどぼかし加工を施した画像がある場所
dst_image_dir = "/content/image_det_2"  # 新しい20枚用の保存先
csv_output_path = "/content/annotations_20.csv"

os.makedirs(dst_image_dir, exist_ok=True)

# 2. 画像のコピー処理
copied_count = 0
for ann in final_annotations_20:
    image_id = ann['image_id']
    img_info = image_map.get(image_id)
    if not img_info:
        continue
        
    file_name = img_info['file_name']
    src_path = os.path.join(src_image_dir, file_name)
    dst_path = os.path.join(dst_image_dir, file_name)
    
    # まだコピーしていなければ実行
    if os.path.exists(src_path) and not os.path.exists(dst_path):
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy(src_path, dst_path)
        copied_count += 1

print("====================================")
print("     画像のコピー ＆ CSV保存処理")
print("====================================")
print(f"- image_det_2 への画像コピー完了: {copied_count} 枚")

# 3. JSON（辞書リスト）をCSV用に平坦化（フラット化）
# position_ratio の辞書を展開して、個別のカラムにします
flat_annotations = []
for ann in final_annotations_20:
    # データのコピーを作成して元のデータを保護
    flat_ann = ann.copy()
    
    # ファイル名もCSV側から直接紐づけられるように追加
    img_info = image_map.get(ann['image_id'])
    flat_ann['file_name'] = img_info['file_name'] if img_info else "Unknown"
    
    # position_ratio が存在すれば展開
    pos_ratio = flat_ann.pop('position_ratio', {})
    for key, value in pos_ratio.items():
        flat_ann[f'pos_{key}'] = value
        
    # COCOの bbox [x, y, w, h] も個別カラムにバラして出力（扱いやすくするため）
    bbox = flat_ann.get('bbox', [0, 0, 0, 0])
    flat_ann['bbox_x'] = bbox[0]
    flat_ann['bbox_y'] = bbox[1]
    flat_ann['bbox_w'] = bbox[2]
    flat_ann['bbox_h'] = bbox[3]
    
    flat_annotations.append(flat_ann)

# 4. Pandas DataFrameに変換してCSVエクスポート
df_csv = pd.DataFrame(flat_annotations)

# カラムの並び順を見やすく整理
column_order = [
    'id', 'image_id', 'file_name', 'category_id', 'area', 
    'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h',
    'pos_x_min_ratio', 'pos_y_min_ratio', 'pos_width_ratio', 'pos_height_ratio',
    'pos_x_center_ratio', 'pos_y_center_ratio', 'grid_position', 'edge_gradient_intensity'
]
# 存在するカラムだけ安全に指定
existing_columns = [col for col in column_order if col in df_csv.columns]
df_csv = df_csv[existing_columns]

# CSV保存
df_csv.to_csv(csv_output_path, index=False, encoding='utf-8')
print(f"- CSVファイルの保存完了: {csv_output_path}")
print("====================================\n")

# CSVの中身をチラ見
print("■ 作成されたCSVデータの先頭3行:")
display(df_csv.head(3))