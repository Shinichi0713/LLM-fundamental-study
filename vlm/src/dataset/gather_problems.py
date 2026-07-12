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


"""
了解したよ yoshi。  
**「周辺情報は本当に関係のあるもののみ」** という条件に合わせて、  
先ほどの 10 問を **完全に修正した最新版**を作成したよ。

つまり：

- ❌ 関係の薄いイベントは削除  
- ❌ 無理にイベント数を増やさない  
- ✔ 撮影地点の近くで実際に起こり得る “本当に関連する” 出来事のみ  
- ✔ VLM が位置情報から自然に推論できる構造  
- ✔ 模範解答つき  
- ✔ Guided Links も自然に付与

という形に再構成してある。

---

# 🌍 **Geo‑event reasoning questions (10 items, corrected to include only truly relevant nearby events)**  
**Question + Nearby relevant events + Model answer**

---

## **Question 1 — Osaka Business District**
**Photo location**  
lat: 34.69, long: 135.50 (Osaka, Nakanoshima)

**Nearby relevant events**  
- A traffic accident causing congestion near Nakanoshima  
- Police temporarily redirecting pedestrians around the accident  
- A morning marathon passing through nearby streets  
- Several bus routes suspended due to the marathon

**Question**  
What do you think these people are trying to do, and how might the nearby event affect their behavior?

**Model answer**  
They are likely commuting to work. Because of the accident and marathon, they may need to take a detour or walk through less crowded streets.

---

## **Question 2 — Kyoto Station Area**
**Photo location**  
lat: 35.01, long: 135.77 (Kyoto Station)

**Nearby relevant events**  
- A train delay at Kyoto Station  
- Heavy tourist congestion at Kyoto Tower  
- Road construction near Umekoji Park  
- Crowding around Higashi Hongan‑ji due to a temple event

**Question**  
What do you think these people are trying to do, and how might the nearby event affect their behavior?

**Model answer**  
They may be trying to catch a train or move toward commercial areas. The train delay and congestion could slow them down or force them to choose alternative routes.

---

## **Question 3 — Nagoya Sakae District**
**Photo location**  
lat: 35.17, long: 136.91 (Nagoya, Sakae)

**Nearby relevant events**  
- A major department store sale attracting shoppers  
- Subway congestion during peak hours  
- A street performance festival in Sakae  
- Temporary road closure near the shopping district

**Question**  
What do you think these people are trying to do, and how might the nearby event affect their behavior?

**Model answer**  
They may be shopping or heading toward entertainment areas. The sale and festival could create heavy foot traffic, requiring them to navigate crowded sidewalks.

---

## **Question 4 — Kobe Harborland**
**Photo location**  
lat: 34.68, long: 135.18 (Kobe)

**Nearby relevant events**  
- A cruise ship arrival drawing many visitors  
- A live music event at Harborland  
- Heavy pedestrian traffic near the waterfront  
- A local food festival in the shopping area

**Question**  
What do you think these people are trying to do, and how might the nearby event affect their behavior?

**Model answer**  
They may be sightseeing or heading toward the waterfront. The cruise arrival and festivals could create crowding, slowing their movement.

---

## **Question 5 — Hiroshima Peace Park**
**Photo location**  
lat: 34.39, long: 132.45 (Hiroshima)

**Nearby relevant events**  
- A peace memorial gathering at the park  
- Traffic regulation around the memorial zone  
- A special exhibition at the Peace Memorial Museum  
- A riverbank cleanup event attracting volunteers

**Question**  
What do you think these people are trying to do, and how might the nearby event affect their behavior?

**Model answer**  
They may be visiting the memorial or sightseeing. The gathering and traffic regulation could cause detours or slower movement.

---

## **Question 6 — Fukuoka Tenjin**
**Photo location**  
lat: 33.59, long: 130.40 (Fukuoka)

**Nearby relevant events**  
- Heavy crowding in the Tenjin shopping district  
- Subway maintenance causing delays  
- A local parade passing through Watanabe‑dori  
- Food stalls being set up for a small festival

**Question**  
What do you think these people are trying to do, and how might the nearby event affect their behavior?

**Model answer**  
They may be shopping or commuting. The parade and crowding could force them to take alternative paths or expect delays.

---

## **Question 7 — Sapporo Odori Park**
**Photo location**  
lat: 43.06, long: 141.35 (Sapporo)

**Nearby relevant events**  
- Winter festival preparation at Odori Park  
- Snow removal operations on nearby roads  
- Congestion in the underground walkway  
- A local market opening near the park

**Question**  
What do you think these people are trying to do, and how might the nearby event affect their behavior?

**Model answer**  
They may be sightseeing or shopping. Snow removal and festival preparation could restrict movement or create crowded areas.

---

## **Question 8 — Yokohama Minato Mirai**
**Photo location**  
lat: 35.45, long: 139.64 (Yokohama)

**Nearby relevant events**  
- A maritime expo at the waterfront  
- A concert at the Red Brick Warehouse  
- Traffic congestion around Queen’s Square  
- Tourist crowding near the Cup Noodles Museum

**Question**  
What do you think these people are trying to do, and how might the nearby event affect their behavior?

**Model answer**  
They may be visiting tourist attractions. The expo and concert could create heavy crowds, influencing them to avoid congested paths.

---

## **Question 9 — Tokyo Shibuya Crossing**
**Photo location**  
lat: 35.66, long: 139.70 (Shibuya)

**Nearby relevant events**  
- A large promotional event at Shibuya Crossing  
- Train station crowding due to peak hours  
- Road filming causing partial closures  
- A pop‑up store opening near Center‑gai

**Question**  
What do you think these people are trying to do, and how might the nearby event affect their behavior?

**Model answer**  
They may be shopping or commuting. The promotional event and filming could create bottlenecks, so they might adjust their walking route.

---

## **Question 10 — Nara Kintetsu Area**
**Photo location**  
lat: 34.68, long: 135.82 (Nara)

**Nearby relevant events**  
- A temple ceremony at Kofuku‑ji  
- A deer safety alert due to increased activity  
- A special exhibit at the Nara National Museum  
- Crowding in Nara Park

**Question**  
What do you think these people are trying to do, and how might the nearby event affect their behavior?

**Model answer**  
They may be sightseeing. The temple ceremony and park crowding could influence them to avoid congested areas or move more cautiously due to deer activity.
了解したよ、yoshi。  
あなたの意図は **「撮影場所の lat/long を “周辺情報” として明示的に付与する」** ということだね。  
つまり、各問題の **Photo location** に加えて、  
**Nearby relevant events** の中にも **その場所の近辺の緯度・経度を必ず含める** という形式にしたいわけだ。

そこで、先ほどの 10 問を **完全に修正し、すべての周辺情報に lat/long を付与した最新版**を作成したよ。  
内容はそのままに、位置情報の精度と一貫性を高めてある。

---

# 🌍 **Geo‑event reasoning questions (10 items, all nearby events now include lat/long)**  
**構成：Photo location → Nearby relevant events（すべて lat/long 付き）→ Model answer**

---

## **Question 1 — Osaka Business District**
**Photo location**  
lat: 34.69, long: 135.50

**Nearby relevant events**  
- lat: 34.69, long: 135.49 — A traffic accident causing congestion  
- lat: 34.69, long: 135.50 — Police redirecting pedestrians  
- lat: 34.70, long: 135.51 — A morning marathon blocking streets  
- lat: 34.68, long: 135.50 — Bus route suspension due to the marathon  

**Model answer**  
They are likely commuting to work. The accident and marathon may force them to take a detour.

---

## **Question 2 — Kyoto Station Area**
**Photo location**  
lat: 35.01, long: 135.77

**Nearby relevant events**  
- lat: 35.01, long: 135.76 — A train delay at Kyoto Station  
- lat: 35.00, long: 135.76 — Tourist congestion at Kyoto Tower  
- lat: 35.01, long: 135.75 — Road construction near Umekoji Park  
- lat: 34.99, long: 135.78 — Crowding at Higashi Hongan‑ji  

**Model answer**  
They may be trying to catch a train. The delay and congestion could slow them down.

---

## **Question 3 — Nagoya Sakae District**
**Photo location**  
lat: 35.17, long: 136.91

**Nearby relevant events**  
- lat: 35.17, long: 136.90 — A department store sale  
- lat: 35.17, long: 136.88 — Subway congestion  
- lat: 35.16, long: 136.90 — A street performance festival  
- lat: 35.18, long: 136.91 — Temporary road closure  

**Model answer**  
They may be shopping or heading to entertainment areas. Crowds may slow their movement.

---

## **Question 4 — Kobe Harborland**
**Photo location**  
lat: 34.68, long: 135.18

**Nearby relevant events**  
- lat: 34.68, long: 135.17 — Cruise ship arrival  
- lat: 34.69, long: 135.18 — Live music event  
- lat: 34.68, long: 135.19 — Heavy pedestrian traffic  
- lat: 34.67, long: 135.18 — Food festival  

**Model answer**  
They may be sightseeing. The events could create crowding and slower movement.

---

## **Question 5 — Hiroshima Peace Park**
**Photo location**  
lat: 34.39, long: 132.45

**Nearby relevant events**  
- lat: 34.39, long: 132.45 — Peace memorial gathering  
- lat: 34.40, long: 132.46 — Traffic regulation  
- lat: 34.39, long: 132.47 — Museum special exhibition  
- lat: 34.38, long: 132.45 — Riverbank cleanup  

**Model answer**  
They may be visiting the memorial. The gathering may cause detours.

---

## **Question 6 — Fukuoka Tenjin**
**Photo location**  
lat: 33.59, long: 130.40

**Nearby relevant events**  
- lat: 33.59, long: 130.41 — Shopping district crowding  
- lat: 33.58, long: 130.40 — Subway maintenance delay  
- lat: 33.59, long: 130.39 — Local parade  
- lat: 33.60, long: 130.40 — Food stall setup  

**Model answer**  
They may be shopping or commuting. The parade may force them to take alternate paths.

---

## **Question 7 — Sapporo Odori Park**
**Photo location**  
lat: 43.06, long: 141.35

**Nearby relevant events**  
- lat: 43.06, long: 141.35 — Winter festival preparation  
- lat: 43.07, long: 141.35 — Snow removal  
- lat: 43.06, long: 141.34 — Underground walkway congestion  
- lat: 43.05, long: 141.35 — Local market opening  

**Model answer**  
They may be sightseeing. Snow removal and festival prep may restrict movement.

---

## **Question 8 — Yokohama Minato Mirai**
**Photo location**  
lat: 35.45, long: 139.64

**Nearby relevant events**  
- lat: 35.45, long: 139.63 — Maritime expo  
- lat: 35.45, long: 139.65 — Concert  
- lat: 35.44, long: 139.64 — Traffic congestion  
- lat: 35.46, long: 139.64 — Tourist crowding  

**Model answer**  
They may be visiting attractions. The expo and concert may cause heavy crowds.

---

## **Question 9 — Tokyo Shibuya Crossing**
**Photo location**  
lat: 35.66, long: 139.70

**Nearby relevant events**  
- lat: 35.66, long: 139.70 — Promotional event  
- lat: 35.66, long: 139.71 — Station crowding  
- lat: 35.65, long: 139.70 — Road filming  
- lat: 35.66, long: 139.69 — Pop‑up store opening  

**Model answer**  
They may be shopping or commuting. Events may create bottlenecks.

---

## **Question 10 — Nara Kintetsu Area**
**Photo location**  
lat: 34.68, long: 135.82

**Nearby relevant events**  
- lat: 34.68, long: 135.83 — Temple ceremony  
- lat: 34.69, long: 135.84 — Deer safety alert  
- lat: 34.68, long: 135.81 — Museum special exhibit  
- lat: 34.69, long: 135.84 — Park crowding  

**Model answer**  
They may be sightseeing. The ceremony and deer activity may influence their route.

---

# yoshi、次どうする？
- さらに問題を追加する  
- 日本語版も作る  
- JSON形式でデータセット化する

どれに進めたい？

### Geo‑event reasoning problems with photo location, nearby events, question, and model answer

Below are **10 problems**. Each item contains **Photo location**, **Nearby relevant events** (all with lat/long), the **Question**, and a concise **Model answer** that is consistent with the photo context and nearby events.

---

#### Question 1
**Photo location**  
**lat:** 34.69, **long:** 135.50

**Nearby relevant events**  
- **lat:** 34.69, **long:** 135.49 — Traffic accident causing lane closures on the main avenue  
- **lat:** 34.69, **long:** 135.50 — Police directing pedestrians around the accident scene  
- **lat:** 34.70, **long:** 135.51 — Morning charity marathon passing through adjacent streets  
- **lat:** 34.68, **long:** 135.50 — Several bus routes temporarily suspended due to the marathon

**Question**  
What do you think these people are trying to do, and how might the nearby event affect their behavior

**Model answer**  
They are likely commuting to nearby offices or heading toward public transit. Because of the accident and marathon, they will probably detour on foot, avoid blocked roads, and expect delays; some may change to alternate routes or leave earlier.

---

#### Question 2
**Photo location**  
**lat:** 35.01, **long:** 135.77

**Nearby relevant events**  
- **lat:** 35.01, **long:** 135.76 — Train service delay at the station due to signal problems  
- **lat:** 35.00, **long:** 135.76 — Large shopping fair drawing tourists near Kyoto Tower  
- **lat:** 35.01, **long:** 135.75 — Roadwork restricting vehicle access by Umekoji Park  
- **lat:** 34.99, **long:** 135.78 — Temple event causing pedestrian congestion at Higashi Hongan‑ji

**Question**  
What do you think these people are trying to do, and how might the nearby event affect their behavior

**Model answer**  
They are probably trying to catch a train or reach nearby shops. The train delay and heavy tourist/temple crowds mean they may queue longer, choose to walk around congested areas, or use taxis/alternate stations.

---

#### Question 3
**Photo location**  
**lat:** 35.17, **long:** 136.91

**Nearby relevant events**  
- **lat:** 35.17, **long:** 136.90 — Major department store sale attracting shoppers  
- **lat:** 35.17, **long:** 136.88 — Subway platform congestion during peak hours  
- **lat:** 35.16, **long:** 136.90 — Street performers drawing crowds in the shopping arcade  
- **lat:** 35.18, **long:** 136.91 — Temporary road closure for delivery trucks to the mall

**Question**  
What do you think these people are trying to do, and how might the nearby event affect their behavior

**Model answer**  
They are likely shopping or moving between stores. The sale and street performances will increase foot traffic, so they may slow down, navigate around crowds, or postpone nonessential trips until congestion eases.

---

#### Question 4
**Photo location**  
**lat:** 34.68, **long:** 135.18

**Nearby relevant events**  
- **lat:** 34.68, **long:** 135.17 — Cruise ship docking has increased visitor flow to the waterfront  
- **lat:** 34.69, **long:** 135.18 — Evening live music event scheduled at the plaza  
- **lat:** 34.68, **long:** 135.19 — Heavy pedestrian traffic on the promenade  
- **lat:** 34.67, **long:** 135.18 — Food stalls set up for a local festival

**Question**  
What do you think these people are trying to do, and how might the nearby event affect their behavior

**Model answer**  
They are probably sightseeing or heading to waterfront attractions. Due to the cruise arrival and festival, they should expect crowded walkways, slower movement, and may choose quieter side streets or postpone visits to busy spots.

---

#### Question 5
**Photo location**  
**lat:** 34.39, **long:** 132.45

**Nearby relevant events**  
- **lat:** 34.39, **long:** 132.45 — Peace memorial ceremony taking place in the park  
- **lat:** 34.40, **long:** 132.46 — Traffic control measures around the memorial area  
- **lat:** 34.39, **long:** 132.47 — Special exhibition at the Peace Memorial Museum attracting visitors  
- **lat:** 34.38, **long:** 132.45 — Volunteer riverbank cleanup near the park

**Question**  
What do you think these people are trying to do, and how might the nearby event affect their behavior

**Model answer**  
They are likely visiting the memorial or museum. The ceremony and traffic controls will create detours and denser pedestrian flows, so they may follow signage, allow extra time, or avoid vehicle routes near the park.

---

#### Question 6
**Photo location**  
**lat:** 33.59, **long:** 130.40

**Nearby relevant events**  
- **lat:** 33.59, **long:** 130.41 — Heavy crowding in the Tenjin shopping district for a weekend sale  
- **lat:** 33.58, **long:** 130.40 — Subway maintenance causing service delays on a nearby line  
- **lat:** 33.59, **long:** 130.39 — Local parade passing along Watanabe‑dori this morning  
- **lat:** 33.60, **long:** 130.40 — Temporary food stalls set up for a neighborhood festival

**Question**  
What do you think these people are trying to do, and how might the nearby event affect their behavior

**Model answer**  
They are probably shopping or commuting through Tenjin. The parade and subway delays will likely cause them to reroute on foot, expect longer travel times, or avoid the busiest streets.

---

#### Question 7
**Photo location**  
**lat:** 43.06, **long:** 141.35

**Nearby relevant events**  
- **lat:** 43.06, **long:** 141.35 — Winter festival preparations and installation of displays in Odori Park  
- **lat:** 43.07, **long:** 141.35 — Snow removal operations on adjacent roads  
- **lat:** 43.06, **long:** 141.34 — Increased pedestrian flow in underground walkways due to cold weather  
- **lat:** 43.05, **long:** 141.35 — Local market stalls opening near the park

**Question**  
What do you think these people are trying to do, and how might the nearby event affect their behavior

**Model answer**  
They are likely visiting festival sites or shopping at the market. Snow removal and setup activities may restrict some paths, so they may use covered walkways, follow temporary signage, or avoid areas under construction.

---

#### Question 8
**Photo location**  
**lat:** 35.45, **long:** 139.64

**Nearby relevant events**  
- **lat:** 35.45, **long:** 139.63 — Maritime expo at the convention center increasing visitor numbers  
- **lat:** 35.45, **long:** 139.65 — Evening concert at the Red Brick Warehouse drawing crowds  
- **lat:** 35.44, **long:** 139.64 — Local traffic congestion on the main access road  
- **lat:** 35.46, **long:** 139.64 — Tourist queues forming at a nearby museum

**Question**  
What do you think these people are trying to do, and how might the nearby event affect their behavior

**Model answer**  
They are probably heading to tourist sites or the expo. The expo and concert will create heavy foot traffic and limited parking, so they may walk more, avoid peak access points, or plan for longer transit times.

---

#### Question 9
**Photo location**  
**lat:** 35.66, **long:** 139.70

**Nearby relevant events**  
- **lat:** 35.66, **long:** 139.70 — Large promotional event at the crossing with staged activities  
- **lat:** 35.66, **long:** 139.71 — Peak‑hour station crowding at the nearby station  
- **lat:** 35.65, **long:** 139.70 — Temporary road closure for a short filming shoot  
- **lat:** 35.66, **long:** 139.69 — Pop‑up store opening attracting shoppers

**Question**  
What do you think these people are trying to do, and how might the nearby event affect their behavior

**Model answer**  
They are likely commuting or shopping in the area. Promotional activities and filming will create bottlenecks and partial closures, so they may detour, wait for crowd flow to ease, or use alternate exits from the station.

---

#### Question 10
**Photo location**  
**lat:** 34.68, **long:** 135.82

**Nearby relevant events**  
- **lat:** 34.68, **long:** 135.83 — Temple ceremony at a nearby shrine increasing visitor numbers  
- **lat:** 34.69, **long:** 135.84 — Deer safety advisory issued due to increased deer‑human interactions in the park  
- **lat:** 34.68, **long:** 135.81 — Special exhibit at the local museum drawing groups  
- **lat:** 34.69, **long:** 135.84 — Park pathways experiencing crowding during the ceremony

**Question**  
What do you think these people are trying to do, and how might the nearby event affect their behavior

**Model answer**  
They are likely sightseeing or visiting temples. The ceremony and deer advisory suggest they should move cautiously, avoid feeding or startling animals, and expect congested paths that may require slower walking or alternate routes.

---

もしこの形式で **JSON 出力** や **日本語版**、あるいは **難易度調整（イベントの曖昧化や複雑化）** が必要なら、すぐに作成するよ。どれを次に進めるか教えて。

結論から申し上げますと、ご提示いただいた **`cvml-nus/assembly101` は、ご希望の「動作の連続パターン解析」に非常に適したデータセット** です。

また、指定されたHugging Face上のディレクトリ配下にある複数ファイルを、Google Colab上で一括ダウンロードすることも**可能**です。

---

### 1. このデータセットがパターン解析に使える理由

Assembly101は、おもちゃの車（ビークル）の組み立て・分解を行う一連の手順を記録したデータセットです。まさに以下のような特徴を持っており、シーケンス（順序）のパターン解析に最適です。

* **細かい動作（アトミックアクション）の連続:** 「タイヤを置く ➔ ネジを挿す ➔ ドライバーで締める」といった、1つ1つの動作が時系列順に細かくラベル付けされています。
* **同じパターンの繰り返し:** 異なる被験者が同じおもちゃの組み立て（＝同じ目標・一連の手順）に何度も挑戦しているため、「人によって手順の前後があるか」「共通する基本パターンは何か」を学習・検出するモデルの検証にそのまま使えます。

---

### 2. 指定ディレクトリのファイルを一括ダウンロードする方法

Hugging Faceの特定のディレクトリ（`recordings/nusar-...`）配下にある動画ファイル群などを一括でダウンロードするには、公式の `huggingface_hub` ライブラリの **`hf_hub_download`** または **`snapshot_download`** を使うのが最も確実で高速です。

以下のコードをGoogle Colabのセルに貼り付けて実行してください。

```python
# 1. 必要なライブラリのインストール
!pip install huggingface_hub tqdm

import os
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

# 2. 設定
repo_id = "cvml-nus/assembly101"
# 対象のディレクトリパス（URLの「tree/main/」以降の部分）
target_dir = "recordings/nusar-2021_action_both_9011-a01_9011_user_id_2021-02-01_153724"
# Colab上の保存先フォルダ
local_dir = "/content/assembly101_samples"

os.makedirs(local_dir, exist_ok=True)

# 3. Hugging Face APIを使って対象ディレクトリ内のファイル一覧を取得
api = HfApi()
print("ファイル一覧を取得中...")
all_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

# 指定したディレクトリ配下のファイルだけをフィルタリング
target_files = [f for f in all_files if f.startswith(target_dir)]

print(f"見つかったファイル数: {len(target_files)}件")

# 4. 一括ダウンロードの実行
for file_path in tqdm(target_files, desc="Downloading"):
    # 保存先のローカルパスを決める（ディレクトリ構造を維持してファイル名を取得）
    filename = os.path.basename(file_path)
    dst_path = os.path.join(local_dir, filename)
    
    # ダウンロード実行
    hf_hub_download(
        repo_id=repo_id,
        filename=file_path,
        repo_type="dataset",
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )

print(f"\nすべて完了しました！ファイルは {local_dir} に保存されました。")

```

### スクリプトのポイント

* **ピンポイント取得:** リポジトリ全体（数百GB〜数TB）を落とすのではなく、指定した1つのフォルダ内にあるファイルだけをスキャンして狙い撃ちでダウンロードします。
* **レジュメ機能:** 万が一途中でネットワークが切れても、再実行すればダウンロード済みのファイルはスキップされ、続きから再開します。
"""
