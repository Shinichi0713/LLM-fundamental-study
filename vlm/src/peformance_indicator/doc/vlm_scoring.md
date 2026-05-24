
[過去VLMの性能指標](https://yoshishinnze.hatenablog.com/entry/2026/01/26/000000)について調べました。
少し時間が経過して今度は、"使う前にどんな性能のVLMかしら"を知りたいということで、実際にベンチマークを使ってVLMの性能評価をしたいと考えるようになりました。

__本日の内容:__
>ベンチマークを用いてVLMの性能評価を実際に行ってみる。

## ベンチマークとは

### 言葉の意味
「ベンチマーク」は、**「比較の基準」** や **「目標とする水準」** を意味する言葉です。

- **元々の意味**  
  測量で使う「基準点」のこと。そこを起点に高さや距離を測る目印です。

- **ビジネス・金融での意味**  
  - 他社や業界平均などと**比較するための基準**  
    例：「業界のベンチマークに合わせてコストを削減する」  
  - 投資の世界では、**市場全体の動きを表す指標**  
    例：日経平均株価やTOPIXは「日本株のベンチマーク」と呼ばれる

- **IT・技術での意味**  
  - 性能を**比較するためのテスト**  
    例：「CPUのベンチマークテストで性能を測る」「Webサイトの表示速度をベンチマークする」

**一言でいうと**  
「自分の成果や性能を、**他のものと比べるための基準・目安**」のことです。

### 背景

LLM/VLMのベンチマークが必要になった主なきっかけは、**「どのモデルがどれだけ優れているかを、公平に比較できる共通の物差し」が欲しくなった**からです。

具体的には、次のような理由があります。

1. **モデルが急増し、性能差が分かりにくくなった**  
   - GPT系、LLaMA系、Gemini系など、さまざまなLLM/VLMが登場し、  
     「どのモデルが本当に強いのか」を比較する手段が必要になりました。

2. **タスクごとの得意・不得意が複雑になった**  
   - 同じ「文章生成」でも、  
     - 長文生成  
     - 推論  
     - コード生成  
     - マルチモーダル（画像＋テキスト）  
     など、得意分野が異なるため、**特定のタスクに特化した評価基準**が必要になりました。

3. **研究・開発の進捗を客観的に測るため**  
   - 新しいアーキテクチャや学習手法を提案しても、  
     「どれだけ性能が上がったか」を**数値で示せる共通のテストセット**がないと、説得力が弱くなります。  
   - そのため、MMLU、GSM8K、HumanEval、MMBench などのベンチマークが整備されました。

4. **安全性・バイアス評価の必要性**  
   - モデルが有害な出力をしないか、偏った回答をしないか、といった**安全性・公平性**も評価する必要が出てきたため、  
     それらを測るためのベンチマーク（例：TruthfulQA、BOLD）も整備されました。

## 公開されているベンチマーク

代表的なVLMベンチマークを、**公開状況・評価内容・Colabでの利用しやすさ**の観点でまとめると、以下のようになります。


| ベンチマーク | 公開状況 | 評価している内容 | Google Colabでの利用しやすさ |
|--------------|----------|------------------|------------------------------|
| **MMBench** | 公開（GitHub・Hugging Face） | 画像＋テキストの多角的な理解（物体認識、OCR、行動認識、関係推論など20次元）[MMBench](https://mmbench.opencompass.org.cn) | △ 中程度。データと評価コードはあるが、GPUが必要で、OpenCompassなどの評価フレームワークと組み合わせる必要がある。 |
| **MMMU / MMMU-Pro** | 公開（GitHub・Hugging Face） | 大学レベルの専門知識＋画像理解・推論（30分野・6分野）[MMMU](https://mmmu-benchmark.github.io) | △ 中程度。MMMU評価コードとデータは公開されているが、GPU＋環境構築が必要。 |
| **VQAv2** | 公開（COCOベースのVQA） | 画像に関する自然言語質問への回答能力（常識・物体・属性など）[LearnOpenCV](https://learnopencv.com/vlm-evaluation-metrics) | 〇 比較的容易。データはCOCOとVQAv2で広く使われており、Colab上で簡単にダウンロードして評価できる。 |
| **TextVQA / DocVQA** | 公開 | 画像中の文字（OCR）を読んで答える能力（看板・文書など）[Datature](https://datature.com/glossary/vlm-benchmarks) | 〇 比較的容易。Hugging Faceなどからデータを取得し、Colab上で評価コードを書くことが可能。 |
| **POPE** | 公開 | 物体幻覚（存在しない物体を言い当てるか）の検出[Datature](https://datature.com/glossary/vlm-benchmarks) | 〇 比較的容易。データセットは小さめで、Colab上で簡単にロードして評価できる。 |
| **MM-Vet** | 公開 | OCR＋空間推論＋知識を統合した複合タスクの能力[Datature](https://datature.com/glossary/vlm-benchmarks) | △ 中程度。評価コードとデータは公開されているが、やや複雑でGPU＋環境整備が必要。 |
| **SEED-Bench** | 公開 | 生成的な理解（画像説明・対話など）を12次元で評価[Datature](https://datature.com/glossary/vlm-benchmarks) | △ 中程度。データと評価コードはあるが、GPUとフレームワークの組み合わせが必要。 |
| **MMStar** | 公開 | 視覚情報が必須な高難度サンプルによる多モーダル能力評価[MMStar](https://mmstar-benchmark.github.io) | △ 中程度。評価コード・データは公開されているが、Colab単体ではやや手間がかかる。 |
| **AMBER** | 公開（GitHub） | 多モーダル幻覚（存在・属性・関係など）を8次元で評価するLLM-freeベンチマーク。生成タスクと識別タスクの両方をカバー[AMBER GitHub](https://github.com/junyangwang0410/AMBER) | △ 中程度。データと`inference.py`が公開されており、Colab上でGPUを使って実行可能だが、環境構築とモデル準備が必要。 |

**まとめ**

- **公開されているか**：上記はいずれもデータ・評価コードが公開されているベンチマークです。
- **評価内容**：  
  - 一般的な画像理解（VQAv2、MMBench）  
  - 専門知識＋推論（MMMU系）  
  - OCR・文書理解（TextVQA、DocVQA）  
  - 安全性・幻覚（POPE、**AMBER**）  
  - 統合能力（MM-Vet、SEED-Bench、MMStar）  
  といった多様な能力を測るよう設計されています。
- **Google Colabでの利用しやすさ**：  
  - VQAv2、TextVQA、POPE などは比較的シンプルで、Colab上で試しやすいです。  
  - MMBench、MMMU、MM-Vet、SEED-Bench、MMStar、**AMBER** は、公開コードがあるものの、GPUと環境構築が必要で、Colab単体ではやや手間がかかります。

「Colabでとりあえず1つ試したい」場合は、**VQAv2** か **TextVQA** が入門として扱いやすく、**AMBER** は幻覚評価に特化したベンチマークとして追加で試すと良いかもしれません。

そして一番容易なわけではありませんが、幻覚の指標に関心があったので、今回は **AMBER** のトライアルを行ってみます。

## 手順

以下に、Colab で AMBER ベンチマークを使って VLM を評価する手順をまとめます。
あくまでざっとした流れです。

### 1. 準備

```python
# 1. AMBER リポジトリを取得
!git clone https://github.com/junyangwang0410/AMBER.git
amber_dir = "/content/AMBER"

# 2. 画像データをダウンロードして解凍
!gdown --id 1MaCHgtupcZUjf007anNl4_MV0o4DjXvl -O /content/AMBER_images.zip
!unzip -q /content/AMBER_images.zip -d /content/AMBER/data/

# 3. ライブラリのインストール
!pip install transformers accelerate torch pillow
```

### 2. モデルとデータの読み込み

```python
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import json
from PIL import Image
import os

# 1. BLIP-2 のロード
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.float16,
    device_map="auto"
)

# 2. 質問データの読み込み
with open("/content/AMBER/data/query/query_discriminative.json", "r", encoding="utf-8") as f:
    queries = json.load(f)

# 3. アノテーションの読み込み
with open("/content/AMBER/data/annotations.json", "r", encoding="utf-8") as f:
    annotations = json.load(f)
annotation_dict = {item["id"]: item for item in annotations}
```

### 3. 画像ファイル名の解決

```python
def resolve_image_filename(img_name):
    # "AMBER_0001.jpg" -> "AMBER_1.jpg"
    import re
    match = re.match(r"AMBER_0*(\d+)\.jpg", img_name)
    if match:
        num = int(match.group(1))
        return f"AMBER_{num}.jpg"
    return img_name
```

### 4. 推論実行

```python
inference_data = []

for q in queries:
    img_name = q["image"]
    question = q["query"]
    qid = q["id"]

    # 画像読み込み
    img_path = os.path.join(amber_dir, "data", "image", resolve_image_filename(img_name))
    image = Image.open(img_path).convert("RGB")

    # BLIP-2 で推論
    inputs = processor(image, question, return_tensors="pt").to(device, torch.float16)
    out = model.generate(**inputs, max_new_tokens=10)
    response = processor.decode(out[0], skip_special_tokens=True)

    # Yes/No に正規化
    if "yes" in response.lower():
        response = "Yes"
    elif "no" in response.lower():
        response = "No"
    else:
        response = "Yes"  # デフォルト（必要に応じて調整）

    inference_data.append({"id": qid, "response": response})

# 結果を保存
with open("/content/inference_data.json", "w", encoding="utf-8") as f:
    json.dump(inference_data, f, indent=2, ensure_ascii=False)

print("推論完了。結果を /content/inference_data.json に保存しました。")
```

### 5. 評価実行

```bash
cd /content/AMBER
python inference.py \
  --inference_data /content/inference_data.json \
  --annotation data/annotations.json
```

これで、AMBER の公式スクリプトが Accuracy / Precision / Recall / F1 などを自動計算して表示します。

### 6. 結果の確認

必要に応じて、前回提示した「サマリ表示コード」を実行して、タスク種別ごとの詳細な指標を確認できます。

## 実装

1. パッケージインストール

```
!pip install torch torchvision transformers accelerate pillow requests
!python -m spacy download en_core_web_lg
!pip install git+https://github.com/huggingface/transformers.git
```

2. データDL

結構要注意です。
githubのレポジトリを読んでその通り進めても上手くいきませんでした。
指定されたjsonファイルと、答えがのっているannotationファイルを結合して解答ファイルをつくります。

```python
import os
import json


amber_dir = "/content/AMBER"
image_dir = os.path.join(amber_dir, "data", "image")

# 判別タスク用クエリファイル（例）
annotation_path = os.path.join(amber_dir, "data", "query", "query_discriminative.json")

print("Image dir:", image_dir)
print("Annotation file:", annotation_path)


with open(annotation_path, "r", encoding="utf-8") as f:
    annotations = json.load(f)

print(f"Total samples: {len(annotations)}")
print("Example annotation:")
print(json.dumps(annotations[0], indent=2, ensure_ascii=False))
```

更に画像ファイルは別です。Google DriveからDLしてください。

```python
import os

image_dir = "/content/AMBER/data/image"
if not os.path.exists(image_dir):
    os.makedirs(image_dir, exist_ok=True)

image_files = os.listdir(image_dir)
print(f"Number of images in {image_dir}: {len(image_files)}")
if len(image_files) > 0:
    print("First few files:", image_files[:5])

# セル: 画像ダウンロード
import gdown
import zipfile

# Google Drive の共有リンク（ファイル ID を抽出）
file_id = "1MaCHgtupcZUjf007anNl4_MV0o4DjXvl"
url = f"https://drive.google.com/uc?id={file_id}"
output_zip = "/content/AMBER_images.zip"

print("Downloading images from Google Drive...")
gdown.download(url, output_zip, quiet=False)

print("Extracting images...")
with zipfile.ZipFile(output_zip, "r") as zip_ref:
    zip_ref.extractall("/content/AMBER/data/image")

print("Done.")
```

3. モデルロード

性能さておきでとりあえず入出力が可能なモデルということで以下を選定しました。

```python
# セル 4: BLIP-2 モデルのロード
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.float16
).to(device)
```

4. 推論

性能評価＝VLMにテスト用データを入力して、解答合わせすることになります。

```python
# セル: 推論ループ（画像パス解決付き）
from tqdm import tqdm

def resolve_image_filename(img_field):
    """
    query_discriminative.json の "image" フィールドを実際のファイル名に変換
    例: "AMBER_0001.jpg" -> "AMBER_1.jpg"
    """
    if img_field.startswith("AMBER_") and img_field.endswith(".jpg"):
        num_part = img_field[6:-4]  # "0001"
        try:
            num = int(num_part)  # 0001 -> 1
            # ゼロ埋めなし
            resolved = f"AMBER_{num}.jpg"
            return resolved
        except ValueError:
            pass
    return img_field

test_subset = queries[:50]  # 先頭 50 サンプルでテスト
inference_results = []
annotation_dict = {item["id"]: item for item in annotations}

for i, q in enumerate(tqdm(test_subset)):
    qid = q["id"]
    ann = annotation_dict.get(qid)

    if ann is None:
        print(f"Warning: No annotation found for id {qid}")
        continue

    img_field = q["image"]
    resolved_img = resolve_image_filename(img_field)
    img_path = os.path.join(amber_dir, "data", "image", resolved_img)

    if not os.path.exists(img_path):
        print(f"Warning: Image not found for id {qid}: {img_path}")
        continue

    question = q["query"]
    task_type = ann.get("type", "unknown")

    try:
        response = vlm_predict(img_path, question, task_type=task_type)
    except Exception as e:
        print(f"Error on sample {qid}: {e}")
        response = "error"

    inference_results.append({
        "id": qid,
        "response": response
    })

# 結果保存
inference_data_path = "/content/inference_data.json"
with open(inference_data_path, "w", encoding="utf-8") as f:
    json.dump(inference_results, f, indent=2, ensure_ascii=False)

print(f"Inference results saved to {inference_data_path}")
```

5. 結果

Colabのフォルダに`inference_data.json`に解答が出力されます。

![1779595244243](image/vlm_scoring/1779595244243.png)

中身はVLMからの回答が入っています。
全ての回答が`Yes`。。。目暗で全部はいと言ってきた感触。。。

```
  {
    "id": 1005,
    "response": "Yes"
  },
  {
    "id": 1006,
    "response": "Yes"
  },
  {
    "id": 1007,
    "response": "Yes"
  },
  {
    "id": 1008,
    "response": "Yes"
  },
  {
    "id": 1009,
    "response": "Yes"
  },
  {
    "id": 1010,
    "response": "Yes"
  },
```

この後解答合わせすると全問題の中で正解が25。というかすべて`Yes`と答えてました。。。
```
 'qa_correct_num': 50,
 'qa_correct_score': 25,
 'qa_no_num': 25,
 'qa_no_score': 0,
 'qa_ans_no_num': 0,
```

## 総括

以下に、Colab で AMBER ベンチマークを使って VLM を評価した内容を簡潔にまとめます。

### 1. ベンチマークとは

- **ベンチマーク**とは、性能や成果を**他と比較するための共通の物差し**です。
- LLM/VLM の世界では、モデルごとの強み・弱みを公平に比較するために、  
  MMBench、MMMU、VQAv2、POPE、AMBER など、さまざまなベンチマークが整備されています。
- 今回は、**幻覚（hallucination）** に特化したベンチマーク **AMBER** を使って、  
  VLM の「Yes/No 判別タスク」の性能を評価しました。

### 2. AMBER を使った評価の流れ（Colab）
ざっと説明するとこの流れです。

![1779595665262](image/vlm_scoring/1779595665262.png)

### 3. 結果の総括

- 今回のテストでは、**モデルがほぼすべての質問に対して「Yes」と回答**していました。
- そのため、集計結果は以下のようになりました。

```python
'qa_correct_num': 50,      # 全サンプル数
'qa_correct_score': 25,   # 正解数（Yes/No のうち Yes 側が半分）
'qa_no_num': 25,          # 正解が「No」のサンプル数
'qa_no_score': 0,         # 「No」と正しく答えた数（0）
```

- **Accuracy**：50.0%（全体の正解率）
- **Precision / Recall / F1**：0.0%（「No」に関する指標が計算不能）

**結論として**、  
- AMBER ベンチマークを使った VLM 評価の一連の流れは、Colab 上で問題なく実行できました。
- 一方で、今回使用した BLIP-2 の設定では、**「No」のケースをほとんど正しく認識できていない**ことが明らかになりました。
- 今後は、プロンプト設計や出力正規化の調整、あるいは別の VLM モデルを用いることで、  
  幻覚検出や Yes/No 判別の性能を改善していく必要があります。


