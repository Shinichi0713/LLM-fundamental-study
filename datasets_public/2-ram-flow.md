Wikipediaのような大規模データセットを扱う際、特に `group_texts` のような「結合して再分割する処理」は、メモリ（RAM）消費量が激増する最大の要因です。

以前のコードにあった `sum(examples[k], [])` という記述は、バッチ内の全データを巨大なリストとしてメモリ上に展開するため、Colabの標準メモリ（12GB〜）だとすぐに溢れてしまいます。

以下の **3つの対策** を順に試してみてください。効果が高い順に並べています。

---

### 対策1: `map`関数のパラメータ調整（最も手軽）

コードを書き換える前に、`dataset.map` の引数を調整してメモリ消費を抑えます。

1. **`num_proc` を削除する（重要）** : 並列処理は速いですが、プロセス数分だけメモリを複製して消費します。RAM不足のときは**シングルプロセス（1つ）**にするのが鉄則です。
2. **`batch_size` を小さくする** : デフォルトは1000ですが、これを **100** や **50** に下げます。
3. **`writer_batch_size` を指定する** : 処理結果をこまめにディスク（キャッシュ）に書き出し、メモリに溜め込まないようにします。

**修正コード:**

**Python**

```
# 1. トークン化
tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    batch_size=100,         # 1000 -> 100 に減らす
    # num_proc=4,           # 並列処理はコメントアウト（無効化）
    writer_batch_size=100,  # こまめにディスクに書き出す
    remove_columns=dataset.column_names
)

# 2. チャンク化
lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=100,         # ここも小さくする
    # num_proc=4,           # 無効化
    writer_batch_size=100
)
```

---

### 対策2: `group_texts` 関数のメモリ最適化

`sum(list, [])` は非常に遅く、メモリ効率が悪いです。Pythonの標準ライブラリ `itertools` を使って、イテレータとして処理するように書き換えると劇的に改善します。

**修正コード:**

**Python**

```
from itertools import chain

# コンテキスト長
block_size = 512

def group_texts(examples):
    # sum([], []) の代わりに itertools.chain を使う
    # これにより巨大なリストを一度に作らずに済みます
    concatenated_examples = {
        k: list(chain(*examples[k])) 
        for k in examples.keys()
    }
  
    total_length = len(concatenated_examples[list(examples.keys())[0]])
  
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
      
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result
```

---

### 対策3: ストリーミング処理 (Streaming) を使う

上記2つでも落ちる場合（特にWikipedia全量などの場合）、データをメモリに展開せず、HDD/ネットワークから逐次読み込む **Streamingモード** を使います。

ただし、`dataset.map` の挙動が変わるため、`IterableDataset` として扱う必要があります。

**Python**

```
# 1. streaming=True でロード
dataset = load_dataset("izumi-lab/wikipedia-ja-20230720", split="train", streaming=True)

# 2. ジェネレータ関数を使った加工（IterableDataset用）
def tokenization_generator(dataset):
    buffer = {"input_ids": [], "attention_mask": []}
  
    for sample in dataset:
        # トークン化
        outputs = tokenizer(sample["text"])
      
        # バッファに追加
        buffer["input_ids"].extend(outputs["input_ids"])
        buffer["attention_mask"].extend(outputs["attention_mask"])
      
        # block_size分溜まったら切り出してyield（放出）
        while len(buffer["input_ids"]) >= block_size:
            yield {
                "input_ids": buffer["input_ids"][:block_size],
                "labels": buffer["input_ids"][:block_size] # MLM用
            }
            # 放出した分を削除
            buffer["input_ids"] = buffer["input_ids"][block_size:]
            buffer["attention_mask"] = buffer["attention_mask"][block_size:]

# 3. IterableDatasetを作成
from datasets import IterableDataset

# 元のdatasetからジェネレータを通して新しいDatasetを作る
lm_dataset_stream = IterableDataset.from_generator(
    tokenization_generator, 
    gen_kwargs={"dataset": dataset}
)

# --- 使い方 ---
# ストリーミングなので len() は使えません。
# DataLoaderにそのまま渡せます。
dataloader = DataLoader(lm_dataset_stream, batch_size=8, collate_fn=data_collator)
```

推奨手順:

まずは 対策1（num_proc削除とbatch_size縮小） と 対策2（itertools使用） を組み合わせて試してください。これで大抵の場合は解決します。それでもダメなら対策3へ進んでください。
