いいですね — 実務で使える **トークナイザ学習コード**を二通り用意します。

用途に応じて使い分けてください：

* **A. SentencePiece（公式で手軽、Unigram/BPE対応）** — 多言語・日本語に強い。
* **B. 🤗 Tokenizers（Hugging Face の高速ライブラリ）** — Byte-level BPE / WordPiece / Unigram を柔軟に高速に学習できる。

どちらも「巨大コーパス（テキストファイル群）」を入力にして語彙を学習します。

コマンドは Colab / ローカルでそのまま実行できます。

---

## A. SentencePiece を使う（簡単・安定）

### インストール

```bash
pip install sentencepiece
```

### 学習コード（BPE または Unigram）

```python
import sentencepiece as spm
import pathlib

# --- 準備 ---
# 複数のテキストファイルを結合して学習用 input.txt を作るのが一般的
# ここでは例として data/ フォルダの *.txt を学習に使う
files = [str(p) for p in pathlib.Path("data").glob("*.txt")]
assert files, "data/*.txt を用意してください"

input_files = ",".join(files)

# 学習パラメータ
model_prefix = "spm_model"   # 出力: spm_model.model, spm_model.vocab
vocab_size = 32000
model_type = "unigram"      # 'unigram' or 'bpe' or 'word' or 'char'
character_coverage = 0.9995 # 日本語なら 1.0 / 0.9995 など

# --- 学習 ---
spm.SentencePieceTrainer.Train(
    input=input_files,
    model_prefix=model_prefix,
    vocab_size=vocab_size,
    character_coverage=character_coverage,
    model_type=model_type,
    user_defined_symbols=["<s>","</s>","<pad>","<unk>"]  # 必要なら
)
print("trained:", model_prefix + ".model")
```

### 使い方（ロード・エンコード）

```python
import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file="spm_model.model")

text = "人工知能と機械学習は面白い。"
pieces = sp.encode_as_pieces(text)
ids = sp.encode_as_ids(text)
print("pieces:", pieces)
print("ids:", ids)
print("decoded:", sp.decode_ids(ids))
```

---

## B. Hugging Face `tokenizers`（より柔軟・高速）

### インストール

```bash
pip install tokenizers
```

### 1) Byte-Level BPE（GPTスタイル）

```python
from tokenizers import ByteLevelBPETokenizer
from pathlib import Path

# 用意: data/*.txt
paths = [str(p) for p in Path("data").glob("*.txt")]

tokenizer = ByteLevelBPETokenizer()

# 学習 (vocab_size, special_tokens)
tokenizer.train(files=paths, vocab_size=50000, min_frequency=2,
                special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])

# 保存
tokenizer.save_model(".", "my_bytebpe")

# 使用例
enc = tokenizer.encode("人工知能が進化しています。")
print(enc.tokens)
print(enc.ids)
```

### 2) Unigram（SentencePieceと同じ発想） via `tokenizers`

```python
from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.trainers import UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace

paths = [str(p) for p in Path("data").glob("*.txt")]

tokenizer = Tokenizer(Unigram())
tokenizer.pre_tokenizer = Whitespace()  # 生データを空白単位で分割して候補を作る場合
trainer = UnigramTrainer(vocab_size=32000, special_tokens=["<s>", "<pad>", "</s>", "<unk>"])

tokenizer.train(paths, trainer)
tokenizer.save("tokenizer-unigram.json")

# 使用例
encoded = tokenizer.encode("自然言語処理を学ぶ。")
print(encoded.tokens)
```

### 3) WordPiece（BERT系）

```python
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
tokenizer.normalizer = NFKC()
tokenizer.pre_tokenizer = Whitespace()
trainer = WordPieceTrainer(vocab_size=30000, special_tokens=["[PAD]","[UNK]","[CLS]","[SEP]","[MASK]"])

paths = [str(p) for p in Path("data").glob("*.txt")]
tokenizer.train(paths, trainer)
tokenizer.save("tokenizer-wordpiece.json")
```

---

## 実務の注意点（ベストプラクティス）

* **コーパスは前処理を丁寧に** ：HTMLタグ除去、正規化（Unicode NFKC）、不要行削除など。
* **vocab_size** は 8k〜100k の間で用途に応じて。日本語なら 8k〜32k が多い。
* **special tokens** （`<pad>` `<unk>` `<s>` `</s>`）は必ず設計しておく。
* **byte-level** （バイト単位）を使うと絵文字やコードに強い（GPT系推奨）。
* SentencePiece の `character_coverage` を日本語なら 1.0 に近く設定。
* 学習は CPUでも可能だが大規模コーパスは時間がかかる。

---

## 参考：小さなデータで試すフルワークフロー（まとめコード）

必要なら「1ファイルで全部やる」サンプルも提示します（学習→保存→ロード→比較）。作りますか？

どの方式を本格的に使いたいですか？（`sentencepiece` / `ByteLevelBPE` / `Unigram` / `WordPiece`）

希望に合わせて、Colabノート風に実行セルを整えます。
