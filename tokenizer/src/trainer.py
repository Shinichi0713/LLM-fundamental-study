from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# 1. 空のBPEモデルを初期化
# [UNK] は未知語（語彙に含まれない文字）を指します
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# 2. 事前トークナイザの設定
# 学習前に、まずは空白（スペース）でテキストを区切る設定にします
tokenizer.pre_tokenizer = Whitespace()

# 3. トレーナー（学習の設定）を定義
# vocab_size: 作成する語彙の総数。初学者は5000〜30000程度で試すのがおすすめ
# special_tokens: 予約済みの特殊な意味を持つトークン
trainer = BpeTrainer(
    vocab_size=10000, 
    show_progress=True,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)

# 4. 学習の実行
# files には学習に使うテキストファイルのリストを指定します
files = ["data/my_corpus.txt"]
tokenizer.train(files, trainer)

# 5. 学習したトークナイザを保存
tokenizer.save("my_tokenizer.json")

print("トークナイザの学習が完了し、保存されました。")

# --- 動作確認 ---
output = tokenizer.encode("Hello, how are you learning AI?")
print("トークンID:", output.ids)
print("トークン文字列:", output.tokens)