from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from tokenizers.normalizers import NFD, StripAccents, Sequence
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing


def train_bpe_tokenizer(
        files,
        vocab_size=20000,
        save_path="bpe-tokenizer.json",
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[MASK]", "[SEP]"]
):
    # BPE tokenizer
    tokenizer = Tokenizer(models.BPE())

    # Normalization: unicode canonical form + accents strip
    tokenizer.normalizer = Sequence([
        NFD(),
        StripAccents()
    ])
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=special_tokens
    )

    # Training
    tokenizer.train(files, trainer)

    # Add post-processing <CLS> ... <SEP>
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $0 [SEP]",
        pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[("[CLS]", 2), ("[SEP]", 4)]
    )

    tokenizer.decoder = decoders.BPEDecoder()

    tokenizer.save(save_path)
    print(f"Tokenizer saved to {save_path}")


if __name__ == "__main__":
    train_bpe_tokenizer(
        files=["dataset.txt"],   # 学習用テキスト
        vocab_size=20000,
        save_path="rope_sparse_tokenizer.json"
    )