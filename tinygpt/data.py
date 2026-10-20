"""Download TinyStories, train a byte-level BPE tokenizer, and write token ids to train.bin / val.bin.

Usage:
    python -m tinygpt.data --out data                 # full dataset (Kaggle)
    python -m tinygpt.data --out data_small --limit 20000   # quick local test
"""
import argparse
import json
import os

import numpy as np

EOS = "<|endoftext|>"


def train_tokenizer(texts, vocab_size):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOS],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(texts, trainer)
    return tok


def encode_to_bin(tok, texts, path, batch_size=10_000):
    """Encode stories, end each with EOS, and save one flat uint16 array."""
    eos = tok.token_to_id(EOS)
    chunks = []
    for i in range(0, len(texts), batch_size):
        for enc in tok.encode_batch(texts[i : i + batch_size]):
            chunks.append(np.array(enc.ids + [eos], dtype=np.uint16))
    ids = np.concatenate(chunks)
    ids.tofile(path)
    return len(ids)


def load_tokenizer(data_dir):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(os.path.join(data_dir, "tokenizer.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N training stories")
    ap.add_argument("--tokenizer-stories", type=int, default=200_000, help="stories used to train BPE")
    args = ap.parse_args()

    from datasets import load_dataset

    os.makedirs(args.out, exist_ok=True)
    ds = load_dataset("roneneldan/TinyStories")
    train = ds["train"]["text"]
    val = ds["validation"]["text"]
    if args.limit:
        train = train[: args.limit]
        val = val[: max(1000, args.limit // 20)]
    train = [t.strip() for t in train]
    val = [t.strip() for t in val]

    print(f"training BPE (vocab {args.vocab}) on {min(len(train), args.tokenizer_stories):,} stories")
    tok = train_tokenizer(train[: args.tokenizer_stories], args.vocab)
    tok.save(os.path.join(args.out, "tokenizer.json"))

    n_train = encode_to_bin(tok, train, os.path.join(args.out, "train.bin"))
    n_val = encode_to_bin(tok, val, os.path.join(args.out, "val.bin"))
    meta = {
        "vocab_size": tok.get_vocab_size(),
        "eos_id": tok.token_to_id(EOS),
        "train_stories": len(train),
        "val_stories": len(val),
        "train_tokens": n_train,
        "val_tokens": n_val,
        "tokens_per_story": round(n_train / len(train), 1),
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
