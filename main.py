import os
import re

import numpy as np

from tokenizer import WordTokenizer
from model import SkipGramWord2Vec, CBOWWord2Vec

BROWN_DIR   = "./brown"

def save_bin(embeddings: np.ndarray, tokenizer, path: str) -> None:
    vocab_size, embed_dim = embeddings.shape
    idx2word = {i: w for w, i in tokenizer.vocab.items()}
    with open(path, "wb") as f:
        f.write(f"{vocab_size} {embed_dim}\n".encode())
        for i in range(vocab_size):
            vec = np.asarray(embeddings[i], dtype=np.float32)
            f.write(idx2word[i].encode() + b" " + vec.tobytes())
    print(f"Saved {vocab_size:,} vectors to {path}")


def load_brown(directory: str) -> list[str]:
    tag_re = re.compile(r"/\S+")
    sentences = []
    for fname in sorted(os.listdir(directory)):
        fpath = os.path.join(directory, fname)
        if not os.path.isfile(fpath):
            continue
        with open(fpath, encoding="latin-1") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                plain = tag_re.sub("", line).strip()
                if plain:
                    sentences.append(plain)
    return sentences


if __name__ == "__main__":
    print("Loading Brown corpus...")
    corpus = load_brown(BROWN_DIR)
    print(f"  {len(corpus):,} sentences")

    tok = WordTokenizer()

    print("Skip-gram")
    sg = SkipGramWord2Vec(embed_dim=100, window=5, neg_k=5, subsample_t=1e-5, lr=0.025)
    sg.fit(corpus, tok, epochs=5)
    E_sg = sg.embeddings
    save_bin(E_sg, tok, "vectors_sg.bin")

    print("CBOW")
    cbow = CBOWWord2Vec(embed_dim=100, window=5, neg_k=5, subsample_t=1e-5, lr=0.025)
    cbow.fit(corpus, tok, epochs=5)
    E_cbow = cbow.embeddings
    save_bin(E_cbow, tok, "vectors_cbow.bin")
