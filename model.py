import numpy as np

from dataset import SkipGramDataset, CBOWDataset, NoiseSampler
from tokenizer import Tokenizer
from abc import ABC, abstractmethod

class Word2Vec(ABC):
    @abstractmethod
    def fit(self, corpus: list[str], tokenizer: Tokenizer, epochs: int) -> None: ...

    @abstractmethod
    def train_step(self, *args) -> float: ...

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))

    @property
    @abstractmethod
    def embeddings(self) -> np.ndarray: ...
    

class SkipGramWord2Vec(Word2Vec):
    """
    Skipgram Word2Vec with negative sampling.
    Call fit to train on a corpus.

    W_in shape embed_dim by vocab_size, center word embeddings.
    W_out shape embed_dim by vocab_size, context word embeddings.

    Sparse SGD, only the touched columns are updated per step.
    """

    def __init__(
        self,
        embed_dim: int  = 100,
        window: int  = 2,
        neg_k: int  = 5,
        subsample_t: float = 1e-5,
        lr: float = 0.025,
    ) -> None:
        self.embed_dim = embed_dim
        self.window = window
        self.neg_k = neg_k
        self.subsample_t = subsample_t
        self.lr = lr

        self.W_in: np.ndarray | None = None
        self.W_out: np.ndarray | None = None

    def fit(self, corpus: list[str], tokenizer: Tokenizer, epochs: int = 5) -> None:
        tokenizer.build_vocab(corpus)
        vocab_size = len(tokenizer.vocab)

        scale = 0.5 / self.embed_dim
        self.W_in  = np.random.uniform(-scale, scale, (self.embed_dim, vocab_size))
        self.W_out = np.zeros((self.embed_dim, vocab_size))

        ds = SkipGramDataset(tokenizer, window=self.window, subsample_t=self.subsample_t)
        ds.build(corpus)  # one build up-front to estimate pair count for LR schedule
        N_est = len(ds.pairs)
        total_steps = N_est * epochs

        sampler = NoiseSampler(tokenizer.vocab, tokenizer.counts)
        print(f"\nTraining  vocab={vocab_size:,}  pairs≈{N_est:,}  "
              f"epochs={epochs}  embed_dim={self.embed_dim}  lr={self.lr}\n")

        global_step = 0
        for epoch in range(1, epochs + 1):
            ds.build(corpus)  # fresh subsampling + window draws each epoch
            pairs = ds.pairs
            N = len(pairs)
            perm = np.random.permutation(N).tolist()
            total_loss = 0.0

            for step, idx in enumerate(perm):
                lr_t = max(self.lr * (1 - global_step / total_steps), self.lr * 1e-4)

                center_idx, context_idx = pairs[idx]
                neg_indices = sampler.sample(self.neg_k, exclude=context_idx)
                loss = self.train_step(center_idx, context_idx, neg_indices, lr_t)
                total_loss += loss
                global_step += 1

                if (step + 1) % 10_000 == 0 or (step + 1) == N:
                    pct = (step + 1) / N * 100
                    bar_filled = int(pct // 5)
                    bar = "█" * bar_filled + "░" * (20 - bar_filled)
                    print(
                        f"\r  epoch {epoch}/{epochs}  [{bar}] {pct:5.1f}%  "
                        f"avg_loss={total_loss / (step + 1):.4f}  lr={lr_t:.6f}",
                        end="", flush=True,
                    )

            print(f"\repoch {epoch}/{epochs}  {N:,} pairs  avg_loss={total_loss / N:.4f}")

    def train_step(
        self,
        center_idx: int,
        context_idx: int,
        neg_indices: np.ndarray,
        lr: float,
    ) -> float:
        # lookup
        v     = self.W_in[:, center_idx]
        u_pos = self.W_out[:, context_idx]
        u_neg = self.W_out[:, neg_indices]

        # scores
        pos_score = np.dot(v, u_pos)
        neg_scores = u_neg.T @ v

        # loss
        pos_sig = self._sigmoid(pos_score)
        neg_sig = self._sigmoid(-neg_scores)
        loss = -np.log(pos_sig + 1e-10) - np.sum(np.log(neg_sig + 1e-10))

        # gradients
        d_pos = pos_sig - 1.0
        d_neg = self._sigmoid(neg_scores)

        grad_v     = d_pos * u_pos + u_neg @ d_neg
        grad_u_pos = d_pos * v
        grad_u_neg = np.outer(v, d_neg)

        # sparse SGD, only touched columns are updated
        self.W_in[:, center_idx] -= lr * grad_v
        self.W_out[:, context_idx] -= lr * grad_u_pos
        self.W_out[:, neg_indices] -= lr * grad_u_neg

        return float(loss)

    @property
    def embeddings(self) -> np.ndarray:
        """Center word embeddings, shape vocab_size by embed_dim."""
        return self.W_in.T


class CBOWWord2Vec(Word2Vec):
    """
    CBOW Word2Vec.
    Call fit to train on a corpus.

    W_in shape embed_dim by vocab_size, context word embeddings.
    W_out shape embed_dim by vocab_size, center word embeddings.
    """

    def __init__(
        self,
        embed_dim: int = 100,
        window: int = 2,
        neg_k: int = 5,
        subsample_t: float = 1e-5,
        lr: float = 0.025,
    ) -> None:
        self.embed_dim = embed_dim
        self.window = window
        self.neg_k = neg_k
        self.subsample_t = subsample_t
        self.lr = lr

        self.W_in: np.ndarray | None = None
        self.W_out: np.ndarray | None = None

    def fit(self, corpus: list[str], tokenizer: Tokenizer, epochs: int = 5) -> None:
        tokenizer.build_vocab(corpus)
        vocab_size = len(tokenizer.vocab)

        scale = 0.5 / self.embed_dim
        self.W_in  = np.random.uniform(-scale, scale, (self.embed_dim, vocab_size))
        self.W_out = np.zeros((self.embed_dim, vocab_size))

        ds = CBOWDataset(tokenizer, window=self.window, subsample_t=self.subsample_t)
        ds.build(corpus)  # one build up-front to estimate pair count for LR schedule
        N_est = len(ds.pairs)
        total_steps = N_est * epochs

        sampler = NoiseSampler(tokenizer.vocab, tokenizer.counts)
        print(f"\nTraining  vocab={vocab_size:,}  pairs≈{N_est:,}  "
              f"epochs={epochs}  embed_dim={self.embed_dim}  lr={self.lr}\n")

        global_step = 0
        for epoch in range(1, epochs + 1):
            ds.build(corpus)  # fresh subsampling + window draws each epoch
            pairs = ds.pairs
            N = len(pairs)
            perm = np.random.permutation(N).tolist()
            total_loss = 0.0

            for step, idx in enumerate(perm):
                lr_t = max(self.lr * (1 - global_step / total_steps), self.lr * 1e-4)

                context_indices, center_idx = pairs[idx]
                neg_indices = sampler.sample(self.neg_k, exclude=center_idx)
                loss = self.train_step(context_indices, center_idx, neg_indices, lr_t)
                total_loss += loss
                global_step += 1

                if (step + 1) % 10_000 == 0 or (step + 1) == N:
                    pct = (step + 1) / N * 100
                    bar_filled = int(pct // 5)
                    bar = "█" * bar_filled + "░" * (20 - bar_filled)
                    print(
                        f"\r  epoch {epoch}/{epochs}  [{bar}] {pct:5.1f}%  "
                        f"avg_loss={total_loss / (step + 1):.4f}  lr={lr_t:.6f}",
                        end="", flush=True,
                    )

            print(f"\repoch {epoch}/{epochs}  {N:,} pairs  avg_loss={total_loss / N:.4f}")

    def train_step(
        self,
        context_indices: list[int],
        center_idx: int,
        neg_indices: np.ndarray,
        lr: float,
    ) -> float:
        # hidden layer, mean of context embeddings
        h     = self.W_in[:, context_indices].mean(axis=1)
        u_pos = self.W_out[:, center_idx]
        u_neg = self.W_out[:, neg_indices]

        # scores
        pos_score = np.dot(h, u_pos)
        neg_scores = u_neg.T @ h

        # loss
        pos_sig = self._sigmoid(pos_score)
        neg_sig = self._sigmoid(-neg_scores)
        loss = -np.log(pos_sig + 1e-10) - np.sum(np.log(neg_sig + 1e-10))

        # gradients
        d_pos = pos_sig - 1.0
        d_neg = self._sigmoid(neg_scores)

        grad_h     = d_pos * u_pos + u_neg @ d_neg
        grad_u_pos = d_pos * h
        grad_u_neg = np.outer(h, d_neg)

        # distribute grad equally to all context columns, chain rule through mean
        self.W_in[:, context_indices] -= lr * (grad_h / len(context_indices))[:, None]
        self.W_out[:, center_idx] -= lr * grad_u_pos
        self.W_out[:, neg_indices] -= lr * grad_u_neg

        return float(loss)

    @property
    def embeddings(self) -> np.ndarray:
        """Context word embeddings, shape vocab_size by embed_dim."""
        return self.W_in.T
