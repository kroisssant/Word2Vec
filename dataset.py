import numpy as np

from tokenizer import Tokenizer

from abc import ABC, abstractmethod

class Dataset(ABC):
    _keep_probs: np.ndarray | None = None

    @abstractmethod
    def build(self, corpus: list[str]): ...

    def _build_keep_probs(self, tokenizer: Tokenizer, subsample_t: float | None) -> None:
        if subsample_t is None:
            self._keep_probs = None
            return
        counts = tokenizer.counts
        total = sum(counts.values())
        vocab = tokenizer.vocab
        probs = np.zeros(len(vocab))
        for word, idx in vocab.items():
            f = counts.get(word, 0) / total
            probs[idx] = (np.sqrt(f / subsample_t) + 1) * (subsample_t / f)
        self._keep_probs = probs

    def _subsample(self, indices: list[int]) -> list[int]:
        keep = np.random.random(len(indices)) < self._keep_probs[indices]
        return [idx for idx, k in zip(indices, keep) if k]

class SkipGramDataset(Dataset):
    """
    Generates center, context index pairs from a corpus for skipgram Word2Vec.

    For each token, every token within the window around it is a context target.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        window: int = 2,
        subsample_t: float | None = 1e-5,
    ) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.tokenizer = tokenizer
        self.window = window
        self.subsample_t = subsample_t
        self._pairs: list[tuple[int, int]] = []

    def build(self, corpus: list[str]) -> None:
        self._build_keep_probs(self.tokenizer, self.subsample_t)
        self._pairs = []
        for text in corpus:
            indices = self.tokenizer.encode(text)
            if self._keep_probs is not None:
                indices = self._subsample(indices)
            self._pairs.extend(self._extract_pairs(indices))

    def _extract_pairs(self, indices: list[int]) -> list[tuple[int, int]]:
        pairs = []
        n = len(indices)
        for i, center in enumerate(indices):
            w  = int(np.random.randint(1, self.window + 1))
            start = max(0, i - w)
            end  = min(n, i + w + 1)
            for j in range(start, end):
                if j != i:
                    pairs.append((center, indices[j]))
        return pairs

    @property
    def pairs(self) -> list[tuple[int, int]]:
        return self._pairs

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> tuple[int, int]:
        return self._pairs[idx]

class CBOWDataset(Dataset):
    """
    Generates context, center tuples from a corpus for CBOW Word2Vec.

    For each token, all tokens within the window around it are grouped as context.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        window: int = 2,
        subsample_t: float | None = 1e-5,
    ) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.tokenizer = tokenizer
        self.window = window
        self.subsample_t = subsample_t
        self._pairs: list[tuple[list[int], int]] = []

    def build(self, corpus: list[str]) -> None:
        self._build_keep_probs(self.tokenizer, self.subsample_t)
        self._pairs = []
        for text in corpus:
            indices = self.tokenizer.encode(text)
            if self._keep_probs is not None:
                indices = self._subsample(indices)
            self._pairs.extend(self._extract_pairs(indices))

    def _extract_pairs(self, indices: list[int]) -> list[tuple[list[int], int]]:
        pairs = []
        n = len(indices)
        for i, center in enumerate(indices):
            w  = int(np.random.randint(1, self.window + 1))
            start = max(0, i - w)
            end  = min(n, i + w + 1)
            ctx = [indices[j] for j in range(start, end) if j != i]
            if ctx:
                pairs.append((ctx, center))
        return pairs

    @property
    def pairs(self) -> list[tuple[list[int], int]]:
        return self._pairs

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> tuple[list[int], int]:
        return self._pairs[idx]
    
class NoiseSampler:
    """
    Unigram noise distribution raised to the 3/4 power for negative sampling.
    """

    def __init__(
        self,
        vocab: dict[str, int],
        counts: dict[str, int],
        power: float = 0.75,
    ) -> None:
        noise = np.zeros(len(vocab))
        for word, idx in vocab.items():
            noise[idx] = counts.get(word, 0) ** power
        self._probs = noise / noise.sum()

    def sample_batch(self, k: int, excludes: np.ndarray) -> np.ndarray:
        """Sample k negatives for each entry in excludes."""
        return np.stack([self.sample(k, int(ex)) for ex in excludes])

    def sample(self, k: int, exclude: int) -> np.ndarray:
        """
        Draw k negative sample indices, guaranteed to differ from exclude.
        Draws slightly more than needed in one call to handle rejections.
        """
        result = np.empty(k, dtype=np.int32)
        filled = 0
        while filled < k:
            draw = np.random.choice(
                len(self._probs),
                size=(k - filled) * 2,
                p=self._probs,
            )
            valid = draw[draw != exclude]
            take = min(len(valid), k - filled)
            result[filled : filled + take] = valid[:take]
            filled += take
        return result
