import re
from abc import ABC, abstractmethod


class Tokenizer(ABC):

    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        """Split text into a list of tokens."""
        ...

    @abstractmethod
    def build_vocab(self, corpus: list[str], min_count: int = 5) -> None:
        """Build vocabulary from a list of texts."""
        ...

    @property
    @abstractmethod
    def vocab(self) -> dict[str, int]:
        """Return the token to index mapping."""
        ...

    @property
    @abstractmethod
    def counts(self) -> dict[str, int]:
        """Return the raw token frequency counts."""
        ...

    def encode(self, text: str) -> list[int]:
        tokens = self.tokenize(text)
        return [self.vocab[t] for t in tokens if t in self.vocab]


class WordTokenizer(Tokenizer):
    """Lowercases and extracts word tokens, keeping internal apostrophes for contractions."""

    # matches contractions and plain words, trailing apostrophes excluded
    _TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)*")

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._counts: dict[str, int] = {}

    def tokenize(self, text: str) -> list[str]:
        return self._TOKEN_RE.findall(text.lower())

    def build_vocab(self, corpus: list[str], min_count: int = 5) -> None:
        self._counts = {}
        for text in corpus:
            for token in self.tokenize(text):
                self._counts[token] = self._counts.get(token, 0) + 1
        filtered = {w: c for w, c in self._counts.items() if c >= min_count}
        self._vocab = {word: idx for idx, word in enumerate(sorted(filtered))}

    @property
    def vocab(self) -> dict[str, int]:
        return self._vocab

    @property
    def counts(self) -> dict[str, int]:
        return self._counts