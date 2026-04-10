"""
Tokenizer implementations for the LLM training pipeline.

To use a tokenizer:
    from tokenizers import CharacterTokenizer, WordTokenizer, BPETokenizer

    tokenizer = CharacterTokenizer(text)
    # or
    tokenizer = WordTokenizer(text)
    # or
    tokenizer = BPETokenizer.from_pretrained("gpt2")

    encoded = tokenizer.encode("hello world")
    decoded = tokenizer.decode(encoded)
"""

from abc import ABC, abstractmethod
from typing import List


class Tokenizer(ABC):
    """Abstract base class for all tokenizers."""

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Convert text to a list of token IDs."""
        pass

    @abstractmethod
    def decode(self, ids: List[int]) -> str:
        """Convert a list of token IDs back to text."""
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        pass

    def save(self, path: str):
        """Save tokenizer state to a file."""
        import json
        state = self._get_state()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> 'Tokenizer':
        """Load tokenizer state from a file."""
        import json
        with open(path, 'r', encoding='utf-8') as f:
            state = json.load(f)
        return cls._from_state(state)

    def _get_state(self) -> dict:
        """Override to define serialization."""
        raise NotImplementedError("Subclass must implement _get_state")

    @classmethod
    def _from_state(cls, state: dict) -> 'Tokenizer':
        """Override to define deserialization."""
        raise NotImplementedError("Subclass must implement _from_state")


class CharacterTokenizer(Tokenizer):
    """Character-level tokenizer. Each unique character is a token."""

    def __init__(self, text: str = None, vocab: List[str] = None):
        if vocab is not None:
            self._vocab = vocab
        elif text is not None:
            self._vocab = sorted(list(set(text)))
        else:
            raise ValueError("Must provide either text or vocab")

        self._stoi = {ch: i for i, ch in enumerate(self._vocab)}
        self._itos = {i: ch for i, ch in enumerate(self._vocab)}

    def encode(self, text: str) -> List[int]:
        return [self._stoi[c] for c in text]

    def decode(self, ids: List[int]) -> str:
        return ''.join(self._itos[i] for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def stoi(self) -> dict:
        """String to index mapping."""
        return self._stoi

    @property
    def itos(self) -> dict:
        """Index to string mapping."""
        return self._itos

    def _get_state(self) -> dict:
        return {'type': 'character', 'vocab': self._vocab}

    @classmethod
    def _from_state(cls, state: dict) -> 'CharacterTokenizer':
        return cls(vocab=state['vocab'])


class WordTokenizer(Tokenizer):
    """Word-level tokenizer. Splits on whitespace."""

    def __init__(self, text: str = None, vocab: List[str] = None, unk_token: str = "<UNK>"):
        self._unk_token = unk_token

        if vocab is not None:
            self._vocab = vocab
        elif text is not None:
            words = text.split()
            self._vocab = [unk_token] + sorted(set(words))
        else:
            raise ValueError("Must provide either text or vocab")

        self._stoi = {w: i for i, w in enumerate(self._vocab)}
        self._itos = {i: w for i, w in enumerate(self._vocab)}
        self._unk_id = self._stoi[unk_token]

    def encode(self, text: str) -> List[int]:
        return [self._stoi.get(w, self._unk_id) for w in text.split()]

    def decode(self, ids: List[int]) -> str:
        return ' '.join(self._itos.get(i, self._unk_token) for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def _get_state(self) -> dict:
        return {'type': 'word', 'vocab': self._vocab, 'unk_token': self._unk_token}

    @classmethod
    def _from_state(cls, state: dict) -> 'WordTokenizer':
        return cls(vocab=state['vocab'], unk_token=state['unk_token'])


class BPETokenizer(Tokenizer):
    """
    Wrapper for HuggingFace tokenizers library.

    Requires: pip install tokenizers

    Usage:
        tokenizer = BPETokenizer.from_pretrained("gpt2")
        # or train your own:
        tokenizer = BPETokenizer.train(texts, vocab_size=1000)
    """

    def __init__(self, hf_tokenizer):
        self._tokenizer = hf_tokenizer

    @classmethod
    def from_pretrained(cls, name: str) -> 'BPETokenizer':
        """Load a pretrained tokenizer from HuggingFace."""
        try:
            from tokenizers import Tokenizer as HFTokenizer
            hf_tok = HFTokenizer.from_pretrained(name)
            return cls(hf_tok)
        except ImportError:
            raise ImportError("Install tokenizers: pip install tokenizers")

    @classmethod
    def train(cls, texts: List[str], vocab_size: int = 1000, min_frequency: int = 2) -> 'BPETokenizer':
        """Train a new BPE tokenizer on the provided texts."""
        try:
            from tokenizers import Tokenizer as HFTokenizer
            from tokenizers.models import BPE
            from tokenizers.trainers import BpeTrainer
            from tokenizers.pre_tokenizers import Whitespace

            tokenizer = HFTokenizer(BPE(unk_token="<UNK>"))
            tokenizer.pre_tokenizer = Whitespace()

            trainer = BpeTrainer(
                vocab_size=vocab_size,
                min_frequency=min_frequency,
                special_tokens=["<UNK>", "<PAD>", "<BOS>", "<EOS>"]
            )
            tokenizer.train_from_iterator(texts, trainer)
            return cls(tokenizer)
        except ImportError:
            raise ImportError("Install tokenizers: pip install tokenizers")

    def encode(self, text: str) -> List[int]:
        return self._tokenizer.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        return self._tokenizer.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    def save(self, path: str):
        self._tokenizer.save(path)

    @classmethod
    def load(cls, path: str) -> 'BPETokenizer':
        try:
            from tokenizers import Tokenizer as HFTokenizer
            hf_tok = HFTokenizer.from_file(path)
            return cls(hf_tok)
        except ImportError:
            raise ImportError("Install tokenizers: pip install tokenizers")


class SentencePieceTokenizer(Tokenizer):
    """
    Wrapper for SentencePiece tokenizer.

    Requires: pip install sentencepiece

    Usage:
        tokenizer = SentencePieceTokenizer.train("input.txt", vocab_size=1000)
        # or load existing:
        tokenizer = SentencePieceTokenizer.load("model.model")
    """

    def __init__(self, sp_model):
        self._sp = sp_model

    @classmethod
    def train(cls, input_file: str, model_prefix: str = "spm", vocab_size: int = 1000) -> 'SentencePieceTokenizer':
        """Train a new SentencePiece model."""
        try:
            import sentencepiece as spm
            spm.SentencePieceTrainer.train(
                input=input_file,
                model_prefix=model_prefix,
                vocab_size=vocab_size,
                model_type='bpe'
            )
            sp = spm.SentencePieceProcessor()
            sp.load(f"{model_prefix}.model")
            return cls(sp)
        except ImportError:
            raise ImportError("Install sentencepiece: pip install sentencepiece")

    @classmethod
    def load(cls, model_path: str) -> 'SentencePieceTokenizer':
        """Load an existing SentencePiece model."""
        try:
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor()
            sp.load(model_path)
            return cls(sp)
        except ImportError:
            raise ImportError("Install sentencepiece: pip install sentencepiece")

    def encode(self, text: str) -> List[int]:
        return self._sp.encode_as_ids(text)

    def decode(self, ids: List[int]) -> str:
        return self._sp.decode_ids(ids)

    @property
    def vocab_size(self) -> int:
        return self._sp.get_piece_size()
