"""Deterministic tokenization, training vocabulary, and bounded feature sequences."""
import json
import re
from collections import Counter
from pathlib import Path

SPECIAL_TOKENS = ("<PAD>", "<UNK>", "<SEP>", "<EMPTY>", "<NL>")
PAD_ID, UNK_ID, SEP_ID, EMPTY_ID, NL_ID = range(5)
INPUT_MODES = ("input_only", "output_only", "input_output")
TOKEN_PATTERN = re.compile(r"\n|\w+|[^\w\s]", flags=re.UNICODE)


def tokenize(text, lowercase=True):
    if not isinstance(text, str):
        raise TypeError("Text must be a string")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if lowercase:
        text = text.lower()
    return ["<NL>" if token == "\n" else token for token in TOKEN_PATTERN.findall(text)]


class Vocabulary:
    def __init__(self, tokens, lowercase=True):
        if list(tokens[:5]) != list(SPECIAL_TOKENS) or len(tokens) != len(set(tokens)):
            raise ValueError("Vocabulary must start with unique reserved tokens")
        self.tokens = list(tokens)
        self.token_to_id = {token: i for i, token in enumerate(tokens)}
        self.lowercase = lowercase

    def __len__(self):
        return len(self.tokens)

    @classmethod
    def fit(cls, training_texts, max_size=20000, lowercase=True):
        """Caller must supply training features only; never validation/test text."""
        if max_size < len(SPECIAL_TOKENS):
            raise ValueError("max_size must include reserved tokens")
        counts = Counter(token for text in training_texts for token in tokenize(text, lowercase))
        for token in SPECIAL_TOKENS:
            counts.pop(token, None)
        ordered = sorted(counts, key=lambda token: (-counts[token], token))
        return cls([*SPECIAL_TOKENS, *ordered[:max_size-len(SPECIAL_TOKENS)]], lowercase)

    def encode(self, text, max_tokens):
        if max_tokens < 1:
            raise ValueError("Token limit must be positive")
        tokens = tokenize(text, self.lowercase)[:max_tokens]
        return [self.token_to_id.get(token, UNK_ID) for token in tokens] or [EMPTY_ID]

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"tokens": self.tokens, "lowercase": self.lowercase}, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(payload["tokens"], payload["lowercase"])


def feature_texts(records, mode):
    if mode not in INPUT_MODES:
        raise ValueError(f"Unknown input mode: {mode}")
    for record in records:
        if mode in ("input_only", "input_output"):
            yield record["LLM_Input"]
        if mode in ("output_only", "input_output"):
            yield record["LLM_output"]


def encode_record(record, vocabulary, mode, text_config):
    if mode not in INPUT_MODES:
        raise ValueError(f"Unknown input mode: {mode}")
    if mode == "input_only":
        return vocabulary.encode(record["LLM_Input"], text_config["prompt_max_tokens"])
    if mode == "output_only":
        return vocabulary.encode(record["LLM_output"], text_config["output_max_tokens"])
    return (vocabulary.encode(record["LLM_Input"], text_config["prompt_max_tokens"])
            + [SEP_ID]
            + vocabulary.encode(record["LLM_output"], text_config["output_max_tokens"]))
