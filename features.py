import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Dict, Optional, Any

VOWELS = set("aeiou")

FIL_PREFIXES = ("nag", "mag", "pag", "nakaka", "naka", "pina", "ipina",
                "napaka", "pinaka", "um", "in", "ka", "ma", "na")

ENG_SUFFIXES = ("tion", "sion", "ing", "ed", "ly", "ment", "ness", "ous",
                "ive", "ance", "ence", "ational")
FIL_SUFFIXES = ("han", "hin", "syon", "in", "an", "ng")

NON_NATIVE_LETTERS = "cfjqvxz"

ENG_CLUSTERS = ("ck", "sch", "str", "th", "ght", "ph", "sh")

MIN_STEM_LEN = 3

ENG_SUFFIXES_BY_LEN = tuple(sorted(ENG_SUFFIXES, key=len, reverse=True))
FIL_SUFFIXES_BY_LEN = tuple(sorted(FIL_SUFFIXES, key=len, reverse=True))
FIL_PREFIXES_BY_LEN = tuple(sorted(FIL_PREFIXES, key=len, reverse=True))


class Context:

    def __init__(self, tokens: Optional[List[str]] = None, index: int = 0):
        self.tokens = tokens if tokens is not None else []
        self.index = index

    def prev(self) -> Optional[str]:
        i = self.index - 1
        return self.tokens[i] if 0 <= i < len(self.tokens) else None

    def next(self) -> Optional[str]:
        i = self.index + 1
        return self.tokens[i] if 0 <= i < len(self.tokens) else None


class Sparse32Caster(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if sp.issparse(X):
            X.indices = X.indices.astype(np.int32)
            X.indptr = X.indptr.astype(np.int32)
        return X


def char_ngrams(text: str, n: int) -> List[str]:
    if not text:
        return []
    padded = "^" + text + "$"
    return [padded[i:i + n] for i in range(len(padded) - n + 1)]


def _ratio(count: int, total: int) -> float:
    return count / total if total else 0.0


def _longest_run(letters: List[str], want_vowel: bool) -> int:
    best = cur = 0
    for c in letters:
        is_vowel = c in VOWELS
        if is_vowel == want_vowel:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _has_reduplication(low: str) -> bool:
    if "-" in low:
        parts = low.split("-")
        if len(parts) == 2 and parts[0] and parts[0] == parts[1]:
            return True
    n = len(low)
    if n >= 4 and n % 2 == 0 and low[:n // 2] == low[n // 2:]:
        return True
    return False


def _longest_fil_prefix(low: str, min_stem: int = 0) -> Optional[str]:
    for p in FIL_PREFIXES_BY_LEN:
        if low.startswith(p) and len(low) - len(p) >= min_stem:
            return p
    return None


def _looks_english(stem: str) -> bool:
    if not stem:
        return False
    return (any(c in NON_NATIVE_LETTERS for c in stem)
            or any(cl in stem for cl in ENG_CLUSTERS)
            or stem.endswith(ENG_SUFFIXES))


def _cue_summary(token: str) -> Dict[str, Any]:
    low = token.lower()
    letters = [c for c in low if c.isalpha()]
    n_letters = len(letters)
    n_vowels = sum(1 for c in letters if c in VOWELS)
    return {
        "fil_prefix": _longest_fil_prefix(low, MIN_STEM_LEN) is not None,
        "eng_suffix": low.endswith(ENG_SUFFIXES),
        "fil_suffix": low.endswith(FIL_SUFFIXES),
        "has_non_native": any(c in NON_NATIVE_LETTERS for c in low),
        "has_eng_cluster": any(cl in low for cl in ENG_CLUSTERS),
        "vowel_ratio": _ratio(n_vowels, n_letters),
        "ends_in_ng": low.endswith("ng"),
        "ends_in_vowel": bool(letters) and letters[-1] in VOWELS,
        "is_punct": n_letters == 0 and len(low) > 0,
        "suffix3": low[-3:] if n_letters >= 3 else "",
    }


def extract_features(token: str,
                     context: Optional[Context] = None,
                     ngram_sizes=(2, 3),
                     neighbour_cues: bool = False,
                     cs_cue: bool = True,
                     clean_affixes: bool = True,
                     affix_strings: bool = True) -> Dict[str, Any]:
    feats: Dict[str, Any] = {}
    raw = token
    low = token.lower()
    n_chars = len(low)

    letters = [c for c in low if c.isalpha()]
    n_letters = len(letters)
    n_vowels = sum(1 for c in letters if c in VOWELS)
    n_cons = n_letters - n_vowels

    feats["length"] = n_chars
    feats["n_letters"] = n_letters
    feats["n_vowels"] = n_vowels
    feats["n_consonants"] = n_cons
    feats["vowel_ratio"] = _ratio(n_vowels, n_letters)
    feats["consonant_ratio"] = _ratio(n_cons, n_letters)
    feats["unique_letter_ratio"] = _ratio(len(set(letters)), n_letters)
    feats["longest_vowel_run"] = _longest_run(letters, want_vowel=True)
    feats["longest_consonant_run"] = _longest_run(letters, want_vowel=False)

    for v in "aeiou":
        feats[f"count_{v}"] = low.count(v)

    non_native = sum(low.count(c) for c in NON_NATIVE_LETTERS)
    feats["non_native_letter_count"] = non_native
    feats["has_non_native_letter"] = non_native > 0

    n_upper = sum(1 for c in raw if c.isupper())
    feats["first_is_upper"] = raw[:1].isupper()
    feats["all_upper"] = raw.isupper() and any(c.isalpha() for c in raw)
    feats["upper_ratio"] = _ratio(n_upper, n_letters)
    feats["is_titlecase"] = raw.istitle()

    feats["has_digit"] = any(c.isdigit() for c in raw)
    feats["is_numeric"] = raw.replace(",", "").replace(".", "").isdigit()
    feats["is_punct"] = all(not c.isalnum() for c in raw) and n_chars > 0
    feats["has_hyphen"] = "-" in raw
    feats["is_hashtag"] = raw.startswith("#")
    feats["has_repeat_char"] = any(
        low[i] == low[i + 1] == low[i + 2]
        for i in range(len(low) - 2)
    )

    matched_prefix = _longest_fil_prefix(low, MIN_STEM_LEN if clean_affixes else 0)
    feats["fil_prefix"] = matched_prefix is not None
    feats["has_hyphen_affix"] = "-" in low and any(
        low.split("-")[0] == p for p in FIL_PREFIXES
    )
    feats["eng_suffix"] = any(low.endswith(s) for s in ENG_SUFFIXES)
    feats["fil_suffix"] = any(low.endswith(s) for s in FIL_SUFFIXES)

    if clean_affixes:
        feats["fil_prefix_which"] = matched_prefix or ""
        feats["eng_suffix_which"] = next(
            (s for s in ENG_SUFFIXES_BY_LEN if low.endswith(s)), "")
        feats["fil_suffix_which"] = next(
            (s for s in FIL_SUFFIXES_BY_LEN if low.endswith(s)), "")

    if cs_cue:
        hyphen_stem = low.split("-", 1)[1] if "-" in low else ""
        hyphen_head = low.split("-", 1)[0] if "-" in low else ""
        prefix_stem = low[len(matched_prefix):] if matched_prefix else ""

        feats["cs_hyphen_fil_eng"] = (
            hyphen_head in FIL_PREFIXES and _looks_english(hyphen_stem))
        feats["cs_prefix_eng_stem"] = bool(matched_prefix) and _looks_english(prefix_stem)
        feats["cs_any"] = feats["cs_hyphen_fil_eng"] or feats["cs_prefix_eng_stem"]

    if affix_strings:
        for k in (1, 2, 3, 4):
            feats[f"pref{k}"] = low[:k] if n_chars >= k else ""
            feats[f"suff{k}"] = low[-k:] if n_chars >= k else ""
    feats["has_eng_cluster"] = any(cl in low for cl in ENG_CLUSTERS)
    feats["has_reduplication"] = _has_reduplication(low)
    feats["ends_in_vowel"] = bool(letters) and letters[-1] in VOWELS
    feats["ends_in_ng"] = low.endswith("ng")
    feats["ends_in_y"] = low.endswith("y")

    ctx = context or Context([token], 0)
    prev_tok = ctx.prev()
    next_tok = ctx.next()
    feats["is_first_in_sentence"] = prev_tok is None
    feats["is_last_in_sentence"] = next_tok is None
    feats["prev_exists"] = prev_tok is not None
    feats["next_exists"] = next_tok is not None
    if prev_tok is not None:
        pl_letters = [c for c in prev_tok.lower() if c.isalpha()]
        feats["prev_ends_in_vowel"] = bool(pl_letters) and pl_letters[-1] in VOWELS
        feats["prev_is_cap"] = prev_tok[:1].isupper()
        feats["prev_len"] = len(prev_tok)
    if next_tok is not None:
        feats["next_is_cap"] = next_tok[:1].isupper()
        feats["next_len"] = len(next_tok)

    if neighbour_cues:
        for label, neighbour in (("prev", prev_tok), ("next", next_tok)):
            if neighbour is None:
                continue
            for cue_name, cue_val in _cue_summary(neighbour).items():
                feats[f"{label}_cue_{cue_name}"] = cue_val

    for n in ngram_sizes:
        for g in char_ngrams(low, n):
            feats[f"{n}gram={g}"] = 1

    return feats


def features_for_sentence(tokens: List[str],
                          ngram_sizes=(2, 3),
                          **flags: bool) -> List[Dict[str, Any]]:
    return [extract_features(tok, Context(tokens, i), ngram_sizes, **flags)
            for i, tok in enumerate(tokens)]


def crf_sequence(tokens: List[str],
                 ngram_sizes=(2, 3),
                 **flags: bool) -> List[Dict[str, Any]]:
    return [
        {k: (v if isinstance(v, (str, bool)) else float(v)) for k, v in d.items()}
        for d in features_for_sentence(tokens, ngram_sizes, **flags)
    ]


if __name__ == "__main__":
    import json

    samples = [
        ["Love", "kita", "."],
        ["Madami", "ang", "nag-march", "sa", "EDSA", "."],
        ["naglunch", "kami", "sa", "DLSU"],
        ["Masarap", "ang", "halo-halo", "merienda"],
    ]
    for sent in samples:
        print("=" * 64)
        print("Sentence:", sent)
        for tok, f in zip(sent, features_for_sentence(sent)):
            plain = {k: v for k, v in f.items() if "gram=" not in k}
            n_ngrams = sum(1 for k in f if "gram=" in k)
            print(f"\n  token={tok!r}  (+{n_ngrams} ngram flags)")
            print("   ", json.dumps(plain, ensure_ascii=False))
