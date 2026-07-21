"""
note: I asked AI to generate comments so the code can be understood easily, lmk if theres smth confusing you

features.py
===========

Feature engineering module for **PinoyBot**, a Filipino code-switched
language identifier. This module is responsible for ONE thing: turning a
word (token) into a set of numeric/boolean features that a machine-learning
model can learn from.

The public API is two functions:
    - extract_features(token, context) -> dict
    - features_for_sentence(tokens)    -> list[dict]
Everything else is a helper.
"""

from typing import List, Dict, Optional, Any

# Vowels used for vowel-ratio style features.
VOWELS = set("aeiou")

# Common Filipino/Tagalog PREFIX cues. NOTE: these are SURFACE PATTERN cues
# (does the word start with these letters?), NOT a dictionary lookup of whole
# words. The project brief forbids looking up whole words in a dictionary, but
# allows using language knowledge to design features -- which is what this is.
FIL_PREFIXES = ("nag", "mag", "pag", "nakaka", "naka", "pina", "ipina",
                "napaka", "pinaka", "um", "in", "ka", "ma", "na")

# Common word ENDINGS that lean toward one language. Same disclaimer as above:
# these are surface cues, not dictionary lookups.
ENG_SUFFIXES = ("tion", "sion", "ing", "ed", "ly", "ment", "ness", "ous",
                "ive", "ance", "ence", "ational")
FIL_SUFFIXES = ("han", "hin", "syon", "in", "an", "ng")

# Letters that are rare in native Filipino spelling; their presence leans ENG.
NON_NATIVE_LETTERS = "cfjqvxz"

# English-looking consonant clusters (leans ENG).
ENG_CLUSTERS = ("ck", "sch", "str", "th", "ght", "ph", "sh")

# Minimum number of letters that must remain AFTER stripping a candidate FIL
# prefix for the match to count. Without this, "ma"/"na"/"ka"/"in" fire on
# almost anything, including short English words.
MIN_STEM_LEN = 3

# Pre-sorted longest-first, so "which affix matched" can short-circuit on the
# most specific match. Computed once at import instead of once per token --
# these sorts were measurably hot when done inside extract_features.
ENG_SUFFIXES_BY_LEN = tuple(sorted(ENG_SUFFIXES, key=len, reverse=True))
FIL_SUFFIXES_BY_LEN = tuple(sorted(FIL_SUFFIXES, key=len, reverse=True))
FIL_PREFIXES_BY_LEN = tuple(sorted(FIL_PREFIXES, key=len, reverse=True))


class Context:
    """Position of a token inside its sentence, used for neighbour features.

    A Context bundles the full list of tokens in a sentence together with the
    index of the "current" token, so feature code can peek at the previous and
    next words. It is optional everywhere -- if you call ``extract_features``
    with no context, the token is treated as a one-word sentence.

    Args:
        tokens: The full list of tokens in the sentence.
        index:  The position of the current token within ``tokens``.
    """

    def __init__(self, tokens: Optional[List[str]] = None, index: int = 0):
        self.tokens = tokens if tokens is not None else []
        self.index = index

    def prev(self) -> Optional[str]:
        """Return the previous token, or None if this is the first token."""
        i = self.index - 1
        return self.tokens[i] if 0 <= i < len(self.tokens) else None

    def next(self) -> Optional[str]:
        """Return the next token, or None if this is the last token."""
        i = self.index + 1
        return self.tokens[i] if 0 <= i < len(self.tokens) else None


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def char_ngrams(text: str, n: int) -> List[str]:
    """Return character n-grams of length ``n`` for ``text``.

    The text is padded with ``^`` (start) and ``$`` (end) so the model can
    learn word-initial and word-final patterns. For example, the 2-grams of
    "sa" are ["^s", "sa", "a$"].

    Args:
        text: The (lowercased) word to break into n-grams.
        n:    The n-gram length (2 = pairs, 3 = triples, ...).

    Returns:
        A list of n-gram strings. Empty if ``text`` is empty.
    """
    if not text:
        return []
    padded = "^" + text + "$"
    return [padded[i:i + n] for i in range(len(padded) - n + 1)]


def _ratio(count: int, total: int) -> float:
    """Safe division: returns 0.0 instead of crashing when ``total`` is 0."""
    return count / total if total else 0.0


def _longest_run(letters: List[str], want_vowel: bool) -> int:
    """Length of the longest consecutive run of vowels (or consonants).

    Filipino words rarely stack many consonants in a row, so a long
    consonant run is a mild signal that a word may be English.

    Args:
        letters:    The word's alphabetic characters, lowercased.
        want_vowel: If True, count vowel runs; if False, consonant runs.

    Returns:
        The length of the longest matching run.
    """
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
    """Detect simple reduplication like "halo-halo" or "araw-araw".

    Reduplication (repeating a chunk) is a common Filipino morphological
    pattern, so it is a useful FIL signal. This checks two cases:
    a hyphen-separated repeat ("halo-halo"), and a whole word made of one
    chunk repeated twice ("bumbum").

    Args:
        low: The lowercased token.

    Returns:
        True if a reduplication pattern is found.
    """
    if "-" in low:
        parts = low.split("-")
        if len(parts) == 2 and parts[0] and parts[0] == parts[1]:
            return True
    n = len(low)
    if n >= 4 and n % 2 == 0 and low[:n // 2] == low[n // 2:]:
        return True
    return False


def _longest_fil_prefix(low: str, min_stem: int = 0) -> Optional[str]:
    """Return the LONGEST matching Filipino prefix, or None.

    Longest-match matters: "nagkaka-" should report "nagkaka"-style evidence
    rather than stopping at the weak "na". ``min_stem`` rejects a match that
    would leave too little word behind it, which is what stops "ma"/"na"/"ka"
    from firing on every short English word.

    Args:
        low:      The lowercased token.
        min_stem: Required number of letters remaining after the prefix.

    Returns:
        The matched prefix string, or None if nothing qualifies.
    """
    # Pre-sorted longest-first, so the first hit IS the longest match.
    for p in FIL_PREFIXES_BY_LEN:
        if low.startswith(p) and len(low) - len(p) >= min_stem:
            return p
    return None


def _looks_english(stem: str) -> bool:
    """Cheap test for 'this chunk of letters looks like an English root'.

    Used for the code-switch cue: a Filipino affix wrapped around something
    that looks English is the canonical CS pattern ("nag-march", "naglunch").
    """
    if not stem:
        return False
    return (any(c in NON_NATIVE_LETTERS for c in stem)
            or any(cl in stem for cl in ENG_CLUSTERS)
            or stem.endswith(ENG_SUFFIXES))


def _cue_summary(token: str) -> Dict[str, Any]:
    """Compact language cues for a NEIGHBOUR token.

    Deliberately small: these get copied onto the current token with a
    ``prev_``/``next_`` prefix, so anything expensive (like n-grams) would
    triple the feature space for little gain. Language clusters in
    code-switched text, so a neighbour's cues are strong evidence.

    Args:
        token: The neighbouring token (raw, not lowercased).

    Returns:
        A dict of small boolean/numeric cues.
    """
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


# ---------------------------------------------------------------------------
# Main feature extractor
# ---------------------------------------------------------------------------
def extract_features(token: str,
                     context: Optional[Context] = None,
                     ngram_sizes=(2, 3),
                     neighbour_cues: bool = False,
                     cs_cue: bool = True,
                     clean_affixes: bool = True,
                     affix_strings: bool = True) -> Dict[str, Any]:
    """Turn a single token into a dictionary of features.

    This is the main public function of the module. It groups features into
    several families (letter composition, capitalization, token shape,
    affix/morphology cues, neighbour context, and character n-grams). Each
    family is documented inline below.

    The four boolean flags exist so each feature family can be switched off
    independently and measured. The defaults are what 5-fold grouped CV
    actually supported, NOT everything-on:

        neighbour_cues=False  measured -0.021 macro F1 (the only effect
                              outside noise, and it was harmful -- the CRF's
                              transition model does this job properly)
        cs_cue=True           within noise, kept: it is the one signal aimed
                              at the weakest class
        clean_affixes=True    within noise on the metric, kept on correctness
                              grounds ("ma"/"na"/"in" were firing on machine,
                              nation, insane)
        affix_strings=True    within noise, kept: cheap and helps the CRF

    ngram_sizes stops at 3 -- adding 4-grams cost -0.03 macro F1 and doubled
    the feature count.

    Args:
        token:          The word to featurize.
        context:        Optional Context giving the surrounding tokens. If None,
                        the token is treated as a one-word sentence.
        ngram_sizes:    Which character n-gram lengths to include.
        neighbour_cues: Include the previous/next token's own language cues.
        cs_cue:         Include the "Filipino affix + English root" CS signal.
        clean_affixes:  Use longest-match prefixes with a minimum stem length,
                        and emit WHICH affix matched rather than one lumped
                        boolean.
        affix_strings:  Include the token's own first/last 1-4 characters as
                        categorical features.

    Returns:
        A dict mapping feature name to value (numbers, booleans, strings, or
        n-gram presence flags like ``"2gram=na": 1``).
    """
    feats: Dict[str, Any] = {}
    raw = token
    low = token.lower()
    n_chars = len(low)

    # --- Letter / vowel composition -------------------------------------
    # How the word is built from letters and vowels. Filipino words tend to
    # be more vowel-balanced; heavy consonant use leans English.
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

    # Count of each vowel (a/o are common in Filipino).
    for v in "aeiou":
        feats[f"count_{v}"] = low.count(v)

    # Letters rare in native Filipino spelling (c, f, j, q, v, x, z) -> ENG cue.
    non_native = sum(low.count(c) for c in NON_NATIVE_LETTERS)
    feats["non_native_letter_count"] = non_native
    feats["has_non_native_letter"] = non_native > 0

    # --- Capitalization --------------------------------------------------
    # Casing helps flag names and abbreviations (which are OTH). ALL-CAPS is a
    # strong abbreviation cue (e.g. "DLSU", "EDSA").
    n_upper = sum(1 for c in raw if c.isupper())
    feats["first_is_upper"] = raw[:1].isupper()
    feats["all_upper"] = raw.isupper() and any(c.isalpha() for c in raw)
    feats["upper_ratio"] = _ratio(n_upper, n_letters)
    feats["is_titlecase"] = raw.istitle()

    # --- Token shape / OTH cues -----------------------------------------
    # Signals that a token is punctuation, a number, a hashtag, or an
    # onomatopoeia -- all of which are tagged OTH.
    feats["has_digit"] = any(c.isdigit() for c in raw)
    feats["is_numeric"] = raw.replace(",", "").replace(".", "").isdigit()
    feats["is_punct"] = all(not c.isalnum() for c in raw) and n_chars > 0
    feats["has_hyphen"] = "-" in raw
    feats["is_hashtag"] = raw.startswith("#")
    feats["has_repeat_char"] = any(
        low[i] == low[i + 1] == low[i + 2]
        for i in range(len(low) - 2)
    )  # 3+ identical letters in a row, e.g. "grrr"

    # --- Affix / morphology surface cues --------------------------------
    # Filipino builds words with affixes and reduplication. These are the most
    # useful cues for spotting FIL words and intra-word code-switches (CS),
    # e.g. "nag-march" = Filipino prefix + English root.
    # ``clean_affixes`` swaps the naive "does it start with any prefix?" test
    # for longest-match + a minimum stem length, and additionally records WHICH
    # affix matched. The lumped boolean treats "nag" (strong FIL evidence) and
    # "ma" (fires on "machine", "market", "manage") as the same signal; letting
    # the model see the actual prefix lets it weight them differently.
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
        # Code-switching at the word level is Filipino morphology wrapped around
        # an English root: "nag-march", "naglunch", "mag-enjoy". Detect it by
        # stripping the FIL affix and asking whether what remains looks English.
        # The hyphenated and glued-together spellings both need covering.
        hyphen_stem = low.split("-", 1)[1] if "-" in low else ""
        hyphen_head = low.split("-", 1)[0] if "-" in low else ""
        prefix_stem = low[len(matched_prefix):] if matched_prefix else ""

        feats["cs_hyphen_fil_eng"] = (
            hyphen_head in FIL_PREFIXES and _looks_english(hyphen_stem))
        feats["cs_prefix_eng_stem"] = bool(matched_prefix) and _looks_english(prefix_stem)
        feats["cs_any"] = feats["cs_hyphen_fil_eng"] or feats["cs_prefix_eng_stem"]

    if affix_strings:
        # The token's own edges as categorical values. Word-final characters
        # carry most of the morphological signal in both languages, so these
        # are usually the single most useful features in word-level LID.
        for k in (1, 2, 3, 4):
            feats[f"pref{k}"] = low[:k] if n_chars >= k else ""
            feats[f"suff{k}"] = low[-k:] if n_chars >= k else ""
    feats["has_eng_cluster"] = any(cl in low for cl in ENG_CLUSTERS)
    feats["has_reduplication"] = _has_reduplication(low)
    feats["ends_in_vowel"] = bool(letters) and letters[-1] in VOWELS
    feats["ends_in_ng"] = low.endswith("ng")
    feats["ends_in_y"] = low.endswith("y")

    # --- Context / neighbour features -----------------------------------
    # Properties of the surrounding words. Language tends to cluster (a
    # Filipino word is often next to other Filipino words), so neighbours help.
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
        # The plain prev_/next_ features above only describe a neighbour's
        # LENGTH and CASING -- they say nothing about what language it is. Since
        # language clusters (a Filipino word is usually surrounded by Filipino
        # words), copying the neighbour's actual language cues onto this token
        # is far more informative than knowing it was 5 characters long.
        for label, neighbour in (("prev", prev_tok), ("next", next_tok)):
            if neighbour is None:
                continue
            for cue_name, cue_val in _cue_summary(neighbour).items():
                feats[f"{label}_cue_{cue_name}"] = cue_val

    # --- Character n-grams (presence flags) -----------------------------
    # Chunks of letters that lean one language (e.g. "tion" -> ENG, "ng" ->
    # FIL). Encoded as keys like "2gram=na": 1 so the DictVectorizer can turn
    # them into numeric columns later.
    for n in ngram_sizes:
        for g in char_ngrams(low, n):
            feats[f"{n}gram={g}"] = 1

    return feats


def features_for_sentence(tokens: List[str],
                          ngram_sizes=(2, 3),
                          **flags: bool) -> List[Dict[str, Any]]:
    """Extract features for every token in a sentence, with context wired up.

    This is the convenient way to featurize real data: it builds the Context
    for each token automatically so neighbour features work.

    Args:
        tokens:      The list of tokens in one sentence.
        ngram_sizes: Which character n-gram lengths to include.
        **flags:     Forwarded to ``extract_features`` (neighbour_cues, cs_cue,
                     clean_affixes, affix_strings). Used by the ablation script
                     to turn one family off at a time.

    Returns:
        A list of feature dicts, one per token, in the same order as ``tokens``.
    """
    return [extract_features(tok, Context(tokens, i), ngram_sizes, **flags)
            for i, tok in enumerate(tokens)]


def crf_sequence(tokens: List[str],
                 ngram_sizes=(2, 3),
                 **flags: bool) -> List[Dict[str, Any]]:
    """Featurize a sentence in the shape python-crfsuite expects.

    Identical to ``features_for_sentence`` except that every non-string,
    non-boolean value is cast to float. crfsuite accepts str/bool/float
    attribute values; numpy scalars and ints coming out of the feature code
    would otherwise raise a confusing type error deep inside the C extension.

    Args:
        tokens:      The list of tokens in one sentence.
        ngram_sizes: Which character n-gram lengths to include.
        **flags:     Forwarded to ``extract_features``.

    Returns:
        A list of crfsuite-safe feature dicts, one per token.
    """
    return [
        {k: (v if isinstance(v, (str, bool)) else float(v)) for k, v in d.items()}
        for d in features_for_sentence(tokens, ngram_sizes, **flags)
    ]


if __name__ == "__main__":
    # Manual sanity check. Run:  python features.py
    # Prints the non-ngram features for a few tricky tokens so you can eyeball
    # that the values make sense before plugging this into the pipeline.
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
