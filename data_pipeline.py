from pathlib import Path                          # Path lets us build file paths that work on Windows AND Mac/Linux
from dataclasses import dataclass, field           # dataclass = shortcut for making simple "data holder" classes
from typing import Any, Dict, List, Optional, Tuple  # type hints only -- documentation for humans, not enforced by Python
import json                                        # to write split_summary.json
import pickle                                      # to save train/val/test as .pkl files
from collections import Counter                    # quick way to count how many FIL/ENG/CS/OTH labels are in a list

import openpyxl                                    # library for reading .xlsx Excel files

from features import features_for_sentence         # Sky's feature-extraction function 
from sklearn.model_selection import GroupShuffleSplit  # does our train/val/test split, keeping whole sentences together

RANDOM_SEED = 42        # fixes the "randomness" so the split comes out identical every time this script runs
TRAIN_FRAC = 0.70        # 70% of sentences go to training
VAL_FRAC = 0.15          # 15% go to validation
TEST_FRAC = 0.15         # 15% go to testing 

DATASET_PATH = Path(__file__).parent / "[shared] dataset_annotation.xlsx" # Dataset 


SHEET_NAME = "FINAL ANNOTATION"          # the exact tab name inside the Excel file we need to read
OUTPUT_DIR = Path(__file__).parent / "pipeline_output"   # folder where train/val/test.pkl will be saved

VALID_LABELS = {"FIL", "ENG", "CS", "OTH"}   # the only 4 labels allowed to exist in the dataset

# 1. Load the raw spreadsheet into a flat list of rows, one per word
@dataclass
class RawRow:
    # This is a "blueprint" for one row of the spreadsheet. Makes the code easier to read than using a dictionary
    word_id: int
    sentence_id: int
    sentence: str
    word: str
    label: Optional[str]  # Optional because a few rows could have a blank label (None)


def load_raw_rows(path: Path = DATASET_PATH, sheet: str = SHEET_NAME) -> List[RawRow]:
    """Read the annotation sheet into a flat list of RawRow, one per word."""
    wb = openpyxl.load_workbook(path, data_only=True)
    # data_only=True: if any Excel cell contains a FORMULA, give us the calculated
    # result instead of the formula text itself. Doesn't matter much here since our
    # sheet is just plain text/numbers, but it's a safe default.

    ws = wb[sheet]   # grab the specific "FINAL ANNOTATION" tab out of the whole workbook

    rows: List[RawRow] = []   # this will collect every row as we read them
    for r in ws.iter_rows(min_row=2, values_only=True):
        # ws.iter_rows(...) walks through the sheet row by row.
        # min_row=2 skips row 1, which is just the header text (word_id, sentence_id, ...).
        # values_only=True gives us plain values like (1, 1, "Love kita.", "Love", "ENG")

        word_id, sentence_id, sentence, word, answer = r[0], r[1], r[2], r[3], r[4]
        # unpack the 5 columns into 5 separate variables, by position (0-4).

        if word_id is None:
            continue
            # Guards against a fully blank row (e.g. an empty row Excel sometimes
            # leaves at the very end of a sheet). Without this, int(None) below
            # would crash the whole script.

        rows.append(RawRow(
            word_id=int(word_id),        # openpyxl sometimes gives numbers as 1.0 instead of 1; force it to a clean int
            sentence_id=int(sentence_id),
            sentence=sentence,
            word=word,
            label=answer,
        ))
    return rows   # hand back the full list of RawRow objects, one per word


# 2: validate labels make sure the dataset is already clean makes sure its consistent.

def validate_labels(rows: List[RawRow]) -> None:
    for row in rows:                       # check every single row, one at a time
        assert row.label in VALID_LABELS, (
            # assert = "if this condition is False, STOP the program right here
            # and show the message below." It's a tripwire, not a fixer.
            f"Bad label {row.label!r} on word {row.word!r} "
            f"(word_id={row.word_id}). Expected one of {VALID_LABELS}."
            # !r shows the value with quotes around it, e.g. 'FIL ' -- this makes
            # it obvious if there's a sneaky trailing space, which plain printing would hide. engng
        )



# 3: reconstruct sentences from the flat word list

@dataclass
class Sentence:
    # A blueprint for "one whole sentence", built up from many RawRow pieces.
    sentence_id: int
    tokens: List[str] = field(default_factory=list)
    # default_factory=list: every new Sentence starts with its OWN empty list.
    # (Without this trick, dataclasses can accidentally make every Sentence
    # object secretly share the SAME list, which causes bizarre bugs.)
    labels: List[str] = field(default_factory=list)


def group_into_sentences(rows: List[RawRow]) -> List[Sentence]:
    by_sentence: Dict[int, Sentence] = {}
    # a lookup table: sentence_id -> the Sentence object being built for it

    for row in sorted(rows, key=lambda r: (r.sentence_id, r.word_id)):
        # sorted(..., key=...) reorders all rows first by sentence_id,
        # then (for rows in the same sentence) by word_id -- guaranteeing
        # words come out in the correct reading order even if the sheet
        # rows were scrambled.

        sent = by_sentence.setdefault(row.sentence_id, Sentence(row.sentence_id))
        # setdefault: "if this sentence_id already has a Sentence object, give
        # me that one. If not, create a new empty Sentence and store it first."
        # Saves us writing an explicit if/else check every loop iteration.

        sent.tokens.append(row.word)     # add this word to its sentence's word list
        sent.labels.append(row.label)    # add this word's label too, same position

    return [by_sentence[sid] for sid in sorted(by_sentence)]
    # by_sentence.keys() (the sentence_ids) could come out in any order from
    # a dict, so we sort them before returning -- keeps output identical every run.


# 4 run features.py over every sentence

def build_feature_dataset(
    sentences: List[Sentence],
) -> Tuple[List[Dict[str, Any]], List[str], List[int]]:
    X: List[Dict[str, Any]] = []   # will hold one feature-dict per word, across ALL sentences
    y: List[str] = []              # will hold the matching label for each word in X
    groups: List[int] = []         # will hold which sentence_id each word in X came from

    for sent in sentences:                              # go sentence by sentence
        feats = features_for_sentence(sent.tokens)       # Sky's function: turns this sentence's words into feature dicts
        X.extend(feats)
        # .extend (not .append!) because feats is ALREADY a list (one dict per
        # word in this sentence) -- extend adds each dict individually to X,
        # instead of nesting a whole list inside X.

        y.extend(sent.labels)                            # add this sentence's labels, same length as feats

        groups.extend([sent.sentence_id] * len(sent.tokens))
        # [5] * 3 makes [5, 5, 5]. So if this sentence has 3 words, we record
        # its sentence_id 3 times -- once per word -- so groups[i] always tells
        # us "which sentence did X[i] come from?"

    return X, y, groups



# 5: split train/val/test BY SENTENCE (not by individual word)
def split_dataset(
    X: List[Dict[str, Any]],
    y: List[str],
    groups: List[int],
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    test_frac: float = TEST_FRAC,
    seed: int = RANDOM_SEED,
) -> Dict[str, Dict[str, list]]:
    assert abs((train_frac + val_frac + test_frac) - 1.0) < 1e-9
    # sanity check: the 3 fractions must add up to 1.0 (100%), or something's
    # misconfigured. The "< 1e-9" instead of "== 1.0" is just to avoid tiny
    # computer rounding errors (like 0.1 + 0.2 != 0.3 exactly) falsely failing.

    # --- Step A: pull out the test set first ---
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    # n_splits=1: we only want ONE random split (not multiple attempts)
    # test_size=test_frac: put 15% of the SENTENCES into the "test" side
    # random_state=seed: use our fixed seed, so this shuffle is reproducible

    trainval_idx, test_idx = next(gss1.split(X, y, groups=groups))
    # .split(...) is a generator; next(...) grabs its one result.
    # groups=groups is the key part -- it tells sklearn "keep everything with
    # the same sentence_id together, never split a sentence across the divide."
    # Returns two lists of INDEX NUMBERS (not the actual data yet).

    # --- Step B: split what's left (85%) into train and val ---
    remaining_groups = [groups[i] for i in trainval_idx]
    # grab just the sentence_ids belonging to the 85% we haven't split yet

    relative_val_frac = val_frac / (train_frac + val_frac)
    # val_frac (0.15) is a fraction of the ORIGINAL 100%, but we're now splitting
    # only the remaining 85%. So we re-express "15% of everything" as
    # "X% of what's left": 0.15 / 0.85 ≈ 0.176 (about 17.6% of the remainder).

    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_val_frac, random_state=seed)
    train_rel_idx, val_rel_idx = next(gss2.split(
        trainval_idx, [y[i] for i in trainval_idx], groups=remaining_groups
    ))
    # same idea as Step A, just applied to the remaining 85% chunk.
    # "_rel_idx" = these are positions WITHIN trainval_idx, not the original X.

    train_idx = [trainval_idx[i] for i in train_rel_idx]
    val_idx = [trainval_idx[i] for i in val_rel_idx]
    # convert those "relative" positions back into real positions in the
    # original X/y/groups lists, so we can actually use them.

    # --- Safety check: make sure no sentence leaked across splits ---
    def sentence_ids(idx):
        return set(groups[i] for i in idx)
        # collect the unique sentence_ids present in this list of indices

    assert not (sentence_ids(train_idx) & sentence_ids(val_idx))
    assert not (sentence_ids(train_idx) & sentence_ids(test_idx))
    assert not (sentence_ids(val_idx) & sentence_ids(test_idx))
    # "&" between two sets gives whatever they have IN COMMON.
    # We assert that overlap is EMPTY between every pair of splits --
    # if any sentence_id appears in two splits, this crashes immediately.

    def subset(idx):
        return {"X": [X[i] for i in idx], "y": [y[i] for i in idx]}
        # given a list of index numbers, pull out just those X's and y's

    return {"train": subset(train_idx), "val": subset(val_idx), "test": subset(test_idx)}
    # final result: a dict of 3 buckets, each holding its own X and y



# 6: save train/val/test to disk

def save_outputs(splits: Dict[str, Dict[str, list]], out_dir: Path = OUTPUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # create the pipeline_output folder if it doesn't exist yet.
    # parents=True: also create any missing parent folders in the path.
    # exist_ok=True: don't crash if the folder is already there.

    for name, data in splits.items():
        # name will be "train", "val", "test" one at a time
        # data will be that split's {"X": [...], "y": [...]} dict

        with open(out_dir / f"{name}.pkl", "wb") as f:
            # "wb" = write, in binary mode (pickle files aren't plain text)
            pickle.dump(data, f)
            # save the entire "data" dict to this file, exactly as it is in memory

    summary = {
        "random_seed": RANDOM_SEED,
        "fractions": {"train": TRAIN_FRAC, "val": VAL_FRAC, "test": TEST_FRAC},
    }
    for name, data in splits.items():
        summary[name] = {
            "n_words": len(data["y"]),                       # how many words ended up in this split
            "label_distribution": dict(Counter(data["y"])),  # how many FIL/ENG/CS/OTH in this split
        }

    with open(out_dir / "split_summary.json", "w", encoding="utf-8") as f:
        # "w" = write, in text mode this time (JSON is human-readable text)
        json.dump(summary, f, indent=2, ensure_ascii=False)
        # indent=2: pretty-print with 2-space indentation, so it's easy to read
        # ensure_ascii=False: allow non-English characters to print normally
        # instead of being escaped into unreadable \uXXXX codes


# ---------------------------------------------------------------------------
# Orchestration: run all six stages in order
# ---------------------------------------------------------------------------
def run_pipeline(path: Path = DATASET_PATH, out_dir: Path = OUTPUT_DIR) -> Dict[str, Dict[str, list]]:
    rows = load_raw_rows(path)          # Stage 1
    validate_labels(rows)               # Stage 2
    sentences = group_into_sentences(rows)   # Stage 3
    X, y, groups = build_feature_dataset(sentences)   # Stage 4
    splits = split_dataset(X, y, groups)     # Stage 5
    save_outputs(splits, out_dir)            # Stage 6
    return splits
    # This one function is the "front door" -- calling it runs the whole
    # pipeline in the correct order, start to finish.


if __name__ == "__main__":
    # This block only runs when you execute "python data_pipeline.py" directly --
    # it will NOT run if someone else imports this file (e.g. "from data_pipeline
    # import run_pipeline" from another script). That's what makes this file
    # both a runnable script AND a reusable module.

    splits = run_pipeline()
    print("Split sizes:")
    for name, data in splits.items():
        print(f"  {name}: {len(data['y'])} words | labels: {dict(Counter(data['y']))}")
    print(f"\nSaved train.pkl / val.pkl / test.pkl / split_summary.json to {OUTPUT_DIR}")