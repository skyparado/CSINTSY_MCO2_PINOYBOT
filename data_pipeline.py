from pathlib import Path                          
from dataclasses import dataclass, field           
from typing import Any, Dict, List, Optional, Tuple  
import json                                        
import pickle                                      
import cloudpickle
from collections import Counter                    

import openpyxl                                    

from features import features_for_sentence         # Sky's feature-extraction function 
from sklearn.model_selection import GroupShuffleSplit 

RANDOM_SEED = 42       
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
    label: Optional[str]  


def load_raw_rows(path: Path = DATASET_PATH, sheet: str = SHEET_NAME) -> List[RawRow]:
    """Read the annotation sheet into a flat list of RawRow, one per word."""
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb[sheet]   # grab the specific "FINAL ANNOTATION" tab out of the whole workbook

    rows: List[RawRow] = []   # this will collect every row as we read them
    for r in ws.iter_rows(min_row=2, values_only=True):

        word_id, sentence_id, sentence, word, answer = r[0], r[1], r[2], r[3], r[4]

        if word_id is None:
            continue
    

        rows.append(RawRow(
            word_id=int(word_id),        
            sentence_id=int(sentence_id),
            sentence=sentence,
            word=word,
            label=answer,
        ))
    return rows   # hand back the full list of RawRow objects


# 2: validate labels make sure the dataset is already clean makes sure its consistent.

def validate_labels(rows: List[RawRow]) -> None:
    for row in rows:                      
        assert row.label in VALID_LABELS, (
            f"Bad label {row.label!r} on word {row.word!r} "
            f"(word_id={row.word_id}). Expected one of {VALID_LABELS}."
        ) 

# 3: reconstruct sentences from the flat word list
@dataclass
#Sorts all rows into sentences, so that words line up in order
class Sentence:
    sentence_id: int
    tokens: List[str] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)


#Loops through each row, grabs the sentence for its sentence id, then appends the word to tokens and the labels.
def group_into_sentences(rows: List[RawRow]) -> List[Sentence]:
    by_sentence: Dict[int, Sentence] = {}

    for row in sorted(rows, key=lambda r: (r.sentence_id, r.word_id)):

        sent = by_sentence.setdefault(row.sentence_id, Sentence(row.sentence_id))
        sent.tokens.append(row.word)     
        sent.labels.append(row.label)    

    # Return built sentences, sorted by sentence_id
    return [by_sentence[sid] for sid in sorted(by_sentence)]


# 4 run features.py over every sentence
def build_feature_dataset(
    sentences: List[Sentence],
) -> Tuple[List[Dict[str, Any]], List[str], List[int]]:
    X: List[Dict[str, Any]] = []   
    y: List[str] = []              
    groups: List[int] = []         

    for sent in sentences:                              
        feats = features_for_sentence(sent.tokens)       # Sky's function: turns this sentence's words into feature dicts
        X.extend(feats) # Clues for each word

        y.extend(sent.labels)      # Correct label for each word                     

        groups.extend([sent.sentence_id] * len(sent.tokens)) # which sentence it came from

    return X, y, groups

# Applys skys features, and groups into 3 lists



# 5: takes groups from part 4 split train/val/test BY SENTENCE (not by individual word)
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

    # A. carve off test first (15%)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)

    trainval_idx, test_idx = next(gss1.split(X, y, groups=groups))
        
    # B.split the remaining 85% into train and val
    remaining_groups = [groups[i] for i in trainval_idx]

    relative_val_frac = val_frac / (train_frac + val_frac)

    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_val_frac, random_state=seed)
    train_rel_idx, val_rel_idx = next(gss2.split(
        trainval_idx, [y[i] for i in trainval_idx], groups=remaining_groups
    ))

    train_idx = [trainval_idx[i] for i in train_rel_idx]
    val_idx = [trainval_idx[i] for i in val_rel_idx]

    # Helper function to know which sentence end up in each split.   (no sentence should be in more than one split)
    def sentence_ids(idx):
        return set(groups[i] for i in idx)

    assert not (sentence_ids(train_idx) & sentence_ids(val_idx))
    assert not (sentence_ids(train_idx) & sentence_ids(test_idx))
    assert not (sentence_ids(val_idx) & sentence_ids(test_idx))

    def subset(idx):
        return {"X": [X[i] for i in idx],
                "y": [y[i] for i in idx],
                "groups": [groups[i] for i in idx]}

    return {"train": subset(train_idx), "val": subset(val_idx), "test": subset(test_idx)}

# 6: save train/val/test to disk

def save_outputs(splits: Dict[str, Dict[str, list]], out_dir: Path = OUTPUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, data in splits.items():

        with open(out_dir / f"{name}.pkl", "wb") as f:
            cloudpickle.dump(data, f)

    summary = {
        "random_seed": RANDOM_SEED,
        "fractions": {"train": TRAIN_FRAC, "val": VAL_FRAC, "test": TEST_FRAC},
    }
    for name, data in splits.items():
        summary[name] = {
            "n_words": len(data["y"]),                       
            "label_distribution": dict(Counter(data["y"])),  
        }

    with open(out_dir / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def run_pipeline(path: Path = DATASET_PATH, out_dir: Path = OUTPUT_DIR) -> Dict[str, Dict[str, list]]:
    rows = load_raw_rows(path)          # Stage 1
    validate_labels(rows)               # Stage 2
    sentences = group_into_sentences(rows)   # Stage 3
    X, y, groups = build_feature_dataset(sentences)   # Stage 4
    splits = split_dataset(X, y, groups)     # Stage 5
    save_outputs(splits, out_dir)            # Stage 6
    return splits


if __name__ == "__main__":

    splits = run_pipeline()
    print("Split sizes:")
    for name, data in splits.items():
        print(f"  {name}: {len(data['y'])} words | labels: {dict(Counter(data['y']))}")
    print(f"\nSaved train.pkl / val.pkl / test.pkl / split_summary.json to {OUTPUT_DIR}")