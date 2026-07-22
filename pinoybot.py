import os
import pickle
import cloudpickle
from pathlib import Path
from typing import List
from features import features_for_sentence, Sparse32Caster

_HERE = Path(__file__).resolve().parent

POSSIBLE_MODEL_PATHS = [
    _HERE / "model_output" / "pinoybot_language-model.pkl",
    _HERE / "pinoybot_language-model.pkl",
    Path("model_output/pinoybot_language-model.pkl"),
    Path("pinoybot_language-model.pkl"),
]

VALID_TAGS = {"ENG", "FIL", "CS", "OTH"}
FALLBACK_TAG = "OTH"
_pipeline = None

def _load_pipeline():
    global _pipeline
    if _pipeline is not None:
        return

    model_path = next((p for p in POSSIBLE_MODEL_PATHS if p.exists()), None)
    if model_path is None:
        raise FileNotFoundError("Model file 'pinoybot_language-model.pkl' not found.")

    with open(model_path, "rb") as f:
        _pipeline = cloudpickle.load(f)
        
        
def tag_language(tokens: List[str]) -> List[str]:    
    if not tokens: 
        return []
    
    _load_pipeline()

    feature_dicts = features_for_sentence(list(tokens))
    
    predicted = _pipeline.predict(feature_dicts)

    tags = []
    for tag in predicted:
        tag = str(tag).strip().upper()
        tags.append(tag if tag in VALID_TAGS else FALLBACK_TAG)
        
        
    if len(tags) != len(tokens):
        raise RuntimeError(
            f"tag_language produced {len(tags)} tags for {len(tokens)} "
            "input tokens -- lengths must match."
        )

    return tags

if __name__ == "__main__":
    example_tokens = ["Love", "kita", "."]
    print("Tokens:", example_tokens)
    tags = tag_language(example_tokens)
    print("Tags:", tags)