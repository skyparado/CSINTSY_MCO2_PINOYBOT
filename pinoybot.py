"""
pinoybot.py

PinoyBot: Filipino Code-Switched Language Identifier

This module provides the main tagging function for the PinoyBot project, which identifies the language of each word in a code-switched Filipino-English text. The function is designed to be called with a list of tokens and returns a list of tags ("ENG", "FIL", "CS", or "OTH").

Model training and feature extraction should be implemented in a separate script. The trained model should be saved and loaded here for prediction.
"""

import os
import pickle
from pathlib import Path
from typing import List
from features import features_for_sentence

_HERE = Path(__file__).resolve().parent
MODEL_DIR = _HERE / "model_output"
MODEL_PATH = MODEL_DIR / "pinoybot_language-model.pkl"
VALID_TAGS = {"ENG", "FIL", "CS", "OTH"}
FALLBACK_TAG = "OTH"
_pipeline = None

def _load_pipeline():
    """Load the trained Pipeline (vectorizer + classifier) from disk, once.
 
    Raises:
        FileNotFoundError: with a clear message if the model file is missing,
            so the failure is obvious instead of a cryptic pickle error.
    """
    global _pipeline
 
    if _pipeline is not None:
        return
 
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Could not find trained model at {MODEL_PATH}. "
            "Make sure model_output/pinoybot_language-model.pkl is present "
            "next to pinoybot.py (run model.py's train_model() first)."
        )
 
    with open(MODEL_PATH, "rb") as f:
        _pipeline = pickle.load(f)

# Main tagging function
def tag_language(tokens: List[str]) -> List[str]:
    """
    Tags each token in the input list with its predicted language.
    Args:
        tokens: List of word tokens (strings).
    Returns:
        tags: List of predicted tags ("ENG", "FIL", "CS", or "OTH"), one per token.
    """
    # Edge case: empty input -> empty output. Nothing to predict, and this
    # avoids calling the pipeline on an empty feature list.
    if not tokens:
        return []
    
    # 1. Load your trained model from disk (e.g., using pickle or joblib)
    #    Example: with open('trained_model.pkl', 'rb') as f: model = pickle.load(f)
    #    (Replace with your actual model loading code)
    _load_pipeline()

    # 2. Extract features from the input tokens to create the feature matrix
    #    Example: features = ... (your feature extraction logic here)
    feature_dicts = features_for_sentence(list(tokens))
    
    
    # 3. Use the model to predict the tags for each token
    #    Example: predicted = model.predict(features)
    predicted = _pipeline.predict(feature_dicts)

    # 4. Convert the predictions to a list of strings ("ENG", "FIL", or "OTH")
    #    Example: tags = [str(tag) for tag in predicted]
    tags = []
    for tag in predicted:
        tag = str(tag).strip().upper()
        tags.append(tag if tag in VALID_TAGS else FALLBACK_TAG)
        
    # check before returning: length must match, since the spec
    # warns this can break the automated checking process.
    if len(tags) != len(tokens):
        raise RuntimeError(
            f"tag_language produced {len(tags)} tags for {len(tokens)} "
            "input tokens -- lengths must match."
        )

    # 5. Return the list of tags
    return tags


if __name__ == "__main__":
    example_tokens = ["Love", "kita", "."]
    print("Tokens:", example_tokens)
    tags = tag_language(example_tokens)
    print("Tags:", tags)
    print("Tags:", tags)
    
    # Demo Examples
    #demo_examples = [
     #   ["I", "love", "you"],         
     #  ["Mahal", "kita", "."],        
     #   ["Ang", "cute", "mo"],          
     #   ["Kain", "tayo", "later"],     
     #   ["Sige", "na", "please"],      
     #   ["Grabe", "ang", "ganda"],     
     #  ["See", "you", "bukas"],        
     #   ["Wala", "akong", "pera"],      
     #   ["Text", "mo", "ko"],           
    #]

    #for i, tokens in enumerate(demo_examples, start=1):
    #    tags = tag_language(tokens)
    #    print(f"\nDemo {i}:")
    #    print("Tokens:", tokens)
    #    print("Tags:  ", tags)