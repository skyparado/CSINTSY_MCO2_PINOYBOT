import os
import cloudpickle
import pandas as pd
import numpy as np
import scipy.sparse as sp

from pathlib import Path
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from features import Sparse32Caster

PIPELINE_PATH = Path(__file__).parent / "pipeline_output"
MODEL_DIR = Path(__file__).parent / "model_output"
MODEL_PATH = MODEL_DIR / "pinoybot_language-model.pkl"
    
    
def load_pipeline(filename: str): 
    
    file_path = PIPELINE_PATH / filename 
    if not file_path.exists():
        raise FileNotFoundError(
            f"ERROR: Pipeline file '{filename}' not found in '{PIPELINE_PATH}' ensure data_pipeline.py has been run and that the correct filename is provided."
        )

    with open(file_path, "rb") as file:
        data = cloudpickle.load(file)
    return data["X"], data["y"]


def train_model():

    print(f"Loading pipeline results...\n")
    X_train, y_train = load_pipeline("train.pkl")
    X_test, y_test = load_pipeline("test.pkl")
    X_val, y_val = load_pipeline("val.pkl")
    
    print(f"Training data size: {len(X_train)} words")
    print(f"Testing data size: {len(X_test)} words")
    print(f"Validation data size: {len(X_val)} words\n")

    clf_pipeline = Pipeline([
        ("vectorizer", DictVectorizer(sparse=True)), 
        ("caster", Sparse32Caster()), 
        ("classifier", SGDClassifier(loss="hinge", class_weight="balanced", random_state=42, n_jobs=-1))   
    ])

    print(f"Training model...\n")
    clf_pipeline.fit(X_train, y_train) 

    print("Evaluate pinoybot on Validation Data:")
    print(f"------------------------------------------------")
    y_validation_pred = clf_pipeline.predict(X_val)
    print("Validation Performance:")
    print(classification_report(y_val, y_validation_pred))

    print("Evaluate pinoybot on Test Data:")
    print(f"------------------------------------------------")
    y_test_pred = clf_pipeline.predict(X_test)
    print("Test Performance:")
    print(classification_report(y_test, y_test_pred))
    
    cm = confusion_matrix(y_test, y_test_pred, labels=clf_pipeline.classes_)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual: {label}" for label in clf_pipeline.classes_],
        columns=[f"Predicted: {label}" for label in clf_pipeline.classes_] 
    )
    print("Test Data Confusion Matrix:")
    print(f"------------------------------------------------")
    print(cm_df)
    print(f"\nTest Data Accuracy: {accuracy_score(y_test, y_test_pred):.4f}\n")
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as model_file:
        cloudpickle.dump(clf_pipeline, model_file)
    print(f"Model saved as '{MODEL_PATH}'")

if __name__ == "__main__":
    train_model()