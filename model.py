import os
import cloudpickle
import pandas as pd
import numpy as np
import scipy.sparse as sp

from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

PIPELINE_PATH = Path(__file__).parent / "pipeline_output"
MODEL_DIR = Path(__file__).parent / "model_output"
MODEL_PATH = MODEL_DIR / "pinoybot_language-model.pkl"

class Sparse32Caster(BaseEstimator, TransformerMixin):
    """
        Helper class that intercepts scipy sparse matrices and casts indices to int32 to prevent sklearn crashes.
    """
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        if sp.issparse(X):
            X.indices = X.indices.astype(np.int32)
            X.indptr = X.indptr.astype(np.int32)
        return X
    
    
def load_pipeline(filename: str): 
    """
      Loads the generated pipeline from the pipeline_output directory.
      Returns the X and y data from the pipeline file. (test, train, or validation) 
    """
    file_path = PIPELINE_PATH / filename #makesure filename input includes .pkl 
    if not file_path.exists():
        raise FileNotFoundError(
            f"ERROR: Pipeline file '{filename}' not found in '{PIPELINE_PATH}' ensure data_pipeline.py has been run and that the correct filename is provided."
        )

    with open(file_path, "rb") as file:
        data = cloudpickle.load(file)
    return data["X"], data["y"]


def train_model():
    """
        Trains the pinoybot language model with SGDClassifier using the generated data from data_pipeline.py
        The trained model is saved to the model_output directory as 'pinoybot_language-model.pkl' and evaluated on the test and validation datasets.
        The classification report, confusion matrix, and accuracy scores are printed to the console for visual validation of the model performance.
    """
    # Load the training, testing, and validation data from the pipeline files
    print(f"Loading pipeline results...\n")
    X_train, y_train = load_pipeline("train.pkl")
    X_test, y_test = load_pipeline("test.pkl")
    X_val, y_val = load_pipeline("val.pkl")
    
    print(f"Training data size: {len(X_train)} words")
    print(f"Testing data size: {len(X_test)} words")
    print(f"Validation data size: {len(X_val)} words\n")

    # Create classifier pipeline for training pinoybot language model.
    clf_pipeline = Pipeline([
        ("vectorizer", DictVectorizer(sparse=True)), # use DictVectorizer to convert feature dictionaries to a sparse matrix
        ("caster", Sparse32Caster()), # use the helper class to convert indices to int32
        ("classifier", SGDClassifier(loss="hinge", class_weight="balanced", random_state=42, n_jobs=-1))    
        ])

    # Train the model 
    print(f"Training model...\n")
    clf_pipeline.fit(X_train, y_train) 

    # Model Evaluation
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
    
    # Print confusion matrix for test data to visualize model performance
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