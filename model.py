import os
import pickle
import cloudpickle
import pandas as pd

from pathlib import Path
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

PIPELINE_PATH = Path(__file__).parent / "pipeline_output"
MODEL_DIR = Path(__file__).parent / "model_output"
MODEL_PATH = MODEL_DIR / "pinoybot_language-model.pkl"

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
    Trains the pinoybot language model with RandomForestClassifier using the generated data from data_pipeline.py
    The trained model is saved to the model_output directory as 'pinoybot_language-model.pkl' and evaluated on the test and validation datasets.
    The classification report, confusion matrix, and accuracy scores are printed to the console for visual validation of the model performance.
    """
    # Load the training, testing, and validation data from the pipeline files
    X_train, y_train = load_pipeline("train.pkl")
    X_test, y_test = load_pipeline("test.pkl")
    X_val, y_val = load_pipeline("val.pkl")
    
    print(f"DEBUG:\n")
    print(f"Training data size: {len(X_train)} words\n")
    print(f"Testing data size: {len(X_test)} words\n")
    print(f"Validation data size: {len(X_val)} words\n")


    # Create classifier pipeline for training pinoybot language model.
    clf_pipeline = Pipeline([
        ("vectorizer", DictVectorizer(sparse=True)), # use DictVectorizer to convert feature dictionaries to a sparse matrix
        ("classifier", RandomForestClassifier(n_estimators=500, random_state=42, verbose=1, class_weight="balanced")) # use RandomForestClassifier with 300 trees and a fixed random state for reproducibility
    ])
    
    # Train the model 
    clf_pipeline.fit(X_train, y_train) 

    # Model Evaluation
    print("Evaluate pinoybot on Validation Data:")
    y_validation_pred = clf_pipeline.predict(X_val)
    print("Validation Performance:")
    print(classification_report(y_val, y_validation_pred))

    print("Evaluate pinoybot on Test Data:")
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
    print(cm_df)
    print(f"\nTest Data Accuracy: {accuracy_score(y_test, y_test_pred):.4f}\n")
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as model_file:
        cloudpickle.dump(clf_pipeline, model_file)
    print(f"Model saved as '{MODEL_PATH}'")

if __name__ == "__main__":
    train_model()