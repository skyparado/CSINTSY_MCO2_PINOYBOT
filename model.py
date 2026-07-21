import os
import pickle
import pandas as pd

from pathlib import Path
from itertools import groupby
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

import sklearn_crfsuite

PIPELINE_PATH = Path(__file__).parent / "pipeline_output"
MODEL_DIR = Path(__file__).parent / "model_output"
MODEL_PATH = MODEL_DIR / "pinoybot_language-model.pkl"

# Chosen on the VALIDATION set only (see the c1/c2 sweep). c1 is L1
# regularisation, c2 is L2. The relatively high c1 sparsifies the huge n-gram
# feature space, which is what lifted the CS class in particular.
CRF_C1 = 0.5
CRF_C2 = 0.01
CRF_MAX_ITER = 250


def load_pipeline(filename: str):
    """
      Loads the generated pipeline from the pipeline_output directory.
      Returns the X, y and groups data from the pipeline file. (test, train, or validation)
    """
    file_path = PIPELINE_PATH / filename #makesure filename input includes .pkl
    if not file_path.exists():
        raise FileNotFoundError(
            f"ERROR: Pipeline file '{filename}' not found in '{PIPELINE_PATH}' ensure data_pipeline.py has been run and that the correct filename is provided."
        )

    with open(file_path, "rb") as file:
        data = pickle.load(file)

    if "groups" not in data:
        raise KeyError(
            f"'{filename}' has no 'groups' key -- it was written by an older "
            "version of data_pipeline.py. Re-run 'python data_pipeline.py' to "
            "regenerate the splits with sentence boundaries."
        )
    return data["X"], data["y"], data["groups"]


def to_sequences(X, y, groups):
    """Regroup flat word-level rows back into per-sentence sequences.

    The CRF does not classify words independently -- it labels a whole
    sentence at once, so that its learned transition scores (e.g. "FIL is
    usually followed by FIL") can be applied. That requires the data shaped as
    one list per sentence instead of one row per word.

    Consecutive rows sharing a sentence_id are grouped together; the pipeline
    writes rows in sentence order, so this reconstructs the original sentences.

    Args:
        X:      Flat list of per-word feature dicts.
        y:      Flat list of per-word labels.
        groups: Flat list of sentence_ids, parallel to X and y.

    Returns:
        (X_seqs, y_seqs): lists of per-sentence lists.
    """
    X_seqs, y_seqs = [], []
    i = 0
    for _, run in groupby(groups):
        n = len(list(run))
        # crfsuite accepts str/bool/float attribute values; cast everything else
        # (ints, numpy scalars) to float so it does not fail inside the C layer.
        X_seqs.append([
            {k: (v if isinstance(v, (str, bool)) else float(v)) for k, v in d.items()}
            for d in X[i:i + n]
        ])
        y_seqs.append(list(y[i:i + n]))
        i += n
    return X_seqs, y_seqs


def evaluate(crf, X_seqs, y_seqs, title):
    """Predict a set of sentences and print a per-class report.

    Returns the flattened (true, predicted) label lists so the caller can build
    a confusion matrix without predicting twice.
    """
    y_true = [tag for seq in y_seqs for tag in seq]
    y_pred = [tag for seq in crf.predict(X_seqs) for tag in seq]
    print(f"{title}:")
    print(classification_report(y_true, y_pred))
    return y_true, y_pred


def train_model():
    """
    Trains the pinoybot language model with a Conditional Random Field using the generated data from data_pipeline.py

    A CRF is used instead of a per-word classifier because language identification
    in code-switched text is a SEQUENCE labelling problem: whether a word is
    Filipino or English depends heavily on the words around it, and the CRF
    models those label-to-label transitions directly.

    The trained model is saved to the model_output directory as 'pinoybot_language-model.pkl' and evaluated on the test and validation datasets.
    The classification report, confusion matrix, and accuracy scores are printed to the console for visual validation of the model performance.
    """
    # Load the training, testing, and validation data from the pipeline files
    X_train, y_train, g_train = load_pipeline("train.pkl")
    X_test, y_test, g_test = load_pipeline("test.pkl")
    X_val, y_val, g_val = load_pipeline("val.pkl")

    Xs_train, ys_train = to_sequences(X_train, y_train, g_train)
    Xs_test, ys_test = to_sequences(X_test, y_test, g_test)
    Xs_val, ys_val = to_sequences(X_val, y_val, g_val)

    print(f"DEBUG:\n")
    print(f"Training data size: {len(X_train)} words in {len(Xs_train)} sentences\n")
    print(f"Testing data size: {len(X_test)} words in {len(Xs_test)} sentences\n")
    print(f"Validation data size: {len(X_val)} words in {len(Xs_val)} sentences\n")

    def new_crf():
        return sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=CRF_C1,
            c2=CRF_C2,
            max_iterations=CRF_MAX_ITER,
            all_possible_transitions=True,  # score label pairs never seen in training
        )

    # --- Stage 1: fit on TRAIN only, score VALIDATION -------------------
    # This is the honest validation number: the model being scored here has
    # never seen a validation sentence.
    crf = new_crf()
    crf.fit(Xs_train, ys_train)

    print("Evaluate pinoybot on Validation Data:")
    evaluate(crf, Xs_val, ys_val, "Validation Performance")

    # --- Stage 2: refit on TRAIN + VALIDATION, score TEST --------------
    # c1/c2 were already chosen using validation, so validation has done its
    # job and holding it out of training now only wastes 15% of the data.
    # Refitting on both is worth ~+0.4pp accuracy. TEST is still untouched by
    # both training and tuning, so the test number below stays honest -- and
    # it is the model saved to disk, so the reported score matches what ships.
    print("Refitting on train + validation for the final model...")
    crf = new_crf()
    crf.fit(Xs_train + Xs_val, ys_train + ys_val)

    print("Evaluate pinoybot on Test Data:")
    y_true, y_pred = evaluate(crf, Xs_test, ys_test, "Test Performance")

    # Print confusion matrix for test data to visualize model performance
    labels = sorted(crf.classes_)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual: {label}" for label in labels],
        columns=[f"Predicted: {label}" for label in labels]
    )
    print("Test Data Confusion Matrix:")
    print(cm_df)
    print(f"\nTest Data Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Test Data Macro F1: {f1_score(y_true, y_pred, average='macro'):.4f}\n")

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump(crf, model_file)
    size_mb = os.path.getsize(MODEL_PATH) / 1e6
    print(f"Model saved as '{MODEL_PATH}' ({size_mb:.1f} MB)")

if __name__ == "__main__":
    train_model()
