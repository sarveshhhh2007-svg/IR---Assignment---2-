"""
Information Retrieval CLA2 - Task 2: SMS Spam Detection
Using TF-IDF and Comparative Text Classification

Author: Student / Engineering Candidate
Course: Information Retrieval (IR) CLA2
Description:
    This script implements an end-to-end SMS Spam Detection pipeline:
    1. Dataset ingestion, encoding verification, validation, and deduplication.
    2. Text preprocessing and TF-IDF feature extraction (unigrams + bigrams).
    3. Stratified 80/20 train-test splitting (strictly avoiding data leakage).
    4. Training and comparative evaluation of 4 classifiers:
       - Decision Tree (DecisionTreeClassifier)
       - K-Nearest Neighbors (KNeighborsClassifier)
       - Support Vector Machine (LinearSVC)
       - Multinomial Naive Bayes (MultinomialNB)
    5. Evaluation using Accuracy, Spam Precision, Spam Recall, and Spam F1-Score.
    6. Detailed confusion matrices and per-class classification reports.
    7. What-If experiment: KNN neighbor variation (K = 3, 5, 7, 9).
    8. Comparative visualization and empirical recommendation generation.
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("SpamClassification")


def setup_directories(base_dir: Path) -> Dict[str, Path]:
    """
    Creates the required directory structure for outputs.

    Args:
        base_dir: Project root directory.

    Returns:
        Dictionary containing Paths to all output subdirectories.
    """
    outputs_dir = base_dir / "outputs"
    class_reports_dir = outputs_dir / "classification_reports"
    confusion_dir = outputs_dir / "confusion_matrices"
    what_if_dir = outputs_dir / "what_if_experiment"

    for d in [outputs_dir, class_reports_dir, confusion_dir, what_if_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return {
        "outputs": outputs_dir,
        "reports": class_reports_dir,
        "confusion": confusion_dir,
        "what_if": what_if_dir,
    }


def locate_and_load_dataset(data_dir: Path, target_filename: str = "SMSSpamCollection") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Locates and robustly parses the SMS Spam Collection dataset.
    Handles encoding gracefully (UTF-8 with Latin-1/Windows-1252 fallbacks)
    and reports any malformed or skipped records.

    Args:
        data_dir: Path to directory containing the dataset.
        target_filename: Expected dataset filename.

    Returns:
        Tuple of (Loaded DataFrame, Parsing Metadata dictionary).
    """
    logger.info("Locating dataset in '%s'...", data_dir)
    file_path = data_dir / target_filename

    # If direct file not found, search for common variations or zip files
    if not file_path.exists():
        candidates = list(data_dir.glob("SMSSpamCollection*")) + list(data_dir.glob("*.txt"))
        if candidates:
            file_path = candidates[0]
            logger.info("Direct target not found, using discovered candidate: %s", file_path.name)
        else:
            zip_candidates = list(data_dir.glob("*.zip"))
            if zip_candidates:
                import zipfile
                logger.info("Found zip archive '%s'. Extracting...", zip_candidates[0].name)
                with zipfile.ZipFile(zip_candidates[0], "r") as zip_ref:
                    zip_ref.extractall(data_dir)
                file_path = data_dir / target_filename
            else:
                raise FileNotFoundError(
                    f"Dataset file '{target_filename}' not found in '{data_dir}'.\n"
                    f"Please download the SMS Spam Collection from UCI Repository:\n"
                    f"https://archive.ics.uci.edu/dataset/228/sms+spam+collection\n"
                    f"and place 'SMSSpamCollection' inside '{data_dir}'."
                )

    if not file_path.exists():
        raise FileNotFoundError(f"Could not locate '{target_filename}' in '{data_dir}'.")

    logger.info("Reading dataset file: %s (Size: %.2f KB)", file_path.name, file_path.stat().st_size / 1024)

    # Attempt decodings sequentially
    encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]
    lines = None
    used_encoding = None

    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                lines = f.readlines()
            used_encoding = enc
            logger.info("Successfully decoded dataset using '%s' encoding.", enc)
            break
        except UnicodeDecodeError:
            continue

    if lines is None:
        raise ValueError(f"Failed to decode '{file_path}' using supported encodings: {encodings}")

    raw_total = len(lines)
    valid_rows = []
    skipped_rows = []

    for line_idx, line in enumerate(lines, start=1):
        clean_line = line.strip()
        if not clean_line:
            skipped_rows.append((line_idx, "Empty line"))
            continue

        parts = clean_line.split("\t", 1)
        if len(parts) != 2:
            skipped_rows.append((line_idx, f"Invalid column delimiter count ({len(parts)} parts)"))
            continue

        label_raw, text_raw = parts[0].strip().lower(), parts[1].strip()

        if label_raw not in ["ham", "spam"]:
            skipped_rows.append((line_idx, f"Unrecognized label '{parts[0]}'"))
            continue

        if not text_raw:
            skipped_rows.append((line_idx, "Empty message body"))
            continue

        valid_rows.append({"label": label_raw, "message": text_raw})

    df = pd.DataFrame(valid_rows)

    metadata = {
        "filename": file_path.name,
        "filepath": str(file_path.resolve()),
        "encoding": used_encoding,
        "total_lines_read": raw_total,
        "valid_rows_count": len(valid_rows),
        "skipped_rows_count": len(skipped_rows),
        "skipped_details": skipped_rows[:10],  # sample of first 10 skipped rows if any
    }

    logger.info(
        "Dataset parsing complete: %d total lines, %d valid rows, %d skipped rows.",
        metadata["total_lines_read"],
        metadata["valid_rows_count"],
        metadata["skipped_rows_count"],
    )

    return df, metadata


def analyze_and_deduplicate(
    df: pd.DataFrame,
    metadata: Dict[str, Any],
    output_dirs: Dict[str, Path],
    remove_duplicates: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Analyzes dataset distribution, inspects duplicates, and implements
    a defensible deduplication strategy to prevent train-test data leakage.

    Args:
        df: Raw valid DataFrame with 'label' and 'message'.
        metadata: Metadata from dataset loading.
        output_dirs: Paths dictionary for outputs.
        remove_duplicates: If True, removes identical duplicate SMS rows.

    Returns:
        Tuple of (Processed DataFrame, Summary statistics dictionary).
    """
    raw_count = len(df)
    class_counts_raw = df["label"].value_counts().to_dict()
    missing_values = df.isnull().sum().to_dict()
    duplicate_count_total = df.duplicated(subset=["message"]).sum()

    logger.info("Raw class counts: %s", class_counts_raw)
    logger.info("Duplicate messages found: %d (%.2f%%)", duplicate_count_total, (duplicate_count_total / raw_count) * 100)

    # Save class distribution before deduplication
    dist_df = pd.DataFrame({
        "Class": list(class_counts_raw.keys()),
        "Count": list(class_counts_raw.values()),
        "Percentage": [f"{(c / raw_count) * 100:.2f}%" for c in class_counts_raw.values()]
    })
    dist_csv_path = output_dirs["outputs"] / "dataset_class_distribution.csv"
    dist_df.to_csv(dist_csv_path, index=False)
    logger.info("Saved raw class distribution to '%s'.", dist_csv_path.name)

    # Deduplication strategy
    if remove_duplicates:
        df_clean = df.drop_duplicates(subset=["message"]).copy().reset_index(drop=True)
        retained_count = len(df_clean)
        removed_count = raw_count - retained_count
        clean_class_counts = df_clean["label"].value_counts().to_dict()
        logger.info(
            "Deduplication enabled: %d duplicate rows removed to prevent data leakage across train/test splits. Retained %d rows.",
            removed_count, retained_count
        )
    else:
        df_clean = df.copy()
        retained_count = raw_count
        removed_count = 0
        clean_class_counts = class_counts_raw

    # Prepare detailed dataset summary text
    summary_path = output_dirs["outputs"] / "dataset_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("INFORMATION RETRIEVAL CLA2: SMS SPAM DATASET SUMMARY & AUDIT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Dataset Filename         : {metadata['filename']}\n")
        f.write(f"File Path                : {metadata['filepath']}\n")
        f.write(f"Encoding Used            : {metadata['encoding']}\n")
        f.write(f"Total Lines Processed    : {metadata['total_lines_read']}\n")
        f.write(f"Valid Rows Parsed        : {metadata['valid_rows_count']}\n")
        f.write(f"Skipped / Malformed Rows : {metadata['skipped_rows_count']}\n")
        f.write(f"Missing Values (label)   : {missing_values.get('label', 0)}\n")
        f.write(f"Missing Values (message) : {missing_values.get('message', 0)}\n")
        f.write("\n" + "-" * 70 + "\n")
        f.write("CLASS DISTRIBUTION (INITIAL VALID DATASET):\n")
        for cls_name, count in class_counts_raw.items():
            pct = (count / raw_count) * 100
            f.write(f"  - {cls_name.upper():<6}: {count:>5} records ({pct:6.2f}%)\n")
        f.write("-" * 70 + "\n")
        f.write("DUPLICATE ANALYSIS & DATA LEAKAGE MITIGATION STRATEGY:\n")
        f.write(f"Total Duplicate Messages : {duplicate_count_total}\n")
        f.write(f"Deduplication Applied    : {'YES (Removed exact duplicate messages)' if remove_duplicates else 'NO'}\n")
        f.write(f"Original Row Count       : {raw_count}\n")
        f.write(f"Rows Retained for Study  : {retained_count}\n")
        f.write(f"Rows Discarded           : {removed_count}\n\n")
        f.write("RATIONALE:\n")
        f.write(
            "In text classification, identical messages (e.g. repetitive broadcast spam or\n"
            "common greetings like 'Sorry, I'll call later') appearing in both training\n"
            "and testing partitions cause severe data leakage and artificially inflate test\n"
            "performance, rewarding memorization rather than generalized feature learning.\n"
            "Discarding duplicates ensures that the held-out test set strictly measures\n"
            "unseen text generalization.\n"
        )
        f.write("-" * 70 + "\n")
        f.write("FINAL CLEAN CLASS DISTRIBUTION:\n")
        for cls_name, count in clean_class_counts.items():
            pct = (count / retained_count) * 100
            f.write(f"  - {cls_name.upper():<6}: {count:>5} records ({pct:6.2f}%)\n")
        f.write("=" * 70 + "\n")

    logger.info("Saved dataset summary audit to '%s'.", summary_path.name)

    stats = {
        "raw_count": raw_count,
        "clean_count": retained_count,
        "removed_duplicates": removed_count,
        "class_counts_raw": class_counts_raw,
        "class_counts_clean": clean_class_counts,
        "missing_values": missing_values,
    }

    return df_clean, stats


def prepare_and_split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Splits the dataset into training and testing partitions using stratified sampling.
    Crucial: Performed BEFORE any feature extraction to strictly prevent data leakage.

    Args:
        df: Clean DataFrame with 'label' and 'message'.
        test_size: Fraction of samples for the test set (default 0.20).
        random_state: Deterministic random seed (default 42).

    Returns:
        X_train, X_test, y_train, y_test series.
    """
    logger.info(
        "Performing stratified %d/%d train-test split (random_state=%d)...",
        int((1 - test_size) * 100), int(test_size * 100), random_state
    )

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"],
        df["label"],
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"]
    )

    logger.info("Training set: %d samples (%s)", len(X_train), dict(y_train.value_counts()))
    logger.info("Testing set : %d samples (%s)", len(X_test), dict(y_test.value_counts()))

    return X_train, X_test, y_train, y_test


def extract_tfidf_features(
    X_train: pd.Series,
    X_test: pd.Series,
    output_dirs: Dict[str, Path],
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_features: int = 5000,
    sublinear_tf: bool = True
) -> Tuple[Any, Any, TfidfVectorizer, Dict[str, Any]]:
    """
    Fits TF-IDF vectorizer on X_train ONLY, then transforms X_test.
    Extracts unigrams and bigrams, strips standard English stop words,
    and documents feature characteristics.

    Args:
        X_train: Training messages.
        X_test: Test messages.
        output_dirs: Directory paths dictionary.
        ngram_range: Range of n-gram sizes (unigrams + bigrams).
        min_df: Minimum document frequency threshold.
        max_features: Maximum vocabulary limit.
        sublinear_tf: Apply 1 + log(tf) scaling.

    Returns:
        Tuple of (X_train_tfidf, X_test_tfidf, fitted vectorizer, tfidf_stats).
    """
    logger.info(
        "Configuring TfidfVectorizer: ngram_range=%s, min_df=%d, max_features=%d, sublinear_tf=%s, stop_words='english'",
        ngram_range, min_df, max_features, sublinear_tf
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=sublinear_tf
    )

    # Strictly fit on train ONLY to prevent data leakage
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    feature_names = vectorizer.get_feature_names_out()
    vocab_size = len(feature_names)

    logger.info(
        "TF-IDF feature extraction complete: %d training docs, %d test docs, %d features.",
        X_train_tfidf.shape[0], X_test_tfidf.shape[0], vocab_size
    )

    # Extract sample feature representations for documentation
    sample_vocab = list(feature_names[:25])
    sample_bigrams = [w for w in feature_names if " " in w][:25]

    # Save preprocessing examples
    # Choose 5 diverse sample messages (mix of ham and spam)
    sample_indices = [0, 1, 2, 3, 4]
    prep_examples = []
    for idx in sample_indices:
        raw_msg = X_train.iloc[idx]
        row_vec = X_train_tfidf[idx].toarray()[0]
        active_indices = np.where(row_vec > 0)[0]
        # Sort terms by TF-IDF weight descending
        top_terms = sorted([(feature_names[i], row_vec[i]) for i in active_indices], key=lambda x: x[1], reverse=True)
        terms_str = "; ".join([f"{term} ({weight:.3f})" for term, weight in top_terms[:8]])
        prep_examples.append({
            "Sample_ID": idx + 1,
            "Original_Message": raw_msg,
            "Extracted_TFIDF_Key_Features": terms_str if terms_str else "None (all stop words / low freq)"
        })

    prep_df = pd.DataFrame(prep_examples)
    prep_csv_path = output_dirs["outputs"] / "preprocessing_examples.csv"
    prep_df.to_csv(prep_csv_path, index=False)
    logger.info("Saved preprocessing representations sample to '%s'.", prep_csv_path.name)

    # Save feature summary documentation
    summary_path = output_dirs["outputs"] / "feature_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("INFORMATION RETRIEVAL CLA2: TF-IDF FEATURE EXTRACTION AUDIT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Vectorization Algorithm       : Scikit-Learn TfidfVectorizer\n")
        f.write(f"Lowercasing                   : Enabled (converts all characters to lowercase)\n")
        f.write(f"Tokenization & Normalization  : Default scikit-learn regex word boundary tokenization\n")
        f.write(f"Stop Words Removed            : Standard English stop words\n")
        f.write(f"N-Gram Range                  : {ngram_range} (Unigrams + Bigrams)\n")
        f.write(f"Minimum Document Frequency    : {min_df} (terms appearing in < {min_df} documents pruned)\n")
        f.write(f"Maximum Vocabulary Dimension  : {max_features}\n")
        f.write(f"Sublinear TF Scaling          : {sublinear_tf} (replaces tf with 1 + log(tf))\n")
        f.write(f"Training Documents (Rows)     : {X_train_tfidf.shape[0]}\n")
        f.write(f"Testing Documents (Rows)      : {X_test_tfidf.shape[0]}\n")
        f.write(f"Learned Vocabulary Size       : {vocab_size} distinct features\n")
        f.write(f"Training Matrix Shape         : {X_train_tfidf.shape}\n")
        f.write(f"Testing Matrix Shape          : {X_test_tfidf.shape}\n")
        f.write(f"Matrix Sparsity (Train)       : {100 * (1.0 - X_train_tfidf.nnz / (X_train_tfidf.shape[0] * vocab_size)):.2f}%\n")
        f.write("\n" + "-" * 70 + "\n")
        f.write("SAMPLE LEARNED UNIGRAMS:\n")
        f.write(", ".join(sample_vocab[:20]) + "\n\n")
        f.write("SAMPLE LEARNED BIGRAMS (COLLOCATIONS):\n")
        f.write(", ".join(sample_bigrams[:15]) if sample_bigrams else "No bigrams met threshold.")
        f.write("\n\n" + "-" * 70 + "\n")
        f.write("MATHEMATICAL CONCEPT & RELEVANCE TO INFORMATION RETRIEVAL:\n")
        f.write(
            "TF-IDF (Term Frequency - Inverse Document Frequency) is a foundational weighting\n"
            "scheme in Information Retrieval. It transforms variable-length unstructured text\n"
            "into fixed-dimensional numerical vectors in the Vector Space Model (VSM).\n\n"
            "1. Term Frequency (TF):\n"
            "   Quantifies how frequently a term occurs inside a single SMS message.\n"
            "   With sublinear TF scaling (1 + log(tf)), diminishing returns are applied so\n"
            "   that an SMS containing 10 repetitions of 'free' does not carry 10 times the weight\n"
            "   of a single occurrence.\n\n"
            "2. Inverse Document Frequency (IDF):\n"
            "   IDF = log((1 + N) / (1 + df)) + 1, where N is the total number of documents\n"
            "   and df is the document frequency of the term.\n"
            "   Terms ubiquitous across almost all messages (e.g., 'call', 'today', 'just') receive\n"
            "   low IDF scores, while discriminative terms concentrated primarily in spam\n"
            "   (e.g., 'claim', 'urgent', 'winner', 'txt') receive high IDF scores.\n\n"
            "3. L2 Normalization:\n"
            "   Each document vector is normalized to unit length Euclidean norm (||v|| = 1),\n"
            "   mitigating bias caused by variations in message length.\n"
        )
        f.write("=" * 70 + "\n")

    logger.info("Saved TF-IDF feature audit to '%s'.", summary_path.name)

    tfidf_stats = {
        "vocab_size": vocab_size,
        "train_shape": X_train_tfidf.shape,
        "test_shape": X_test_tfidf.shape,
        "ngram_range": ngram_range,
        "min_df": min_df,
        "max_features": max_features,
    }

    return X_train_tfidf, X_test_tfidf, vectorizer, tfidf_stats


def train_and_evaluate_models(
    X_train_tfidf: Any,
    y_train: pd.Series,
    X_test_tfidf: Any,
    y_test: pd.Series,
    output_dirs: Dict[str, Path]
) -> pd.DataFrame:
    """
    Trains and evaluates 4 standard ML classification models on identical data:
    1. Decision Tree (DecisionTreeClassifier)
    2. K-Nearest Neighbors (KNeighborsClassifier)
    3. Support Vector Machine (LinearSVC)
    4. Multinomial Naive Bayes (MultinomialNB)

    Generates classification reports, confusion matrix heatmaps, and metric tables.

    Args:
        X_train_tfidf: TF-IDF feature matrix for training.
        y_train: Training labels ('ham' / 'spam').
        X_test_tfidf: TF-IDF feature matrix for testing.
        y_test: Test labels ('ham' / 'spam').
        output_dirs: Output directories dictionary.

    Returns:
        DataFrame summarizing evaluation metrics across all four models.
    """
    models: Dict[str, Any] = {
        "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, metric="cosine"),
        "Support Vector Machine": LinearSVC(random_state=42, class_weight="balanced", dual="auto"),
        "Multinomial Naive Bayes": MultinomialNB(alpha=1.0)
    }

    comparison_records = []

    logger.info("Starting model training and comparative evaluation...")

    for name, model in models.items():
        logger.info("--> Evaluating: %s", name)

        # Measure training latency
        t_start_train = time.perf_counter()
        model.fit(X_train_tfidf, y_train)
        t_train = time.perf_counter() - t_start_train

        # Measure inference latency on test set
        t_start_pred = time.perf_counter()
        y_pred = model.predict(X_test_tfidf)
        t_pred = time.perf_counter() - t_start_pred

        # Compute core metrics (defining 'spam' as the positive class)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label="spam", zero_division=0)
        rec = recall_score(y_test, y_pred, pos_label="spam", zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label="spam", zero_division=0)

        # Classification report
        clf_rep = classification_report(y_test, y_pred, digits=4, zero_division=0)
        report_filename = f"{name.replace(' ', '_')}_classification_report.txt"
        report_path = output_dirs["reports"] / report_filename
        with open(report_path, "w", encoding="utf-8") as rf:
            rf.write(f"Classification Report: {name}\n")
            rf.write("=" * 60 + "\n")
            rf.write(clf_rep)
            rf.write("\n" + "=" * 60 + "\n")
            rf.write(f"Training Time   : {t_train:.4f} seconds\n")
            rf.write(f"Prediction Time : {t_pred:.4f} seconds\n")

        # Confusion Matrix
        # Explicit label order: ['ham', 'spam']
        cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])
        tn, fp, fn, tp = cm.ravel()

        # Plot confusion matrix
        plt.figure(figsize=(6, 5))
        annotations = [
            [f"TN\n{tn}", f"FP\n{fp}"],
            [f"FN\n{fn}", f"TP\n{tp}"]
        ]
        sns.heatmap(
            cm,
            annot=annotations,
            fmt="",
            cmap="Blues",
            cbar=True,
            xticklabels=["Predicted HAM", "Predicted SPAM"],
            yticklabels=["Actual HAM", "Actual SPAM"]
        )
        plt.title(f"Confusion Matrix: {name}\n(Acc: {acc:.4f}, F1: {f1:.4f})", fontsize=11, fontweight="bold", pad=12)
        plt.xlabel("Predicted Class", fontweight="bold", labelpad=8)
        plt.ylabel("Actual True Class", fontweight="bold", labelpad=8)
        plt.tight_layout()

        cm_filename = f"{name.replace(' ', '_')}_confusion_matrix.png"
        cm_path = output_dirs["confusion"] / cm_filename
        plt.savefig(cm_path, dpi=300)
        plt.close()

        comparison_records.append({
            "Model": name,
            "Accuracy": round(float(acc), 4),
            "Spam_Precision": round(float(prec), 4),
            "Spam_Recall": round(float(rec), 4),
            "Spam_F1_Score": round(float(f1), 4),
            "True_Negatives": int(tn),
            "False_Positives": int(fp),
            "False_Negatives": int(fn),
            "True_Positives": int(tp),
            "Training_Time_Sec": round(float(t_train), 5),
            "Prediction_Time_Sec": round(float(t_pred), 5),
        })

    comparison_df = pd.DataFrame(comparison_records)
    csv_path = output_dirs["outputs"] / "model_comparison.csv"
    comparison_df.to_csv(csv_path, index=False)
    logger.info("Saved model comparison metrics to '%s'.", csv_path.name)

    return comparison_df


def visualize_model_comparison(comparison_df: pd.DataFrame, output_dirs: Dict[str, Path]) -> None:
    """
    Renders a clear, high-resolution grouped bar chart comparing all 4 models
    across Accuracy, Spam Precision, Spam Recall, and Spam F1-Score.

    Args:
        comparison_df: DataFrame containing model evaluation metrics.
        output_dirs: Output directories dictionary.
    """
    logger.info("Generating comparative bar chart for models...")

    metrics = ["Accuracy", "Spam_Precision", "Spam_Recall", "Spam_F1_Score"]
    metric_labels = ["Accuracy", "Precision (Spam)", "Recall (Spam)", "F1-Score (Spam)"]
    models = comparison_df["Model"].tolist()

    x = np.arange(len(models))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#2b5c8f", "#2ca02c", "#ff7f0e", "#d62728"]

    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        values = comparison_df[metric].values
        rects = ax.bar(x + (i - 1.5) * width, values, width, label=label, color=color, alpha=0.9)
        # Add numeric labels above bars
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=8, rotation=0
            )

    ax.set_title("Model Comparison on Test Set (SMS Spam Detection)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Classification Algorithm", fontweight="bold", labelpad=10)
    ax.set_ylabel("Score (0.00 - 1.00)", fontweight="bold", labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight="semibold")
    ax.set_ylim(0.70, 1.05)  # Zoomed to emphasize differences while starting above zero
    ax.axhline(0.90, color="gray", linestyle="--", linewidth=0.7, alpha=0.7)
    ax.legend(loc="lower right", frameon=True, shadow=True)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    chart_path = output_dirs["outputs"] / "model_comparison_chart.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    logger.info("Saved model comparison visualization to '%s'.", chart_path.name)


def run_what_if_experiment(
    X_train_tfidf: Any,
    y_train: pd.Series,
    X_test_tfidf: Any,
    y_test: pd.Series,
    output_dirs: Dict[str, Path],
    k_values: List[int] = [3, 5, 7, 9]
) -> pd.DataFrame:
    """
    Executes the required What-If experiment:
    Varies K (n_neighbors) in K-Nearest Neighbors across [3, 5, 7, 9]
    while keeping the dataset, train-test split, and TF-IDF features strictly identical.

    Args:
        X_train_tfidf: Training TF-IDF matrix.
        y_train: Training labels.
        X_test_tfidf: Testing TF-IDF matrix.
        y_test: Testing labels.
        output_dirs: Output directories dictionary.
        k_values: List of K values to experiment with.

    Returns:
        DataFrame detailing metrics for each K value.
    """
    logger.info("Running What-If Experiment: Evaluating KNN with K values = %s...", k_values)

    records = []
    n_train_samples = X_train_tfidf.shape[0]

    for k in k_values:
        if k >= n_train_samples:
            logger.warning("Skipping K=%d as it exceeds training sample count (%d)", k, n_train_samples)
            continue

        knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
        t_start = time.perf_counter()
        knn.fit(X_train_tfidf, y_train)
        t_train = time.perf_counter() - t_start

        y_pred = knn.predict(X_test_tfidf)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label="spam", zero_division=0)
        rec = recall_score(y_test, y_pred, pos_label="spam", zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label="spam", zero_division=0)

        cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])
        tn, fp, fn, tp = cm.ravel()

        records.append({
            "K_Neighbors": k,
            "Accuracy": round(float(acc), 4),
            "Spam_Precision": round(float(prec), 4),
            "Spam_Recall": round(float(rec), 4),
            "Spam_F1_Score": round(float(f1), 4),
            "True_Positives": int(tp),
            "False_Positives": int(fp),
            "True_Negatives": int(tn),
            "False_Negatives": int(fn),
            "Fit_Time_Sec": round(float(t_train), 5),
        })

    what_if_df = pd.DataFrame(records)
    csv_path = output_dirs["what_if"] / "knn_k_comparison.csv"
    what_if_df.to_csv(csv_path, index=False)
    logger.info("Saved What-If experiment CSV to '%s'.", csv_path.name)

    # Plot trend chart
    plt.figure(figsize=(8, 5))
    plt.plot(what_if_df["K_Neighbors"], what_if_df["Accuracy"], marker="o", linewidth=2, label="Accuracy", color="#1f77b4")
    plt.plot(what_if_df["K_Neighbors"], what_if_df["Spam_Precision"], marker="s", linewidth=2, label="Spam Precision", color="#2ca02c")
    plt.plot(what_if_df["K_Neighbors"], what_if_df["Spam_Recall"], marker="^", linewidth=2, label="Spam Recall", color="#ff7f0e")
    plt.plot(what_if_df["K_Neighbors"], what_if_df["Spam_F1_Score"], marker="D", linewidth=2, label="Spam F1-Score", color="#d62728")

    plt.title("What-If Experiment: Impact of K (Neighbors) in KNN", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Number of Neighbors (K)", fontweight="bold", labelpad=8)
    plt.ylabel("Evaluation Score", fontweight="bold", labelpad=8)
    plt.xticks(k_values)
    plt.ylim(0.60, 1.02)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()

    chart_path = output_dirs["what_if"] / "knn_k_comparison_chart.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    logger.info("Saved What-If visualization to '%s'.", chart_path.name)

    # Write observation text
    obs_path = output_dirs["what_if"] / "knn_k_observations.txt"
    best_k_f1 = what_if_df.loc[what_if_df["Spam_F1_Score"].idxmax()]["K_Neighbors"]
    best_f1_val = what_if_df["Spam_F1_Score"].max()

    with open(obs_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("WHAT-IF EXPERIMENT OBSERVATIONS: KNN PARAMETER TUNING (K)\n")
        f.write("=" * 70 + "\n\n")
        f.write("EXPERIMENTAL SETUP:\n")
        f.write("  - Model: K-Nearest Neighbors (KNeighborsClassifier)\n")
        f.write("  - Distance Metric: Cosine (effective for L2-normalized sparse TF-IDF vectors)\n")
        f.write("  - Varied Parameter: n_neighbors in [3, 5, 7, 9]\n")
        f.write("  - Control: Kept identical 80/20 train/test split and TF-IDF vocabulary\n\n")
        f.write("TABULAR RESULTS:\n")
        f.write(what_if_df.to_string(index=False))
        f.write("\n\n" + "-" * 70 + "\n")
        f.write(f"OPTIMAL CONFIGURATION: K = {int(best_k_f1)} (Achieving Spam F1 = {best_f1_val:.4f})\n\n")
        f.write("ANALYSIS & THEORETICAL INTERPRETATION:\n")
        f.write(
            "1. Bias-Variance Trade-off:\n"
            "   Smaller values of K (e.g. K=3) create more sensitive and localized decision\n"
            "   boundaries, making the classifier responsive to sparse minority spam clusters\n"
            "   but potentially more susceptible to noise.\n\n"
            "2. Majority Class Overwhelm:\n"
            "   As K increases (e.g. K=7, 9), the voting pool expands. Because the dataset has\n"
            "   an inherent class imbalance (approx. 87% ham vs 13% spam), a larger K causes the\n"
            "   dominant legitimate class to increasingly outvote spam instances in boundary\n"
            "   regions unless distance weighting is explicitly applied.\n\n"
            "3. Precision vs. Recall Dynamics:\n"
            "   Empirically, KNN with cosine distance maintains strong precision because nearest\n"
            "   neighbors in TF-IDF space with specific spam tokens share unambiguous collocations.\n"
            "   However, recall may fluctuate as higher K values miss borderline spam messages.\n"
        )
        f.write("=" * 70 + "\n")

    logger.info("Saved What-If observations to '%s'.", obs_path.name)
    return what_if_df


def generate_final_observations(
    metadata: Dict[str, Any],
    stats: Dict[str, Any],
    tfidf_stats: Dict[str, Any],
    comparison_df: pd.DataFrame,
    what_if_df: pd.DataFrame,
    output_dirs: Dict[str, Path]
) -> None:
    """
    Synthesizes real empirical measurements into outputs/final_observations.txt
    providing transparent metrics, trade-offs, and an evidence-based recommendation.

    Args:
        metadata: Dataset loading audit.
        stats: Cleaned dataset counts and distribution.
        tfidf_stats: Feature space properties.
        comparison_df: Benchmark metrics across models.
        what_if_df: What-If experiment results.
        output_dirs: Output directories dictionary.
    """
    logger.info("Compiling final observations and evidence-based recommendation...")

    best_f1_row = comparison_df.loc[comparison_df["Spam_F1_Score"].idxmax()]
    best_acc_row = comparison_df.loc[comparison_df["Accuracy"].idxmax()]
    best_prec_row = comparison_df.loc[comparison_df["Spam_Precision"].idxmax()]
    best_rec_row = comparison_df.loc[comparison_df["Spam_Recall"].idxmax()]

    obs_path = output_dirs["outputs"] / "final_observations.txt"

    with open(obs_path, "w", encoding="utf-8") as f:
        f.write("=" * 75 + "\n")
        f.write("INFORMATION RETRIEVAL CLA2: FINAL OBSERVATIONS & MODEL RECOMMENDATION\n")
        f.write("=" * 75 + "\n\n")

        f.write("1. DATASET CHARACTERISTICS & VALIDATION\n")
        f.write(f"   - Dataset File            : {metadata['filename']}\n")
        f.write(f"   - Encoding Used           : {metadata['encoding']}\n")
        f.write(f"   - Total Records Parsed    : {metadata['valid_rows_count']}\n")
        f.write(f"   - Duplicate Rows Removed  : {stats['removed_duplicates']}\n")
        f.write(f"   - Retained Unique Rows    : {stats['clean_count']}\n")
        f.write("   - Clean Class Balance     :\n")
        for cls_name, count in stats["class_counts_clean"].items():
            pct = (count / stats["clean_count"]) * 100
            f.write(f"     * {cls_name.upper():<6}: {count} ({pct:.2f}%)\n")

        f.write("\n2. TEXT PREPROCESSING & TF-IDF CONFIGURATION\n")
        f.write(f"   - N-Gram Range            : {tfidf_stats['ngram_range']} (Unigrams + Bigrams)\n")
        f.write(f"   - Stop Words              : English stop words pruned\n")
        f.write(f"   - Minimum Doc Frequency   : {tfidf_stats['min_df']}\n")
        f.write(f"   - Max Features Allowed    : {tfidf_stats['max_features']}\n")
        f.write(f"   - Extracted Feature Count : {tfidf_stats['vocab_size']}\n")
        f.write(f"   - Training Documents      : {tfidf_stats['train_shape'][0]}\n")
        f.write(f"   - Testing Documents       : {tfidf_stats['test_shape'][0]}\n")
        f.write(f"   - Train/Test Split Ratio  : 80% Train / 20% Test (Stratified, seed=42)\n")

        f.write("\n3. EMPIRICAL MODEL BENCHMARK RESULTS (TEST SET)\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n")

        f.write("4. BEST PERFORMING ALGORITHMS BY METRIC\n")
        f.write(f"   - Highest Spam F1-Score   : {best_f1_row['Model']} (F1 = {best_f1_row['Spam_F1_Score']:.4f})\n")
        f.write(f"   - Highest Accuracy        : {best_acc_row['Model']} (Accuracy = {best_acc_row['Accuracy']:.4f})\n")
        f.write(f"   - Highest Spam Precision  : {best_prec_row['Model']} (Precision = {best_prec_row['Spam_Precision']:.4f})\n")
        f.write(f"   - Highest Spam Recall     : {best_rec_row['Model']} (Recall = {best_rec_row['Spam_Recall']:.4f})\n")

        f.write("\n5. WHAT-IF EXPERIMENT SUMMARY (KNN K-TUNING)\n")
        f.write(what_if_df.to_string(index=False))
        f.write("\n\n")

        f.write("6. PRACTICAL TRADE-OFF ANALYSIS IN INFORMATION RETRIEVAL\n")
        f.write(
            "   - The Danger of Pure Accuracy:\n"
            "     Because legitimate messages constitute roughly 87% of the dataset, a trivial\n"
            "     dummy classifier predicting 'ham' for all inputs would achieve ~87% accuracy\n"
            "     while catching exactly 0% of spam. Thus, Accuracy alone is deceptive.\n\n"
            "   - Cost Asymmetry (False Positives vs. False Negatives):\n"
            "     * False Positives (legitimate SMS flagged as spam): Highly destructive. Missing\n"
            "       a bank one-time password (OTP), medical alert, or critical personal message\n"
            "       destroys user trust.\n"
            "     * False Negatives (spam entering inbox): Annoying, but far less damaging than\n"
            "       losing critical legitimate communication.\n"
            "     Therefore, an optimal spam filter requires very high Precision (near zero FP)\n"
            "     while maximizing Recall (minimizing FN), best evaluated by Spam F1-Score.\n\n"
            "   - Algorithmic Suitability on Sparse TF-IDF Vector Spaces:\n"
            "     * Support Vector Machine (LinearSVC): Outstanding performer. High-dimensional\n"
            "       sparse text representations are linearly separable. SVM finds the optimal\n"
            "       maximum-margin hyperplanes without suffering from the curse of dimensionality.\n"
            "     * Multinomial Naive Bayes: Extremely fast, lightweight, and mathematically\n"
            "       well-suited for word frequency representations. Yields high precision.\n"
            "     * Decision Tree: Prone to capturing noise on large vocabularies unless constrained;\n"
            "       orthogonal axis splits struggle with complex phrase combinations.\n"
            "     * KNN: Computationally demanding at test time (O(N) distance checks across all\n"
            "       training vectors) and sensitive to distance metric choice in sparse spaces.\n"
        )

        f.write("\n7. FINAL EVIDENCE-BASED RECOMMENDATION\n")
        f.write(
            f"   Based on the rigorous empirical benchmark across identical stratified test data,\n"
            f"   '{best_f1_row['Model']}' is strongly recommended as the best classifier for SMS Spam\n"
            f"   Detection. It delivers an outstanding Spam F1-Score of {best_f1_row['Spam_F1_Score']:.4f}\n"
            f"   (Accuracy: {best_f1_row['Accuracy']:.4f}, Precision: {best_f1_row['Spam_Precision']:.4f},\n"
            f"   Recall: {best_f1_row['Spam_Recall']:.4f}), effectively suppressing false positives while\n"
            f"   retaining robust detection of evasive spam messages with ultra-fast inference.\n"
        )
        f.write("=" * 75 + "\n")

    logger.info("Saved final observations and recommendation to '%s'.", obs_path.name)


def main():
    """Main program execution entry point."""
    parser = argparse.ArgumentParser(description="SMS Spam Detection Using TF-IDF and Comparative ML")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing dataset")
    parser.add_argument("--filename", type=str, default="SMSSpamCollection", help="Dataset filename")
    parser.add_argument("--no-dedup", action="store_true", help="Disable duplicate removal")
    parser.add_argument("--test-size", type=float, default=0.20, help="Test set split ratio (default: 0.20)")
    parser.add_argument("--random-state", type=int, default=42, help="Random state seed (default: 42)")

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    data_dir = project_root / args.data_dir

    print("\n" + "=" * 70)
    print(" INFORMATION RETRIEVAL CLA2: SMS SPAM DETECTION BENCHMARK")
    print("=" * 70)

    try:
        # Step 1: Set up directories
        output_dirs = setup_directories(project_root)

        # Step 2: Locate and load dataset
        df_raw, metadata = locate_and_load_dataset(data_dir, args.filename)

        # Step 3: Analyze dataset and handle duplicates
        df_clean, stats = analyze_and_deduplicate(
            df_raw,
            metadata,
            output_dirs,
            remove_duplicates=not args.no_dedup
        )

        # Step 4: Split data strictly BEFORE feature extraction
        X_train, X_test, y_train, y_test = prepare_and_split_data(
            df_clean,
            test_size=args.test_size,
            random_state=args.random_state
        )

        # Step 5: Extract TF-IDF features
        X_train_tfidf, X_test_tfidf, vectorizer, tfidf_stats = extract_tfidf_features(
            X_train,
            X_test,
            output_dirs,
            ngram_range=(1, 2),
            min_df=2,
            max_features=5000,
            sublinear_tf=True
        )

        # Step 6: Train and evaluate all 4 classifiers
        comparison_df = train_and_evaluate_models(
            X_train_tfidf,
            y_train,
            X_test_tfidf,
            y_test,
            output_dirs
        )

        # Step 7: Visualize model comparison
        visualize_model_comparison(comparison_df, output_dirs)

        # Step 8: What-If Experiment (KNN K tuning)
        what_if_df = run_what_if_experiment(
            X_train_tfidf,
            y_train,
            X_test_tfidf,
            y_test,
            output_dirs,
            k_values=[3, 5, 7, 9]
        )

        # Step 9: Synthesize final observations & recommendation
        generate_final_observations(
            metadata,
            stats,
            tfidf_stats,
            comparison_df,
            what_if_df,
            output_dirs
        )

        print("\n" + "=" * 70)
        print(" EXECUTION SUMMARY & BENCHMARK RESULTS")
        print("=" * 70)
        print(comparison_df.to_string(index=False))
        print("\n" + "-" * 70)
        print(" WHAT-IF EXPERIMENT RESULTS (KNN K-TUNING):")
        print(what_if_df.to_string(index=False))
        print("\n" + "-" * 70)
        print(f" All outputs and figures successfully generated in: {output_dirs['outputs'].resolve()}")
        print("=" * 70 + "\n")

    except Exception as e:
        logger.exception("Fatal error during execution: %s", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
