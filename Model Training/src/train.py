import os
import re
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    cohen_kappa_score, mean_squared_error
)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import SMOTE

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors, pagesizes
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ----------------------------
# Setup
# ----------------------------

nltk.download("stopwords")
nltk.download("wordnet")

DATA_PATH = "../data/fake_job_postings.csv"
MODEL_DIR = "models"

TARGET_COLUMN = "fraudulent"

THRESHOLD = 0.30

TEXT_COLUMNS = [
    "title",
    "description",
    "requirements",
    "benefits",
    "company_profile"
]

DROP_COLUMNS = [
    "job_id",
    "has_questions",
    "industry",
    "function"
]

os.makedirs(MODEL_DIR, exist_ok=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# ----------------------------
# Text Cleaning
# ----------------------------

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"[^\w\s]", " ", text)

    words = [
        lemmatizer.lemmatize(w)
        for w in text.split()
        if w not in stop_words
    ]

    return " ".join(words)


# ----------------------------
# Load Dataset
# ----------------------------

def load_data():

    df = pd.read_csv(DATA_PATH)

    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])

    for col in TEXT_COLUMNS:

        if col in df.columns:

            df[col] = df[col].fillna("")

    df["combined_text"] = (

        df["title"] + " " +
        df["description"] + " " +
        df["requirements"] + " " +
        df["benefits"] + " " +
        df["company_profile"]

    ).apply(clean_text)

    categorical_cols = [

        c for c in df.select_dtypes(include=["object"]).columns
        if c not in TEXT_COLUMNS + ["combined_text"]

    ]

    for col in categorical_cols:

        df[col] = df[col].fillna("missing")

    X = df.drop(columns=[TARGET_COLUMN])

    y = df[TARGET_COLUMN]

    return X, y, categorical_cols


# ----------------------------
# Evaluation
# ----------------------------

def evaluate_model(y_true, y_pred, y_proba):

    return {

        "Accuracy": accuracy_score(y_true, y_pred),

        "Precision": precision_score(y_true, y_pred),

        "Recall": recall_score(y_true, y_pred),

        "F1 Score": f1_score(y_true, y_pred),

        "ROC-AUC": roc_auc_score(y_true, y_proba),

        "Cohen Kappa": cohen_kappa_score(y_true, y_pred),

        "MSE": mean_squared_error(y_true, y_pred),

        "Confusion Matrix": confusion_matrix(y_true, y_pred).tolist(),

        "Classification Report": classification_report(y_true, y_pred)

    }


# ----------------------------
# CSV Comparison
# ----------------------------

def generate_comparison_table(results):

    rows = []

    for name, m in results.items():

        rows.append({

            "Model": name,

            "Accuracy": m["Accuracy"],

            "Precision": m["Precision"],

            "Recall": m["Recall"],

            "F1 Score": m["F1 Score"],

            "ROC-AUC": m["ROC-AUC"],

            "Cohen Kappa": m["Cohen Kappa"],

            "MSE": m["MSE"]

        })

    df = pd.DataFrame(rows)

    df = df.sort_values(

        by=["Recall", "F1 Score"],

        ascending=False

    ).reset_index(drop=True)

    df.insert(0, "Rank", df.index + 1)

    df.to_csv(f"{MODEL_DIR}/model_comparison.csv", index=False)

    return df


# ----------------------------
# PDF Report
# ----------------------------

def generate_pdf_report(comparison_df, detailed_results):

    pdf_path = f"{MODEL_DIR}/model_performance_report.pdf"

    doc = SimpleDocTemplate(pdf_path, pagesize=pagesizes.A4)

    styles = getSampleStyleSheet()

    elements = []

    elements.append(Paragraph(
        "Fraud Job Detection – Model Performance Report",
        styles["Heading1"]
    ))

    elements.append(Spacer(1, 0.5 * inch))

    elements.append(Paragraph(
        "Model Comparison Summary",
        styles["Heading2"]
    ))

    elements.append(Spacer(1, 0.2 * inch))

    table_data = [comparison_df.columns.tolist()] + comparison_df.round(4).values.tolist()

    table = Table(table_data, repeatRows=1)

    table.setStyle(TableStyle([

        ("BACKGROUND", (0,0), (-1,0), colors.grey),

        ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke),

        ("GRID", (0,0), (-1,-1), 0.5, colors.black),

        ("ALIGN", (1,1), (-1,-1), "CENTER"),

        ("FONTSIZE", (0,0), (-1,-1), 8)

    ]))

    elements.append(table)

    elements.append(Spacer(1, 0.5 * inch))

    for name, m in detailed_results.items():

        elements.append(Paragraph(name, styles["Heading3"]))

        elements.append(Spacer(1, 0.1 * inch))

        elements.append(Paragraph(

            f"Accuracy: {m['Accuracy']:.4f}<br/>"
            f"Precision: {m['Precision']:.4f}<br/>"
            f"Recall: {m['Recall']:.4f}<br/>"
            f"F1 Score: {m['F1 Score']:.4f}<br/>"
            f"ROC-AUC: {m['ROC-AUC']:.4f}<br/>"
            f"Cohen Kappa: {m['Cohen Kappa']:.4f}<br/>"
            f"MSE: {m['MSE']:.4f}",

            styles["Normal"]

        ))

        elements.append(Spacer(1, 0.5 * inch))

        cm = m["Confusion Matrix"]

        cm_table = Table([

            ["", "Predicted Real", "Predicted Fraud"],

            ["Actual Real", cm[0][0], cm[0][1]],

            ["Actual Fraud", cm[1][0], cm[1][1]]

        ])

        cm_table.setStyle(TableStyle([

            ("GRID", (0,0), (-1,-1), 0.5, colors.black),

            ("ALIGN", (1,1), (-1,-1), "CENTER")

        ]))

        elements.append(cm_table)

        elements.append(Spacer(1, 0.5 * inch))

    doc.build(elements)

    print(f"\nPDF report saved: {pdf_path}")


# ----------------------------
# Training
# ----------------------------

def train():

    print(f"\nUsing classification threshold: {THRESHOLD}")

    X, y, categorical_cols = load_data()

    X_train, X_test, y_train, y_test = train_test_split(

        X, y,

        test_size=0.2,

        random_state=42,

        stratify=y

    )

    preprocessor = ColumnTransformer([

        ("text", TfidfVectorizer(

            max_features=15000,

            ngram_range=(1,3),

            min_df=5,

            max_df=0.9,

            sublinear_tf=True

        ), "combined_text"),

        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)

    ])

    X_train_proc = preprocessor.fit_transform(X_train)

    X_test_proc = preprocessor.transform(X_test)

    smote = SMOTE(random_state=42)

    X_train_res, y_train_res = smote.fit_resample(X_train_proc, y_train)

    models = {

        "RandomForest": RandomForestClassifier(

            n_estimators=600,

            min_samples_leaf=2,

            max_features="sqrt",

            random_state=42,

            n_jobs=-1

        ),

        "KNN": KNeighborsClassifier(

            n_neighbors=9,

            weights="distance"

        ),

        "LogisticRegression": LogisticRegression(

            max_iter=3000,

            n_jobs=-1

        )

    }

    detailed_results = {}

    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(X_train_res, y_train_res)

        y_proba = model.predict_proba(X_test_proc)[:,1]

        y_pred = (y_proba >= THRESHOLD).astype(int)

        metrics = evaluate_model(y_test, y_pred, y_proba)

        detailed_results[name] = metrics

        print(classification_report(y_test, y_pred))

        joblib.dump(

            {"preprocessor": preprocessor, "model": model},

            f"{MODEL_DIR}/{name.lower()}_model.pkl"

        )

    comparison_df = generate_comparison_table(detailed_results)

    generate_pdf_report(comparison_df, detailed_results)

    print("\nTraining complete. Models, CSV, and PDF report saved.")

    return comparison_df


if __name__ == "__main__":

    train()