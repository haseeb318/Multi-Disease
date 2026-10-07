"""
Train the Diabetes SVM model and save it to app/models/
Run this script once from the project root: python training/train_diabetes.py
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# ── Load dataset ─────────────────────────────────────────────────────────────
data_path = os.path.join(os.path.dirname(__file__), 'data', 'diabetes.csv')
df = pd.read_csv(data_path)

print(f"Dataset shape: {df.shape}")
print(df.describe())

# ── Split features / labels ──────────────────────────────────────────────────
X = df.drop(columns='Outcome', axis=1)
Y = df['Outcome']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)
print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

# ── Train ────────────────────────────────────────────────────────────────────
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

train_acc = accuracy_score(Y_train, classifier.predict(X_train))
test_acc  = accuracy_score(Y_test,  classifier.predict(X_test))
print(f"Training accuracy : {train_acc:.4f}")
print(f"Test accuracy     : {test_acc:.4f}")

# ── Save model ───────────────────────────────────────────────────────────────
out_path = os.path.join(os.path.dirname(__file__), '..', 'app', 'models', 'diabetes_model.pkl')
os.makedirs(os.path.dirname(out_path), exist_ok=True)

with open(out_path, 'wb') as f:
    pickle.dump(classifier, f)

print(f"Model saved to: {os.path.abspath(out_path)}")
