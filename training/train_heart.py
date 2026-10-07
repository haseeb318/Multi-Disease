"""
Train the Heart Disease Logistic Regression model and save it to app/models/
Run this script once from the project root: python training/train_heart.py
"""
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ── Load dataset ─────────────────────────────────────────────────────────────
data_path = os.path.join(os.path.dirname(__file__), 'data', 'heart.csv')
df = pd.read_csv(data_path)

print(f"Dataset shape: {df.shape}")
print(df['target'].value_counts())

# ── Split features / labels ──────────────────────────────────────────────────
X = df.drop(columns='target', axis=1)
Y = df['target']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)
print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

# ── Train ────────────────────────────────────────────────────────────────────
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

train_acc = accuracy_score(Y_train, model.predict(X_train))
test_acc  = accuracy_score(Y_test,  model.predict(X_test))
print(f"Training accuracy : {train_acc:.4f}")
print(f"Test accuracy     : {test_acc:.4f}")

# ── Save model ───────────────────────────────────────────────────────────────
out_path = os.path.join(os.path.dirname(__file__), '..', 'app', 'models', 'heart_disease_model.pkl')
os.makedirs(os.path.dirname(out_path), exist_ok=True)

with open(out_path, 'wb') as f:
    pickle.dump(model, f)

print(f"Model saved to: {os.path.abspath(out_path)}")
