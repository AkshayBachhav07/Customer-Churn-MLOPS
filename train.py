import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

np.random.seed(42)
n_samples = 1500

tenure_months = np.random.randint(1, 72, size=n_samples)
monthly_charges = np.random.uniform(20.0, 120.0, size=n_samples)
contract_type = np.random.choice([0, 1, 2], size=n_samples) # 0: Month-to-month, 1: 1-Year, 2: 2-Year
internet_service = np.random.choice([0, 1, 2], size=n_samples) # 0: DSL, 1: Fiber optic, 2: None
payment_method = np.random.choice([0, 1], size=n_samples) # 0: Manual/Check, 1: Auto-pay/Card
support_tickets_count = np.random.randint(0, 7, size=n_samples)

data = pd.DataFrame({
    "tenure_months": tenure_months,
    "monthly_charges": monthly_charges,
    "contract_type": contract_type,
    "internet_service": internet_service,
    "payment_method": payment_method,
    "support_tickets_count": support_tickets_count
})

logits = (
    -0.05 * data["tenure_months"]
    + 0.03 * data["monthly_charges"]
    - 0.9 * data["contract_type"]
    + 0.5 * data["support_tickets_count"]
    - 0.4 * data["payment_method"]
    - 0.5
)
prob = 1.0 / (1.0 + np.exp(-logits))
data["churn"] = (prob > 0.45).astype(int)

data.to_csv("data/dataset.csv", index=False)
print("[1/3] Dataset saved to data/dataset.csv")

X = data.drop(columns=["churn"])
y = data["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(n_estimators=120, max_depth=7, random_state=42))
])

pipeline.fit(X_train, y_train)
acc = accuracy_score(y_test, pipeline.predict(X_test))
print(f"[2/3] Model trained with accuracy: {acc * 100:.2f}%")
print(classification_report(y_test, pipeline.predict(X_test)))

joblib.dump(pipeline, "models/model.pkl")
print("[3/3] Exported pipeline artifact to models/model.pkl")
