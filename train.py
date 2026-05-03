import csv
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

CSV_PATH   = "gesture_data.csv"
MODEL_PATH = "gesture_model.pkl"

# Veri yükle
X, y = [], []
with open(CSV_PATH) as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        if row:
            y.append(row[0])
            X.append([float(v) for v in row[1:]])

X = np.array(X)
y = np.array(y)

le = LabelEncoder()
y_enc = le.fit_transform(y)

print(f"Veri: {len(X)} örnek, {len(le.classes_)} sınıf")
print(f"Sınıflar: {list(le.classes_)}\n")

# Cross-validation
model = RandomForestClassifier(n_estimators=50, max_depth=None, random_state=42, n_jobs=-1)
scores = cross_val_score(model, X, y_enc, cv=5, scoring="accuracy")
print(f"5-Fold CV doğruluk: {scores.mean():.3f} ± {scores.std():.3f}")

# Tam veriyle eğit
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
model = RandomForestClassifier(n_estimators=50, max_depth=None, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nTest seti raporu:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Final model tüm veriyle
model.fit(X, y_enc)
joblib.dump({"model": model, "encoder": le}, MODEL_PATH)
print(f"Model kaydedildi: {MODEL_PATH}")
