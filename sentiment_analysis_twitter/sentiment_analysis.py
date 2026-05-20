# =========================
# Sentiment Analysis Project
# =========================

import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split, KFold, ShuffleSplit, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier


# -------------------------
# 1. Data Cleaning
# -------------------------
def clean_tweet(text):
    text = re.sub(r"http\S+|www\S+", '', text)
    text = text.lower()
    return text


# -------------------------
# 2. Load Dataset
# -------------------------
df = pd.read_csv(
    'training.1600000.processed.noemoticon.csv',
    encoding='ISO-8859-1',
    header=None
)

df.columns = ['target', 'id', 'date', 'flag', 'user', 'text']

# Convert labels (4 -> 1)
df['target'] = df['target'].replace(4, 1)

X = df['text'].apply(clean_tweet).values
y = df['target'].values


# -------------------------
# 3. TF-IDF Vectorization
# -------------------------
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X_tfidf = vectorizer.fit_transform(X)


# -------------------------
# 4. Train/Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)


# -------------------------
# 5. Logistic Regression Model
# -------------------------
base_model = LogisticRegression(max_iter=1000)


# -------------------------
# 6. Cross Validation (ShuffleSplit)
# -------------------------
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
scores = cross_val_score(base_model, X_train, y_train, cv=cv)

print("Cross-validation scores:", scores)
print("Average score:", np.mean(scores))


# -------------------------
# 7. K-Fold Validation
# -------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in kf.split(X_train):
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    model = base_model.fit(X_tr, y_tr)
    preds = model.predict(X_val)

    print("Fold accuracy:", accuracy_score(y_val, preds))


# -------------------------
# 8. Simple Ensemble (AdaBoost style)
# -------------------------
n_estimators = 5
weights = np.ones(X_train.shape[0]) / X_train.shape[0]

models = []
alphas = []

for i in range(n_estimators):
    model = DecisionTreeClassifier(max_depth=1)
    model.fit(X_train, y_train, sample_weight=weights)

    preds = model.predict(X_train)

    error = np.sum(weights * (preds != y_train)) / np.sum(weights)

    if error > 0.5:
        break

    alpha = 0.5 * np.log((1 - error) / (error + 1e-10))

    models.append(model)
    alphas.append(alpha)

    weights *= np.exp(alpha * (preds != y_train))
    weights /= np.sum(weights)


# -------------------------
# 9. Final Prediction
# -------------------------
final_pred = np.zeros(X_test.shape[0])

for alpha, model in zip(alphas, models):
    final_pred += alpha * model.predict(X_test)

final_pred = np.sign(final_pred)

print("Ensemble Accuracy:", accuracy_score(y_test, final_pred))