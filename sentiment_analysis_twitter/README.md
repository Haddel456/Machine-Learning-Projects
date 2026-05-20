# Twitter Sentiment Analysis using Machine Learning

## Overview
This project performs sentiment analysis on a large Twitter dataset using Machine Learning techniques.  
The goal is to classify tweets as **positive (1)** or **negative (0)** using Natural Language Processing (NLP) and different machine learning models.

The project includes:
- Text preprocessing
- TF-IDF feature extraction
- Logistic Regression classification
- Cross-validation techniques
- Ensemble learning (AdaBoost-style implementation)

---

## Dataset
The dataset used is:

`training.1600000.processed.noemoticon.csv`

It contains 1.6 million tweets labeled as:
- 0 → Negative sentiment  
- 4 → Positive sentiment (converted to 1 in preprocessing)

---

## Data Preprocessing
- Removal of URLs from tweets
- Lowercasing text
- Label conversion (4 → 1)
- Text vectorization using TF-IDF

---

## Feature Extraction
Text is converted into numerical features using:

TF-IDF Vectorizer:
- Stop words removed (English)
- Maximum features: 10,000

---

## Machine Learning Models

### 1. Logistic Regression
A baseline classifier is trained using:
- Train/Test split (80/20 and 50/50 experiments)
- Cross-validation using:
  - ShuffleSplit
  - K-Fold Cross Validation

---

### 2. Cross Validation Techniques
The project evaluates model performance using:
- ShuffleSplit Cross Validation
- K-Fold Cross Validation (5 folds)

This helps measure model stability and generalization.

---

### 3. Ensemble Learning (AdaBoost-style Implementation)
A custom ensemble model is implemented using:
- Decision Tree Stumps (max_depth = 1)
- Sample weighting
- Alpha weight calculation

Final prediction:

:contentReference[oaicite:0]{index=0}

---

## Evaluation Metrics
- Accuracy Score
- Cross-validation average accuracy
- Ensemble model accuracy

---

## Results
The project compares:
- Logistic Regression performance
- Different train/test splits (80/20 and 50/50)
- Cross-validation stability
- Ensemble model performance

Key observation:
- TF-IDF + Logistic Regression performs strongly for text classification
- Ensemble improves robustness using weak learners

---

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Regex (re)

---



## Project Structure

```bash
Breast-Cancer-ML-Analysis/
│
├── sentiment_analysis.ipynb
├── sentiment_analysis.py
│
├── requirements.txt
├── README.md

```

## How to Run

### 1. Clone the Repository
```bash
    git clone https://github.com/your-username/
```


### 2. Install dependencies
```bash
    pip install -r requirements.txt
```

### 3. Run the Project
```bash
    sentiment_analysis.py
```


## Run the Notebook
```bash
    jupyter notebook
```

## Then open:
```bash
cancer_analysis.ipynb