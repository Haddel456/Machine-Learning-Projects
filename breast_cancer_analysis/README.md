# Breast Cancer Machine Learning Analysis

## Overview
This project focuses on applying Machine Learning optimization techniques to a breast cancer dataset.  
The project includes data preprocessing, feature normalization, dimensionality reduction using Singular Value Decomposition (SVD), and performance comparison between different Gradient Descent approaches.

The main goal is to analyze how optimization algorithms and dimensionality reduction affect:
- Model convergence
- Cost function minimization
- Runtime performance

---

## Features
- Data preprocessing using Pandas and NumPy
- Feature normalization (Standardization)
- Stochastic Gradient Descent (SGD)
- Mini-Batch Gradient Descent
- Dimensionality Reduction using SVD
- Runtime comparison before and after dimensionality reduction
- Visualization of learning rates and cost convergence using Matplotlib

---

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- SciPy

---

## Dataset
The dataset used in this project is:

`cancer_data.csv`

The dataset contains multiple numerical features related to breast cancer diagnosis.

---

## Machine Learning Techniques

### 1. Data Normalization
Features are normalized using standardization to improve gradient descent convergence and training stability.

### 2. Stochastic Gradient Descent (SGD)
The project applies SGD with different learning rates:
- 0.1
- 0.01
- 0.001

The Mean Squared Error (MSE) cost function is tracked over iterations to analyze convergence behavior.

### 3. Mini-Batch Gradient Descent
Mini-Batch Gradient Descent is implemented to improve training efficiency by updating parameters using small batches instead of the entire dataset.

Batch size used:
- 32

### 4. Dimensionality Reduction using SVD
Singular Value Decomposition (SVD) is used to reduce feature dimensions while preserving important information.

The dataset is reduced to:
- 3 principal components

This helps improve runtime efficiency and reduces computational complexity.

---

## Results
The project compares:
- Different learning rates
- SGD vs Mini-Batch Gradient Descent
- Full-dimensional data vs Reduced-dimensional data

The results demonstrate:
- Faster convergence with proper learning rates
- Reduced runtime after dimensionality reduction
- Improved computational efficiency using SVD

---

## Visualizations
The project includes:
- Cost vs Iterations plots
- Learning rate comparison graphs
- Runtime comparison before and after dimensionality reduction

---

## Project Structure

```bash
Breast-Cancer-ML-Analysis/
│
├── cancer_data.csv
│
├── cancer_analysis.ipynb
├── cancer_analysis.py
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
    python cancer_analysis.py
```


## Run the Notebook
```bash
    jupyter notebook
```

## Then open:
```bash
cancer_analysis.ipynb
```