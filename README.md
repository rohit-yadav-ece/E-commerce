# E-Commerce Conversion Prediction
### Summer Analytics 2026 — Mini Hackathon | IIT Guwahati C&A Club

Predicting whether a user on an e-commerce platform will make a purchase or not, based on their behavioural and demographic data.

---

##  Problem Statement

Binary classification — given user-level features like pages viewed, time on site, income, and device type, predict:
- `1` → User converted (made a purchase)
- `0` → User did not convert

**Evaluation Metric:** F1 Score

---

##  Project Structure

```
CA_Hackathon/
│
├── data/
│   ├── train.csv              # Training data with labels
│   ├── public_test.csv        # Validation data with labels
│   ├── private_test.csv       # Test data without labels (submission target)
│   └── sample_submission.csv  # Submission format reference
│
├── notebook.ipynb             # Full pipeline notebook
├── submission.csv             # Final predictions
└── report.pdf                 # One-page methodology report
```

---

##  Approach

**1. EDA**
Explored missing values, class imbalance (~69/31 split), outliers, and feature distributions.

**2. Preprocessing**
- Iterative Imputer for missing values in Age, Income, Time_On_Site
- Capped Time_On_Site outlier at 99th percentile
- Label Encoding for Device_Type and Traffic_Source

**3. Feature Engineering**
Created 8 new features including:
- `Products_per_Page`, `Time_per_Page`, `Time_per_Product`
- `High_Engagement` flag, `Activity_Score`, `Age_Group`

**4. Modelling**
Soft voting ensemble of XGBoost + LightGBM with threshold tuning to maximise F1.

---

##  Results

| Model | F1 Score |
|---|---|
| Baseline — Random Forest | 0.3084 |
| XGBoost + Feature Engineering + Tuning | 0.5381 |
| XGBoost + Iterative Imputer | 0.5407 |
| LightGBM alone | 0.5442 |
| **XGBoost + LightGBM Ensemble (Final)** | **0.5454** |

---

##  Libraries Used

```
pandas • NumPy • scikit-learn • XGBoost • LightGBM • Matplotlib
```

---

##  How to Run

1. Clone the repo and place the dataset files inside `data/`
2. Open `notebook.ipynb` in Jupyter Notebook
3. Run all cells top to bottom — `submission.csv` will be generated automatically
