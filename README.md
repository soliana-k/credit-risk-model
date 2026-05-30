## Credit Risk Model
### Credit Scoring Business Understanding
#### 1. How does the Basel II Accord's emphasis on risk measurement influence the need for an interpretable and well-documented model?
- The Basel II Accord shifts the focus from model performance to model defensibility. Because credit risk models directly impact financial stability and borrower access, they are subject to regulatory scrutiny that demands full transparency. This mandates an interpretable architecture where every input has a traceable impact on the output and comprehensive documentation that serves as an audit trail for regulators. Consequently,  the modeling choices prioritize statistical interpretability over raw predictive power to ensure compliance, auditability, and ethical fairness.

#### 2. Without a direct "default" label, why is a proxy variable necessary, and what business risks does proxy-based prediction introduce?
- Because the dataset lacks historical loan default records, I employed an RFM-based clustering approach to derive a high-risk proxy variable. While this allows me to initiate model training, it introduces an inherent approximation risk. I acknowledge that behavioral disengagement is a categorical proxy not a financial certainty and thus, the model predicts the probability of behavioral risk rather than contractual default. To mitigate business risk, this model must be treated as one of several inputs in a human-in-the-loop loan origination process, and its performance must be validated against real-world repayment data as soon as the buy-now-pay-later service begins accumulating credit outcomes.

#### 3. What are the key trade-offs between a simple, interpretable model (e.g., Logistic Regression with WoE) and a high-performance model (e.g., Gradient Boosting) in a regulated financial context?
- In choosing between a simple, interpretable model (Logistic Regression with WoE) and a high-performance model (Gradient Boosting), there is a fundamental trade-off between predictive precision and regulatory transparency. While Gradient Boosting offers superior performance by capturing complex, non-linear data patterns, its black-box nature poses significant hurdles for the rigorous, documentable audit trails required by Basel II standards. Conversely, Logistic Regression provides a transparent, monotonic relationship between inputs and outcomes, aligning with the regulatory need for explainable AI. the strategy favors an initial focus on interpretable models to satisfy compliance requirements, with high-performance models utilized as secondary benchmarks to measure the accuracy gap and justify future model complexity.

--- 

# EDA Pipeline for Credit Risk Modeling

## Project Overview

This project implements a **clean, reusable, and well-architectured Exploratory Data Analysis (EDA) pipeline** for a financial transaction dataset used in **credit risk and fraud detection modeling**.

The pipeline was built to fully satisfy the given assignment requirements while maintaining production-grade code quality, reproducibility, and scalability.

---

## Task Requirements & Coverage

The assignment required the following:

### Completed Tasks:

1. **Overview of the Data** — Shape, columns, data types, memory usage
2. **Summary Statistics** — Central tendency, dispersion, skewness, kurtosis + automated insights
3. **Distribution of Numerical Features** — Histograms and count plots
4. **Distribution of Categorical Features** — Frequency analysis with top-N handling
5. **Correlation Analysis** — Pearson correlation matrix + heatmap
6. **Identifying Missing Values** — Detection with smart imputation advice
7. **Outlier Detection** — IQR method + **Box plots** for visual identification

**All 7 requirements are fully implemented.**

---

## Architecture Highlights

- **Pipeline-as-a-Class** design (`EDAPipeline`)
- Uses **inner classes** (`_Visualizer`, `_Reporter`) to avoid God Class anti-pattern
- Highly **configurable** through `EDAConfig` dataclass
- **Reproducible** — Timestamped output folders + saved `config.json`
- Method chaining support (`load_data().run()`)
- Separation of concerns (Analysis, Visualization, Reporting)
- Works well in **Jupyter Notebooks** (plots displayed inline + saved)

---

## Key Features

- Automatic data sampling for large datasets
- Smart distribution insights (skewness & kurtosis interpretation)
- Professional text report with outliers and correlation summary
- All plots saved in organized `plots/` folder
- Box plots for every numerical feature
- Clean logging throughout the pipeline
- Fraud imbalance awareness built into insights

---

## 📁 Project Structure

```bash
credit-risk-model/
├── .github/workflows/ci.yml          
├── data/                             
│   ├── raw/                          
│   └── processed/                    
├── notebooks/
│   └── eda.ipynb                     
├── src/
│   ├── __init__.py
│   ├── eda_pipeline.py               
│   ├── data_processing.py           #future
│   └── api/                   
│       └──  main.py       
├── tests/
│   └── test.py
├── outputs/                         
├── Dockerfile           # future
├── docker-compose.yml   # future
├── requirements.txt
├── .gitignore
└── README.md
```

---

##  How to Use

```python
from eda_pipeline import EDAPipeline, EDAConfig

# Load The Dataframe beforehand 
# Configuration
config = EDAConfig(
    sample_size=None,      # Set number for large data, leave None for all
    save_plots=True,
    include_plots=True
)

# Run pipeline
pipeline = EDAPipeline(config)

results = (pipeline
           .load_data(df)        # df is the pandas DataFrame
           .run())

print("EDA completed successfully!")

```

## Dataset Insights (Fraud Detection Context)

- **Highly Imbalanced Target**: `FraudResult` has only **0.2%** positive cases.
- Extreme skewness in `Amount` and `Value`
- High outlier rates in monetary features
- `CountryCode` is constant → should be dropped
- Strong correlation between `Amount` and `Value`

---

## Technologies Used

- Python 3
- pandas, matplotlib, seaborn
- dataclasses, pathlib, logging
- Designed for Jupyter Notebook environment


---

**Author:** Kalkidan Kassahun  
**Date:** May 30, 2026 - first version 
**Context:** Credit Risk Modeling


