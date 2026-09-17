# Smart City Traffic Intelligence: From Data Analytics to AI-Powered Mobility - Capstone Project

A complete data science pipeline analyzing traffic patterns using SQL, Python, and machine learning with MLOps, deployment, and monitoring.

## Project Structure

```
Capstone/
├── Part1_Data_analytics/          # BI & Statistical Analysis
│   ├── SQL/                       # Analysis queries
│   ├── Capstone Part 1.pbix       # Power BI dashboard
│   ├── statistics/			       #Probability & Conditional Probability
│   └── insights_report.pdf                   
│
├── Part2_Python/                  # Data Engineering Pipeline
│   ├── pipeline.py                # Data cleaning
│   ├── feature_engineering.py     # Feature creation
│   ├── visualisation.py           # Generate charts
│   ├── traffic_analytics_app.py   # CLI interface
│   └── pipeline.log               # Consolidated logs
│
└── Part3_Machine_Learning/        # Machine Learning & AI
    ├── data/                      # Preprocessed datasets
    ├── models/                    # Trained model artifacts
    ├── notebooks/                 # ML & Analysis notebooks
    │   ├── 01_data_preprocessing.ipynb          # ⭐ RUN THIS FIRST
    │   ├── 02_task1_classification.ipynb        # Classification models
    │   ├── 03_task1_regression.ipynb            # Regression models
    │   ├── 04_task2_clustering.ipynb            # K-means clustering
    │   ├── 05_task2_association_rules.ipynb     # Association rules
    │   ├── 06_task3_deep_learning.ipynb         # Neural networks + SHAP
    │   └── 07_task4_quantization.ipynb          # Model optimization
    ├── deployment/                # FastAPI deployment
    │   └── deployment_api.py      # Production API server
    ├── monitoring/                # MLOps & Monitoring
    │   └── 10_task6_deployment_monitoring.ipynb # Drift detection & alerting
    ├── recommendation_system/     # Travel recommendations
    │   └── 08_task5_recommendation_system.ipynb # Recommendation engine
    ├── mlflow/                    # Experiment tracking
    │   └── 09_task6_mlops_simulation.ipynb      # Model versioning & MLflow
    ├── mlflow.db                  # SQLite experiment database
    └── responsible_ai_report.pdf  # Bias, fairness & governance analysis
```

## Quick Start

### Part 2: Data Engineering (Python)

**Important:** All scripts must be run in order and generate output to `pipeline.log`

```bash
cd "Capstone/Part2_Python"

python pipeline.py                  # Clean data
python feature_engineering.py       # Create features
python visualisation.py             # Generate charts
python traffic_analytics_app.py     # Interactive CLI
```

### Part 3: Machine Learning (Jupyter)

⚠️ **MUST RUN FIRST:** Execute `01_data_preprocessing.ipynb` before running any model notebooks

```bash
cd "Capstone/Part3_Machine_Learning"
jupyter notebook

# Run in order:
# 1. 01_data_preprocessing.ipynb         ← Required for all models
# 2. 02_task1_classification.ipynb
# 3. 03_task1_regression.ipynb
# 4. 04_task2_clustering.ipynb
# 5. 05_task2_association_rules.ipynb
# 6. 06_task3_deep_learning.ipynb
# 7. 07_task4_quantization.ipynb
# 8. 08_task5_recommendation_system.ipynb
# 9. 09_task6_mlops_simulation.ipynb
# 10. 10_task6_deployment_monitoring.ipynb
```

**View MLflow Experiments:**
```bash
cd "Capstone/Part3_Machine_Learning"
mlflow ui --backend-store-uri "sqlite:///mlflow.db" --host 0.0.0.0 --port 8080
# Open http://localhost:8080
```

**Test FastAPI Deployment:**
```bash
cd "Capstone/Part3_Machine_Learning/deployment"
python deployment_api.py
# API runs on http://localhost:8000
# Swagger docs: http://localhost:8000/docs
```

## What Each Part Does

**Part 1 (BI & Analysis)**
- Power BI dashboard for traffic trends
- SQL analysis of patterns and correlations
- Statistical insights and probability analysis

**Part 2 (Data Engineering)**
- Clean 48,187 traffic records
- Engineer 35 ML-ready features
- Generate 6 visualization charts
- Provide CLI interface for data queries

**Part 3 (Machine Learning & AI)**
- **Task 1:** Classification & Regression models
- **Task 2:** K-means clustering & Association rules mining
- **Task 3:** Deep learning with SHAP explainability
- **Task 4:** Model quantization & experiment tracking (MLflow)
- **Task 5:** Travel time recommendation engine
- **Task 6:** MLOps simulation with deployment, monitoring & alerting
- **Task 7:** Responsible AI report (bias, fairness, governance)

## Part 3 Components

### Notebooks (ML & Analysis)
- `01_data_preprocessing.ipynb` - Feature scaling and train/test split
- `02_task1_classification.ipynb` - Traffic condition classification
- `03_task1_regression.ipynb` - Traffic volume forecasting
- `04_task2_clustering.ipynb` - Traffic pattern segmentation (K-means)
- `05_task2_association_rules.ipynb` - Pattern discovery (Apriori)
- `06_task3_deep_learning.ipynb` - Neural network with SHAP explanations
- `07_task4_quantization.ipynb` - Model optimization for edge deployment

### Models
- `models/` - Trained model artifacts

### Recommendation System
- `08_task5_recommendation_system.ipynb` - Optimal travel time suggestions

### MLOps & Deployment
- `09_task6_mlops_simulation.ipynb` - Model versioning & experiment tracking

### Reports & Tracking
- `10_task6_deployment_monitoring.ipynb` - Drift detection & alerting dashboard
- `deployment_api.py` - FastAPI production server
- `responsible_ai_report.pdf` - Bias/fairness analysis & governance framework

## Data Summary

- **Dataset:** Traffic Volume
- **Time Range:** Oct 2012 - Sep 2018
- **Records:** 48,187 hourly observations
- **Target:** Traffic volume prediction

**Last Updated:** September 17, 2026
