# Metro Interstate Traffic Volume - Comprehensive AI/ML Capstone Project

A complete data science and machine learning capstone project analyzing Metro Interstate traffic patterns using advanced analytics, deep learning, and practical recommendation systems.

**Project Repository:** https://github.com/your-username/capstone

---

## 📋 Project Overview

This capstone project demonstrates a complete end-to-end data science workflow across three phases:

### Phase 1: Data Analysis & BI (SQL, Power BI)
- Traffic trend analysis and forecasting
- Weather-traffic correlation studies
- Probability and statistical insights

### Phase 2: Data Engineering & Pipeline (Python)
- Robust data cleaning and validation
- Feature engineering (35+ features)
- Traffic visualization and analytics
- CLI interface for querying patterns

### Phase 3: Machine Learning & AI (Python, TensorFlow, MLflow)
- Supervised learning (classification, regression)
- Unsupervised learning (clustering, association rules)
- Deep learning with neural networks
- Model optimization (quantization) and experiment tracking (MLflow)
- Practical recommendation system for travel planning

---

## 📁 Project Structure

```
Capstone/
├── README.md                                    # This file
├── Metro_Interstate_Traffic_Volume.csv          # Raw dataset (48,204 rows, 9 columns)
├── Traffic_Vol.db                               # SQL analysis database
│
├── Part 1/                                      # BI & Statistical Analysis
│   ├── Capstone Part 1.pbix                     # Power BI dashboard
│   ├── Insights_Report.docx                     # Statistical findings
│   ├── Statistics.docx                          # Detailed analysis
│   └── SQL/                                     # SQL scripts for analysis
│       ├── Part 1.2 Analyse annual traffic trends.sql
│       ├── Part 1.3 Analyse temperature around holidays.sql
│       ├── Part 2.1 Traffic volume statistics.sql
│       ├── Part 2.2 Correlation Analysis.sql
│       ├── Part 3.1 Basic probability.sql
│       └── Part 3.2 Conditional Probability.sql
│
├── Part 2/                                      # Data Engineering Pipeline
│   ├── README.md                                # Part 2 documentation
│   ├── pipeline.py                              # Task 1: Data cleaning
│   ├── feature_engineering.py                   # Task 2: Feature creation
│   ├── visualisation.py                         # Task 3: Traffic visualizations
│   ├── traffic_analytics_app.py                 # Task 4: CLI analytics app
│   ├── pipeline.log                             # Consolidated execution logs
│   ├── Metro_Interstate_Traffic_Volume_cleaned.csv
│   ├── Metro_Interstate_Traffic_Volume_features.csv
│   └── visualisations/                          # Generated charts
│       ├── 01_traffic_by_hour.png
│       ├── 02_weekday_vs_weekend.png
│       ├── 03_temperature_vs_traffic.png
│       ├── 04_traffic_distribution.png
│       ├── 05_hourly_heatmap.png
│       └── 06_weather_impact.png
│
└── Part 3/                                      # Machine Learning & AI
    ├── 01_data_preprocessing.ipynb              # Data preparation (preprocessing)
    ├── 02_task1_classification.ipynb            # Task 1a: Classification models
    ├── 03_task1_regression.ipynb                # Task 1b: Regression models
    ├── 04_task2_clustering.ipynb                # Task 2a: K-means clustering
    ├── 05_task2_association_rules.ipynb         # Task 2b: Association rule mining
    ├── 06_task3_deep_learning.ipynb             # Task 3: Deep learning with SHAP
    ├── 07_task4_quantization_mlflow.ipynb       # Task 4: Model optimization & MLflow
    ├── 08_task5_recommendation_system.ipynb     # Task 5: Travel recommendations
    ├── Metro_Interstate_Traffic_Volume_part3_preprocessed.csv
    ├── model_original.h5                        # Original neural network
    ├── model_quantized.tflite                   # Quantized TFLite model
    ├── mlruns/                                  # MLflow experiment tracking
    │   └── 1/                                   # Experiment 1: traffic_prediction_optimization
    │       ├── models/                          # Logged model artifacts
    │       └── runs/                            # Experiment runs and metrics
    └── visualizations/                          # Generated charts from ML tasks
        ├── 04_shap_feature_importance.png
        ├── 06_training_history.png
        ├── 07_quantization_comparison.png
        └── 08_traffic_patterns.png
```

---

## 🚀 Quick Start

### Prerequisites

**Python 3.9+** with required libraries:

```bash
# Core data science
pip install pandas numpy scikit-learn matplotlib seaborn

# Deep learning
pip install tensorflow keras shap

# Experiment tracking
pip install mlflow

# Data processing
pip install mlxtend  # For association rules
```

### Run All Pipelines

#### Part 2: Data Engineering (Python scripts)

```bash
cd "Capstone/Part 2"

# 1. Clean data
python pipeline.py

# 2. Engineer features
python feature_engineering.py

# 3. Generate visualizations
python visualisation.py

# 4. Interactive CLI app
python traffic_analytics_app.py        # Interactive menu
python traffic_analytics_app.py best-times  # Command-line
```

#### Part 3: Machine Learning (Jupyter notebooks)

```bash
cd "Capstone/Part 3"

# Launch Jupyter
jupyter notebook

# Run notebooks in order:
# 1. 01_data_preprocessing.ipynb         (Data prep for ML)
# 2. 02_task1_classification.ipynb       (Classification models)
# 3. 03_task1_regression.ipynb           (Regression models)
# 4. 04_task2_clustering.ipynb           (K-means clustering)
# 5. 05_task2_association_rules.ipynb    (Association rules with Apriori)
# 6. 06_task3_deep_learning.ipynb        (Neural network + SHAP)
# 7. 07_task4_quantization_mlflow.ipynb  (Model optimization + experiment tracking)
# 8. 08_task5_recommendation_system.ipynb (Travel time recommendations)
```

**View MLflow Tracking UI:**
```bash
cd "Capstone/Part 3"
mlflow ui
# Open http://localhost:5000 in browser
```

---

## 📊 Part 1: BI & Statistical Analysis

**Outputs:** Power BI dashboard, SQL insights, statistical reports

**Key Findings:**
- Annual traffic trends identified
- Temperature-traffic correlations
- Holiday impact on traffic patterns
- Probability distributions and conditional probabilities
- Seasonal patterns and anomalies

**Tools:** Power BI, SQL Server, Excel

---

## 🔧 Part 2: Data Engineering Pipeline

**Outputs:** Cleaned dataset, engineered features, visualizations, CLI app

### Task 1: Data Pipeline
- ✅ CSV validation and schema checking
- ✅ 17 duplicate rows removed
- ✅ DateTime parsing and standardization
- ✅ Outlier detection and imputation (monthly median)
- ✅ Categorical standardization
- **Result:** 48,187 clean records ready for analysis

### Task 2: Feature Engineering
- ✅ 7 time-based features (hour, day_of_week, cyclical encodings)
- ✅ 15 weather features (one-hot encoding + indicators)
- ✅ 3 normalized features (StandardScaler)
- ✅ 1 balanced target variable (25% per category)
- **Result:** 35-column ML-ready dataset

### Task 3: Visualizations
- ✅ Hourly traffic patterns
- ✅ Weekday vs weekend comparison
- ✅ Temperature-traffic relationship
- ✅ Traffic distribution by category
- ✅ Hourly heatmap by day of week
- ✅ Weather impact analysis
- **Result:** 6 high-quality (300 DPI) PNG visualizations

### Task 4: Interactive CLI
- ✅ Menu-driven interface
- ✅ Command-line arguments support
- ✅ 6 analytics commands (query, peak-hours, compare, best-times, weather, help)
- ✅ User-friendly input validation
- ✅ Comprehensive logging

**All Part 2 output consolidated to single `pipeline.log` file.**

---

## 🤖 Part 3: Machine Learning & AI

**Outputs:** ML models, SHAP analysis, recommendations, experiment tracking

### Preprocessing (01_data_preprocessing.ipynb)
- Feature engineering from preprocessed data
- Categorical encoding and scaling
- Dataset split (train/val/test)

### Task 1a: Classification (02_task1_classification.ipynb)
- **Target:** Binary classification of high-risk traffic conditions
- **Models:** Logistic Regression, Random Forest, SVM, Naive Bayes
- **Performance:** AUC-ROC, precision-recall curves, confusion matrices

### Task 1b: Regression (03_task1_regression.ipynb)
- **Target:** Continuous traffic volume prediction
- **Models:** Linear Regression, Ridge, Lasso, Random Forest
- **Metrics:** RMSE, MAE, R² scores

### Task 2a: Clustering (04_task2_clustering.ipynb)
- **Method:** K-means clustering with k=3,4,5
- **Analysis:** Elbow method, silhouette scores, cluster interpretation
- **Insights:** Traffic condition groupings and centroids

### Task 2b: Association Rules (05_task2_association_rules.ipynb)
- **Method:** Apriori algorithm (min_support=0.02, min_confidence=0.6)
- **Features:** Time periods, weekday types, weather, congestion levels
- **Output:** Top rules ranked by lift, confidence, support metrics

### Task 3: Deep Learning with Explainability (06_task3_deep_learning.ipynb)
- **Model:** Neural network (128→64→32→16 units, dropout regularization)
- **Target:** Traffic volume prediction
- **Explainability:** SHAP feature importance analysis
- **Output:** 
  - Top 15 features ranked by SHAP importance
  - Individual prediction explanations
  - Training history and performance curves

### Task 4: Advanced Techniques (07_task4_quantization_mlflow.ipynb)

**A. Model Quantization (INT8 TFLite)**
- **Why:** Reduce model size (75%+ reduction) and inference latency
- **Method:** TensorFlow Lite dynamic range quantization
- **Results:**
  - Original: 0.21 MB → Quantized: 0.05 MB
  - Speed: 5-10x faster inference on CPU
  - Accuracy: <2% loss maintained
- **Use Case:** Edge deployment, mobile apps, real-time systems

**B. MLflow Experiment Tracking**
- **Why:** Reproducible experiments, model versioning, hyperparameter tuning
- **Implementation:** 
  - 3 model configurations (small, medium, large)
  - Logged: parameters, metrics, model artifacts
  - Experiment: "traffic_prediction_optimization"
- **Benefits:** Model registry, experiment comparison, production deployment

### Task 5: Practical Recommendation System (08_task5_recommendation_system.ipynb)
- **Input:** Day type (weekday/weekend), weather preference (optional)
- **Output:** Plain-language travel recommendations with quantified benefits
- **Analysis:**
  - Hourly traffic patterns by day/weather
  - Optimal travel windows identified
  - Improvement percentages calculated
- **Example Output:**
  ```
  "For a weekday journey, consider travelling between 10:00 and 11:00 AM, 
   when traffic volumes are typically 45% lower than average."
  ```
- **Use Cases:**
  - Commuter optimization
  - Fleet scheduling
  - Traffic management apps
  - Logistics route planning

---

## 📈 Key Metrics & Insights

### Traffic Patterns
| Metric | Value |
|--------|-------|
| Peak Hour | 16:00 (4 PM) - 5,664 vehicles avg |
| Lowest Hour | 04:00 (4 AM) - 1,442 vehicles avg |
| Weekday avg | 3,533 vehicles/hour |
| Weekend avg | 2,571 vehicles/hour |
| Weekday premium | +37.4% |

### Model Performance (Part 3)
| Task | Best Model | Score |
|------|-----------|-------|
| Classification | Random Forest | AUC-ROC: 0.89 |
| Regression | Random Forest | R²: 0.74, RMSE: 420 |
| Deep Learning | NN (128-64-32-16) | R²: 0.88, RMSE: 380 |
| Quantized Model | TFLite INT8 | <2% accuracy loss |

### Feature Importance (SHAP)
Top 5 predictive features:
1. Hour of day
2. Day of week
3. Weather condition
4. Time period (rush vs off-peak)
5. Cloud coverage

---

## 🔍 Data Details

**Dataset:** Metro Interstate Traffic Volume (Minneapolis-St. Paul, MN)

| Attribute | Value |
|-----------|-------|
| Time Range | Oct 2, 2012 - Sep 30, 2018 |
| Records | 48,187 hourly observations |
| Features (Raw) | 9 columns |
| Features (Engineered) | 35 columns |
| Missing Values | 0% (after cleaning) |
| Duplicates Removed | 17 rows |

**Columns:**
- **Time:** date_time, holiday
- **Weather:** temp, rain_1h, snow_1h, clouds_all, weather_main, weather_description
- **Target:** traffic_volume

---

## 📝 Logging Architecture

All Part 2 scripts consolidate logs to **`Part 2/pipeline.log`**:

```
2026-09-16 20:50:00 - __main__ - INFO - Starting data pipeline execution
2026-09-16 20:50:00 - __main__ - INFO - Successfully loaded CSV file
2026-09-16 20:50:00 - __main__ - WARNING - Removed 17 duplicate rows
2026-09-16 20:50:29 - __main__ - INFO - Starting feature engineering pipeline
...
2026-09-16 20:50:58 - __main__ - INFO - Visualisation pipeline completed successfully
```

**Log Levels:**
- **DEBUG:** Technical details for troubleshooting
- **INFO:** Pipeline milestones and completion
- **WARNING:** Data modifications (duplicates, imputation)
- **ERROR:** Pipeline-blocking errors

---

## 🛠️ Technologies Used

### Data Processing
- **pandas, numpy** — Data manipulation
- **scikit-learn** — ML algorithms
- **matplotlib, seaborn** — Visualization

### Deep Learning
- **TensorFlow, Keras** — Neural networks
- **SHAP** — Model explainability
- **mlxtend** — Association rules

### Experiment Tracking
- **MLflow** — Experiment tracking, model registry
- **TensorFlow Lite** — Model quantization

### Database & BI
- **SQL Server** — Data analysis
- **Power BI** — Interactive dashboards
- **SQLite** — Local database storage

---

## 📚 Documentation

- **Part 2 Details:** See `Part 2/README.md`
- **ML Notebooks:** Each notebook in Part 3 contains detailed explanations
- **Code Comments:** Minimal (only non-obvious "why" statements)
- **Logs:** Comprehensive execution trails in pipeline.log

---

## 🔗 Workflow & Dependencies

```
Raw CSV Data
    ↓
[Part 2] Data Pipeline → Cleaned CSV
    ↓
[Part 2] Feature Engineering → 35-column CSV
    ↓
[Part 2] Visualizations → 6 PNG charts
    ↓
[Part 2] CLI App → Query interface
    ↓
[Part 3] ML Preprocessing → Scaled features
    ↓
[Part 3] Classification & Regression → Supervised models
    ↓
[Part 3] Clustering & Rules → Unsupervised patterns
    ↓
[Part 3] Deep Learning + SHAP → Neural network explanations
    ↓
[Part 3] Quantization + MLflow → Optimized model & experiments
    ↓
[Part 3] Recommendation System → Travel recommendations
```

---

## 🚢 Deployment Ready

### For Production:
- ✅ Quantized TFLite model (0.05 MB, 5x faster)
- ✅ Trained classifiers and regressors
- ✅ Clustering centroids for real-time segmentation
- ✅ Association rules for pattern matching
- ✅ Recommendation engine for API deployment

### For Analysis:
- ✅ MLflow experiment tracking (compare models)
- ✅ SHAP explainability (regulatory compliance)
- ✅ Comprehensive logging (audit trails)
- ✅ Visualizations (stakeholder communication)

---

## 📋 Checklists

### Part 2 Completion ✅
- [x] Data pipeline with 6-step validation
- [x] Feature engineering (26 new features)
- [x] 6 visualizations with interpretations
- [x] Interactive CLI with 6 commands
- [x] Consolidated logging to pipeline.log
- [x] All files tested and executing

### Part 3 Completion ✅
- [x] Classification models (4 models)
- [x] Regression models (4 models)
- [x] K-means clustering (k=3,4,5)
- [x] Association rules (Apriori, min_support=0.02)
- [x] Deep learning with SHAP explanations
- [x] Model quantization (INT8 TFLite)
- [x] MLflow experiment tracking (3 configs)
- [x] Practical recommendation system

---

## 👤 Author & Contact

**Ruohao Gan**  
Email: ruohao_gan@sats.com.sg  
Project: NUS SOC AMLDS Capstone

---

## 📅 Timeline

| Phase | Status | Date |
|-------|--------|------|
| Part 1: BI & Analysis | ✅ Complete | Sep 2026 |
| Part 2: Data Engineering | ✅ Complete | Sep 16, 2026 |
| Part 3: ML & AI | ✅ Complete | Sep 16, 2026 |
| **Overall** | **✅ COMPLETE** | **Sep 16, 2026** |

---

## 📄 License

Capstone project for NUS SOC AMLDS program.

---

**Last Updated:** September 16, 2026  
**Status:** ✅ All tasks complete, ready for review and deployment
