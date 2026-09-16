# Metro Interstate Traffic Volume Analysis

A comprehensive data science project for cleaning, engineering, visualizing, and querying traffic volume data using Python.

## 📋 Project Overview

This project implements a complete data pipeline for analyzing Metro Interstate traffic patterns:

1. **Data Pipeline** — Loads raw CSV, validates schema, removes duplicates, standardizes values, handles outliers
2. **Feature Engineering** — Creates time-based, weather-based, normalized, and target features for ML
3. **Visualizations** — Generates 6 meaningful traffic pattern visualizations
4. **Interactive CLI** — Provides command-line and menu-based interfaces to query traffic data

## 📁 Project Structure

```
Capstone/Part 2/
├── README.md                                      # This file
├── pipeline.py                                    # Task 1: Data cleaning pipeline
├── feature_engineering.py                         # Task 2: Feature engineering
├── visualisation.py                               # Task 3: Traffic visualizations
├── traffic_analytics_app.py                       # Task 4: CLI application (menu + command-line)
├── visualisations/                                # Output folder
│   ├── 01_traffic_by_hour.png
│   ├── 02_weekday_vs_weekend.png
│   ├── 03_temperature_vs_traffic.png
│   ├── 04_traffic_distribution.png
│   ├── 05_hourly_heatmap.png
│   └── 06_weather_impact.png
└── pipeline.log                               # Pipeline execution log
```

## 🚀 How to Run the Project

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Task 1: Data Pipeline

Run the data cleaning pipeline:

```bash
# Executable script
python pipeline.py

```

**Output:**
- Cleaned CSV: `Metro_Interstate_Traffic_Volume_cleaned.csv`
- Log file: `pipeline.log`

### Task 2: Feature Engineering

Transform cleaned data into ML-ready features:

```bash
# Executable script
python feature_engineering.py

```

**Output:**
- Feature-engineered CSV: `Metro_Interstate_Traffic_Volume_features.csv`
- Log file: `pipeline.log`

### Task 3: Visualizations

Generate traffic pattern visualizations:

```bash
# Executable script
python visualisation.py

```

**Output:**
- 6 PNG visualizations in `visualisations/` folder
- Log file: `pipeline.log`

### Task 4: Interactive Traffic Analytics CLI

**Two ways to use the application:**

#### Option A: Interactive Menu (Default)
```bash
python traffic_analytics_app.py
```
Displays a user-friendly menu where you select options by number:
1. Query Traffic by Date/Time
2. Find Peak Traffic Hours
3. Compare Weekday vs Weekend
4. Find Best Times to Travel
5. Analyze Weather Impact
6. View Help
0. Exit

#### Option B: Command-Line Arguments
```bash
# Query traffic for a specific date/time
python traffic_analytics_app.py query-time 2016-06-15 16:00

# Show peak traffic hours
python traffic_analytics_app.py peak-hours --limit 5

# Compare weekday vs weekend
python traffic_analytics_app.py weekday-compare

# Find best times to travel
python traffic_analytics_app.py best-times --max-volume 1500

# Analyze weather impact
python traffic_analytics_app.py weather-impact

# View help
python traffic_analytics_app.py help
```

**Log file:** `traffic_app.log`

## 📊 Logging Configuration

### Overview

All Python scripts use Python's `logging` module with the following configuration:

- **Logger Name**: Uses `logging.getLogger(__name__)` in all modules (no root logger)
- **Log Handlers**: 
  - **Console Handler**: INFO level and above (user-facing messages)
  - **File Handler**: DEBUG level and above (comprehensive audit trail)
- **Log Format**: 
  ```
  YYYY-MM-DD HH:MM:SS - MODULE_NAME - LOG_LEVEL - MESSAGE
  ```

### Log Levels and Usage

| Level | Color | Purpose | Examples |
|-------|-------|---------|----------|
| **DEBUG** | Gray | Fine-grained technical details for troubleshooting | Parsed values, intermediate calculations, quartile thresholds |
| **INFO** | Blue | Normal pipeline milestones | File loaded, step completed, data saved, command executed |
| **WARNING** | Yellow | Recoverable unexpected events | Rows dropped, values imputed, outliers handled, data modified |
| **ERROR** | Red | Errors that prevent pipeline continuation | File not found, schema validation failed, parsing error |

### Log Files

Each task generates its own log file:

- **pipeline.log** — Data pipeline execution
  - File loading and schema validation
  - Duplicate removal, datetime parsing
  - Categorical standardization, outlier imputation
  
- **feature_engineering.log** — Feature creation
  - Dataset shape before/after
  - Time feature calculations
  - Weather feature encoding
  - Normalization statistics
  
- **visualisation.log** — Visualization generation
  - Each figure save confirmation
  - Figure interpretation and statistics
  - Peak hour/day identification
  
- **traffic_app.log** — CLI application usage
  - Command invocation with arguments
  - Query results and statistics
  - User input validation
  - Error handling

### No Print Statements

✅ **Logging is used for all internal progress reporting**
- Pipeline status: Logged at INFO level
- Data changes: Logged at WARNING level
- Intermediate values: Logged at DEBUG level

✓ **Print is used only for CLI output**
- Query results displayed to user
- Menu options and prompts
- Table-formatted statistics
- Error messages to user

## 🔑 Key Features

### Data Pipeline
- ✅ CSV loading with try/except (no bare except)
- ✅ Schema validation before processing
- ✅ Duplicate removal with logging
- ✅ DateTime parsing and validation
- ✅ Categorical standardization
- ✅ Outlier detection (monthly median imputation)
- ✅ Comprehensive logging (INFO/WARNING/ERROR)

### Feature Engineering
- ✅ 7 time features (hour, day_of_week, is_weekend, cyclical encodings)
- ✅ 15 weather features (one-hot encoding + derived indicators)
- ✅ 3 normalized features (StandardScaler: mean≈0, std≈1)
- ✅ 1 balanced target variable (25% per category)
- ✅ Dataset shape logging before/after
- ✅ DEBUG-level intermediate value logging

### Visualizations
- ✅ 6 meaningful traffic pattern charts
- ✅ Each visualization includes interpretation
- ✅ INFO-level logging for each saved figure
- ✅ 300 DPI PNG output

### CLI Application
- ✅ 6 commands (help, query-time, peak-hours, weekday-compare, best-times, weather-impact)
- ✅ Interactive menu interface (default)
- ✅ Command-line interface (backward compatible)
- ✅ User-friendly input validation
- ✅ Clear error messages (no raw tracebacks)
- ✅ Comprehensive logging of commands and arguments

## 📈 Data Flow

```
Raw Data
   ↓
[Task 1: Pipeline] → Cleaned Data
   ↓
[Task 2: Features] → ML-Ready Data
   ↓
[Task 3: Visualize] → Charts & Insights
   ↓
[Task 4: CLI] → Query Interface
```

## 🔍 Data Coverage

- **Time Period**: October 2, 2012 to September 30, 2018
- **Records**: 48,187 hourly observations
- **Hours**: 0-23 (midnight to 11 PM)
- **Columns**: 9 original → 35 engineered

## 📌 Key Insights

- **Peak Traffic**: 16:00 (4 PM) with ~5,664 vehicles
- **Weekday Impact**: 37.4% higher than weekends
- **Best Travel Time**: 04:00 AM (lowest traffic)
- **Weather Effect**: 75.5% difference between best/worst conditions
- **Target Balance**: Perfect 25% distribution across 4 congestion levels

## 🛠️ Development Notes

### Logging Best Practices Implemented

1. ✅ Module-level loggers with `logging.getLogger(__name__)`
2. ✅ Handler configuration only in entry points
3. ✅ Formatter with timestamp, level, module name, message
4. ✅ Meaningful log levels (DEBUG → ERROR)
5. ✅ No print() for internal progress
6. ✅ User-facing output via print() only

### File Organization

- Python scripts: Single-purpose modules
- Jupyter notebooks: Interactive exploration versions
- Log files: Per-module audit trails
- CSV files: Raw → Cleaned → Featured progression
- Images: High-quality (300 DPI) visualizations

## 📝 License

This project is created as part of a capstone assignment.

## 👤 Author

Ruohao Gan (ruohao_gan@sats.com.sg)

## 📅 Timeline

- **Task 1**: Data Pipeline Construction
- **Task 2**: Feature Engineering
- **Task 3**: Traffic Pattern Visualizations
- **Task 4**: Mini CLI Application
- **Task 5**: GitHub & Documentation

---

**Last Updated**: September 16, 2026
