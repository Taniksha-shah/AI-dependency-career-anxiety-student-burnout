# 🧠 Analyzing AI Dependency, Career Anxiety & Student Burnout Among Students

> A data-driven EDA exploring how student reliance on AI tools correlates with psychological stress, career preparedness, and academic motivation.

---

## 📌 Overview

This notebook performs a full **Exploratory Data Analysis (EDA)** on a student survey dataset, examining the intersecting relationships between:

- 🤖 AI tool adoption & dependency
- 💼 Career anxiety & placement readiness
- 🔥 Burnout, stress & motivation
- 📚 Academic habits & self-development behavior

The analysis moves from raw data inspection → feature engineering → skew profiling → correlation heatmapping to produce interpretable, actionable insights.

---

## 📂 Dataset

**Source:** [Kaggle — AI Dependency, Career Anxiety and Student Burnout](https://www.kaggle.com/datasets/sridipbasu/ai-depndency-career-anxiety-and-student-burnout)  
**File:** `ai_dependency_career_anxiety_students.csv`  
**Author:** sridipbasu

### Feature Categories

| Category | Key Features |
|---|---|
| 🧑 Demographics | `age`, `gender`, `stream`, `degree_type`, `college_tier`, `urban_or_rural`, `year_of_study` |
| 🤖 AI Usage | `daily_ai_tool_usage_hrs`, `primary_ai_tools_used`, `uses_ai_for_assignments`, `ai_replaces_own_thinking_score`, `ai_dependency_score` |
| 💼 Career | `placement_anxiety_score`, `fear_of_job_loss_to_ai`, `career_clarity_score`, `internship_experience`, `resume_confidence_score`, `interview_anxiety_score`, `overall_career_readiness_score` |
| 🧠 Wellbeing | `sleep_hours`, `stress_level`, `burnout_score`, `motivation_score`, `social_media_hrs_per_day`, `self_learning_hours_per_week`, `skill_development_courses_taken` |

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-EDA-150458?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-linear_algebra-013243?logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-visualization-orange)
![Seaborn](https://img.shields.io/badge/Seaborn-heatmap-4c72b0)
![scikit-learn](https://img.shields.io/badge/scikit--learn-encoding-F7931E?logo=scikit-learn&logoColor=white)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/Taniksha-shah/AI-dependency-career-anxiety-student-burnout.git
cd AI-dependency-career-anxiety-student-burnout
```

### 2. Install dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 3. Download the dataset
```bash
# Using Kaggle CLI
kaggle datasets download sridipbasu/ai-depndency-career-anxiety-and-student-burnout
unzip ai-depndency-career-anxiety-and-student-burnout.zip
```

### 4. Update the data path in the notebook
```python
# Change this line if running locally
df = pd.read_csv('./ai_dependency_career_anxiety_students.csv')
```

### 5. Run the notebook
```bash
jupyter notebook notebook.ipynb
```

> **Note:** The notebook was originally built on Kaggle where the dataset is pre-mounted at `/kaggle/input/`. If running locally, update the path as shown above.

---

## 📊 Analysis Pipeline

```
1. Data Loading & Inspection     →  shape, dtypes, describe(), null audit
2. Distribution Visualization    →  histogram for every column
3. Missing Value Imputation      →  "None" fill / median fill / mode fill
4. Feature Encoding              →  one-hot + ordinal encoding + feature engineering
5. Skew Analysis                 →  Pearson skewness coefficients for 37+ features
6. Correlation Heatmap           →  37×37 Pearson matrix + scatter plot validation
```

### Encoding Strategy

| Column | Strategy | Reason |
|---|---|---|
| `primary_ai_tools_used` | Fill NaN → `"None"`, then one-hot | Nominal, no ordering |
| `gender`, `stream`, `degree_type`, `urban_or_rural` | One-hot (`get_dummies`) | Nominal categories |
| `uses_ai_for_assignments` | Ordinal: Never → Rarely → Sometimes → Frequently → Always | Ordered frequency |
| `college_tier` | Ordinal: Tier 3 → Tier 2 → Tier 1 | Ordered quality |
| `self_learning_hours_per_week`, `social_media_hrs_per_day`, `sleep_hours` | Median imputation | Numeric, skew-robust |
| `seeks_career_counseling` | Mode imputation | Categorical |

### Engineered Features

```python
processed_df['CS/IT']      = processed_df['stream_CS/IT']
processed_df['non_CS/IT']  = processed_df['stream_Arts/Sciences'] + processed_df['stream_Commerce/Management'] + processed_df['stream_Engineering (Non-CS)']
processed_df['Urban']      = processed_df['urban_or_rural_Urban']
```

---

## 🔍 Key Findings

### ✅ Positive Correlations

**1. AI Usage → Psychological Stress**
Higher `daily_ai_tool_usage_hrs`, `ai_dependency_score`, and `ai_replaces_own_thinking_score` are positively correlated with `burnout_score`, `stress_level`, `interview_anxiety_score`, and `seeks_career_counseling` — particularly among CS/IT students.

**2. Year of Study → Career Readiness**
Students in later years, especially from engineering backgrounds, show more `internship_experience`, higher `resume_confidence_score`, more `weekly_job_application_count`, and stronger `overall_career_readiness_score`.

**3. Self-Development → Motivation**
`daily_study_hours`, `self_learning_hours_per_week`, and `skill_development_courses_taken` are all positively linked with `motivation_score`, `career_clarity_score`, and `overall_career_readiness_score`.

**4. College Tier → Internship Access**
Higher-tier colleges correlate with greater internship exposure, suggesting structural advantages in industry access.

**5. Urban Background → AI Adoption**
A mild positive correlation exists between urban students and AI usage variables, likely reflecting better access and exposure.

---

### ❌ Negative Correlations

**1. AI Dependency → Lower Motivation & Readiness**
Higher AI dependency and burnout scores are negatively correlated with `motivation_score` and `overall_career_readiness_score`.

**2. Non-CS/IT Background → Lower AI Usage**
Non-technical students exhibit lower AI usage, dependency, and associated anxiety metrics compared to CS/IT peers.

**3. Social Media → Reduced Productivity**
Higher `social_media_hrs_per_day` is negatively associated with `daily_study_hours`, `self_learning_hours_per_week`, and `motivation_score`.

**4. Sleep Deprivation → Higher Stress**
Lower `sleep_hours` correlates with higher `stress_level` and elevated `burnout_score`.

---

### 💡 Core Insight

> Students with higher AI usage and dependency tend to experience increased stress, burnout, and interview anxiety — especially in CS/IT streams. Meanwhile, proactive behaviors like self-learning, internships, and skill development are strongly linked with motivation and career readiness. Social media overuse and sleep deprivation further compound negative outcomes.

---

## 📈 Skew Analysis Highlights

| Feature | Observation |
|---|---|
| Population | Mostly undergrad urban students |
| Stream | Heavily engineering/CS-oriented |
| AI adoption | Moderate overall; heavy use concentrated in a subgroup |
| Career concern | Strong right-skew on `fear_of_job_loss_to_ai` |
| Tool adoption | ChatGPT dominant; Copilot & Perplexity show niche adoption |
| Wellbeing | Stress and burnout relatively balanced |
| Career prep | Uneven — `skill_development_courses_taken` right-skewed |
| Job seeking | `weekly_job_application_count` heavily right-skewed |

---

## 📁 Notebook Structure

```
notebook.ipynb
├── Section 1  — Importing Required Libraries
├── Section 2  — Dataset Overview
│   ├── Load CSV → df
│   ├── df.shape, df.head(), df.info(), df.describe()
│   └── Null audit + unique value inspection
├── Section 3  — Visualizing Distributions
│   └── Histogram loop across all columns
├── Section 4  — Handling Missing Values
├── Section 5  — Encoding
│   ├── One-hot encoding (get_dummies)
│   ├── Ordinal encoding (OrdinalEncoder)
│   └── Feature engineering (CS/IT, non_CS/IT, Urban)
├── Section 6  — Skew Analysis
│   └── Skew coefficients for 37 features + interpretation
├── Section 7  — Correlation Analysis
│   ├── 37×37 Pearson heatmap
│   ├── Sorted correlation with overall_career_readiness_score
│   └── Scatter plots: AI usage vs stress | self-learning vs readiness | sleep vs burnout
└── Section 8  — Conclusion & Future Enhancements
```

---

## 🔮 Future Enhancements

- [ ] **ML Models** — Predict burnout, stress, and career readiness with classification/regression models
- [ ] **SHAP Analysis** — Identify most influential features driving anxiety and AI dependency
- [ ] **Hypothesis Testing** — Statistically validate correlations with t-tests, ANOVA, chi-squared
- [ ] **Longitudinal Data** — Track how AI dependency and anxiety evolve over a student's academic journey
- [ ] **Interactive Dashboards** — Build Plotly/Streamlit dashboards for dynamic filtering and exploration
- [ ] **Clustering** — Apply K-Means or DBSCAN to identify distinct student behavioral profiles
- [ ] **Larger Dataset** — Expand to more diverse populations, geographies, and institution types
- [ ] **Cross-Stream Comparison** — Systematic comparison of AI usage across Arts, Commerce, and Engineering streams

---

## 📄 License

This project is open source. Dataset is publicly available on Kaggle under its respective license.

---

## 🙌 Acknowledgements

- Dataset: [sridipbasu on Kaggle](https://www.kaggle.com/datasets/sridipbasu/ai-depndency-career-anxiety-and-student-burnout)
- Environment: [Kaggle Notebooks](https://www.kaggle.com/code)