
# NYC School Bus Delay Prediction

*Forecasting delay duration & severity using machine learning on NYC DOE’s breakdown and delay dataset.*

## 1. Project Overview
This project builds a data‑driven delay prediction system using **1.22M+ incident records**, cleaned to **1.01M usable rows**, to:
- Predict bus delay duration (regression)
- Classify delays into Low / Moderate / High categories (classification)
- Identify borough‑level, vendor‑level, and temporal delay patterns
- Provide foundations for a real‑time “Delay‑Radar” system

## 2. Dataset Summary
- Raw size: 1.22M rows × 20 columns  
- Cleaned: 1.01M rows × 19 columns  
- Issues fixed: missing values, inconsistent recording, unrealistic values, time parsing.

## 3. Data Preparation
```python
df['Delay_Minutes'] = df['How_long_delayed'].str.extract(r'(\d+\.?\d*)').astype(float)
df['HOUR'] = pd.to_datetime(df['Occurred_On']).dt.hour
df = df[df['Number_Of_Students_On_The_Bus'] <= 80]
```

## 4. Feature Engineering
```python
df['Delay_Class'] = pd.cut(
    df['Delay_Minutes'],
    bins=[-1, 20, 45, df['Delay_Minutes'].max()],
    labels=[0, 1, 2]
)

freq_cols = ['Bus_Company_Name', 'Route_Number']
for col in freq_cols:
    df[col + '_FreqEnc'] = df[col].map(df[col].value_counts())
```

## 5. Exploratory Data Analysis
Key insights:
- Delays are right‑skewed (peak 20–45 min range)
- Highest delays: Rockland, Manhattan, Queens
- Delays increased post‑2016 (~20 min rise)
- Delay_Minutes strongly correlates with Delay_Class

## 6. Modeling Results

### Regression
| Model | MAE | R² |
|-------|------|------|
| Linear Regression | 13 min | 0.16 |
| **XGBoost Regressor** | **8 min** | **0.64** |

### Classification
| Model | Accuracy | Notes |
|-------|-----------|--------|
| Logistic Regression | 84% | Struggled separating classes |
| **XGBoost Classifier** | **96%** | High‑delay recall: 83% |

## 7. System Concept — “Delay‑Radar”
1. Data validation pipeline  
2. Predictive ML engine  
3. Real‑time classification  
4. Dashboard + parent alerts  
5. Vendor performance monitoring  

## 8. Recommended Repo Structure
```
nyc-school-bus-delay-prediction/
│── data/
│── notebooks/
│── src/
│── results/
│── README.md
│── requirements.txt
```

## 9. Future Enhancements
- Integrate weather + real‑time traffic APIs  
- Build a live dashboard (Streamlit / FastAPI)  
- Automate ETL  
- Vendor‑level scorecards  

## Author
Abrar Altaf Lone — MS Data Science
