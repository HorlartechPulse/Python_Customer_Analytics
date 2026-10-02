# Customer Behavior & Anomaly Analysis — Summary Report

**Dataset:** UCI Adult Income (proxy for socio-economic customer segments)  
**Pipeline:** Python · pandas · matplotlib  

## 1. Data ingestion

- Source: `data/adult.csv`
- Raw rows: **32,561** · columns: **15**

## 2. Data quality

| Metric | Value |
|--------|-------|
| Rows after cleaning | 32,561 |
| Rows dropped | 0 |
| Duplicate rows (raw) | 24 |
| High-income rate (>50K) | 24.1% |

Cleaning rules: strip strings, treat `?` as null, coerce numerics, drop null age/hours/income, derive `high_income`.

## 3. Exploratory findings

### Education and high income
Top education levels by % high earners (see `high_income_by_education.csv` and figure).

### Occupations
Top occupations by high-earner share:

     occupation    n  high_rate  high_rate_pct
Exec-managerial 4066   0.484014           48.4
 Prof-specialty 4140   0.449034           44.9
Protective-serv  649   0.325116           32.5
   Tech-support  928   0.304957           30.5
          Sales 3650   0.269315           26.9

### Behavior segments

                      segment     n  high_income_rate   avg_age  avg_hours
               Core workforce 12839               0.0 45.166524  36.784173
                 Early career  8199               0.0 23.097695  33.551531
 High earner · standard hours  4175               1.0 45.114251  38.062754
 Long hours · not high income  3682               0.0 38.029060  57.786257
High earner · intensive hours  3666               1.0 43.265412  53.912166

### Correlations (numeric)

                  age  education_num  capital_gain  capital_loss  hours_per_week  high_income
age             1.000          0.037         0.078         0.058           0.069        0.234
education_num   0.037          1.000         0.123         0.080           0.148        0.335
capital_gain    0.078          0.123         1.000        -0.032           0.078        0.223
capital_loss    0.058          0.080        -0.032         1.000           0.054        0.151
hours_per_week  0.069          0.148         0.078         0.054           1.000        0.230
high_income     0.234          0.335         0.223         0.151           0.230        1.000

## 4. Anomaly analysis

| Flag | Count |
|------|------:|
| Extreme hours (|z| > 3) | 440 |
| Capital-gain outliers (3×IQR) | 2,712 |
| Young (<25) high income | 61 |
| Any anomaly | 3,153 (9.68%) |

## 5. Figures

- `high_income_by_education.png`
- `occupation_high_earners_donut.png`
- `hours_by_income.png`
- `anomaly_counts.png`
- `behavior_segments.png`


## 6. Recommendations

1. **Target education-linked offers** toward groups with high >50K rates (professional degrees).
2. **Occupation packs** — prioritize Tech-support / Exec-managerial / Prof-specialty style segments for premium products.
3. **Review long-hours low-income** segment for retention risk / upskilling programs.
4. **Investigate capital-gain outliers** before using income models in production (data entry or rare liquidity events).
5. **Monitor young high-income** anomalies for fraud or data-quality issues in onboarding.

---
*Reproducible run: `python src/run_analysis.py`*
