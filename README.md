# Python Customer Behavior & Anomaly Analytics

**Case study:** Customer behavior and anomaly analysis using the UCI **Adult Income** dataset as a socio-economic customer proxy.

## Deliverables

| Deliverable | Path |
|-------------|------|
| Data ingestion | `src/ingest.py` |
| Data quality checks | `src/quality.py` |
| pandas analysis | `src/analysis.py` |
| Visualization / EDA | `src/visualize.py` |
| Reproducible pipeline | `src/run_analysis.py` |
| Notebook-style script | `notebooks/01_customer_behavior_eda.py` |
| Summary report | `reports/summary_report.md` (after run) |
| Figures | `outputs/figures/` |

## Quick start

```bash
cd Python_Customer_Analytics
pip install -r requirements.txt
python src/run_analysis.py
```

## Dataset

- **UCI Adult** (`data/adult.csv`): census-like attributes (age, education, occupation, hours, income class).
- Target proxy: `income` → `high_income` (>50K).

## Analysis modules

1. **Ingest** — CSV load, header normalization  
2. **Quality** — null/`?` profiling, cleaning, quality JSON  
3. **Behavior** — high-income rates by education/occupation, rule-based segments  
4. **Anomalies** — extreme hours (z-score), capital-gain IQR outliers, young high-earners  
5. **Visuals** — bar, donut, histograms, anomaly counts  

## Author

Abdulwaheed Toheeb Olanrewaju — Python Analytics Portfolio
