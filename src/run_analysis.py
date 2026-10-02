"""End-to-end reproducible pipeline."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# allow running as script
sys.path.insert(0, str(Path(__file__).parent))

from config import REPORT_DIR, FIG_DIR
from ingest import load_raw
from quality import profile, clean, quality_report
from analysis import (
    high_income_by_category,
    occupation_high_earner_share,
    correlation_numeric,
    detect_anomalies,
    anomaly_summary,
    behavior_segments,
)
from visualize import (
    plot_high_income_by_education,
    plot_occupation_donut,
    plot_hours_distribution,
    plot_anomaly_counts,
    plot_segment_sizes,
)


def main():
    print("1. Ingest...")
    raw = load_raw()
    print(f"   Raw shape: {raw.shape}")

    print("2. Quality profile + clean...")
    prof = profile(raw.replace("?", pd.NA))
    clean_df = clean(raw)
    qrep = quality_report(raw, clean_df)
    print(f"   Clean shape: {clean_df.shape}")

    print("3. Analysis...")
    by_edu = high_income_by_category(clean_df, "education", min_n=30)
    by_occ = high_income_by_category(clean_df, "occupation", min_n=50)
    top_occ = occupation_high_earner_share(clean_df, top_n=5)
    corr = correlation_numeric(clean_df)
    flagged = detect_anomalies(clean_df)
    anom = anomaly_summary(flagged)
    segs = behavior_segments(clean_df)

    print("4. Visualizations...")
    figs = [
        plot_high_income_by_education(clean_df),
        plot_occupation_donut(top_occ),
        plot_hours_distribution(clean_df),
        plot_anomaly_counts(anom),
        plot_segment_sizes(segs),
    ]

    print("5. Export tables + report...")
    prof.to_csv(REPORT_DIR / "data_profile.csv", index=False)
    by_edu.to_csv(REPORT_DIR / "high_income_by_education.csv", index=False)
    by_occ.to_csv(REPORT_DIR / "high_income_by_occupation.csv", index=False)
    top_occ.to_csv(REPORT_DIR / "top_occupations_high_earners.csv", index=False)
    corr.to_csv(REPORT_DIR / "numeric_correlation.csv")
    segs.to_csv(REPORT_DIR / "behavior_segments.csv", index=False)
    flagged[flagged["anomaly_score"] > 0].head(500).to_csv(
        REPORT_DIR / "anomaly_sample.csv", index=False
    )
    (REPORT_DIR / "quality_summary.json").write_text(json.dumps(qrep, indent=2))
    (REPORT_DIR / "anomaly_summary.json").write_text(json.dumps(anom, indent=2))

    # Markdown summary report
    report = f"""# Customer Behavior & Anomaly Analysis — Summary Report

**Dataset:** UCI Adult Income (proxy for socio-economic customer segments)  
**Pipeline:** Python · pandas · matplotlib  

## 1. Data ingestion

- Source: `data/adult.csv`
- Raw rows: **{qrep['rows_raw']:,}** · columns: **{raw.shape[1]}**

## 2. Data quality

| Metric | Value |
|--------|-------|
| Rows after cleaning | {qrep['rows_clean']:,} |
| Rows dropped | {qrep['rows_dropped']:,} |
| Duplicate rows (raw) | {qrep['duplicate_rows_raw']:,} |
| High-income rate (>50K) | {qrep['high_income_rate']*100:.1f}% |

Cleaning rules: strip strings, treat `?` as null, coerce numerics, drop null age/hours/income, derive `high_income`.

## 3. Exploratory findings

### Education and high income
Top education levels by % high earners (see `high_income_by_education.csv` and figure).

### Occupations
Top occupations by high-earner share:

{top_occ.to_string(index=False)}

### Behavior segments

{segs.to_string(index=False)}

### Correlations (numeric)

{corr.to_string()}

## 4. Anomaly analysis

| Flag | Count |
|------|------:|
| Extreme hours (|z| > 3) | {anom['extreme_hours']:,} |
| Capital-gain outliers (3×IQR) | {anom['capital_gain_outliers']:,} |
| Young (<25) high income | {anom['young_high_income']:,} |
| Any anomaly | {anom['any_anomaly']:,} ({anom['anomaly_rate_pct']}%) |

## 5. Figures

"""
    for f in figs:
        report += f"- `{f.name}`\n"
    report += """

## 6. Recommendations

1. **Target education-linked offers** toward groups with high >50K rates (professional degrees).
2. **Occupation packs** — prioritize Tech-support / Exec-managerial / Prof-specialty style segments for premium products.
3. **Review long-hours low-income** segment for retention risk / upskilling programs.
4. **Investigate capital-gain outliers** before using income models in production (data entry or rare liquidity events).
5. **Monitor young high-income** anomalies for fraud or data-quality issues in onboarding.

---
*Reproducible run: `python src/run_analysis.py`*
"""
    (REPORT_DIR / "summary_report.md").write_text(report)
    print("Done. Report:", REPORT_DIR / "summary_report.md")
    print("Figures:", list(FIG_DIR.glob("*.png")))


if __name__ == "__main__":
    main()
