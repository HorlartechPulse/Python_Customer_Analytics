from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "adult.csv"
FIG_DIR = ROOT / "outputs" / "figures"
REPORT_DIR = ROOT / "reports"
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
INCOME_HIGH = ">50K"
