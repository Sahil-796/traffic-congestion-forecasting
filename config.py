"""Every path, constant and threshold lives here.

If a number appears anywhere else in the codebase, it should have come from
this file. That way there is exactly one place to change a decision, and
exactly one place to point at during the viva.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUT_FIG = ROOT / "outputs" / "figures"
OUT_TAB = ROOT / "outputs" / "tables"
OUT_MODEL = ROOT / "outputs" / "models"

for _p in (DATA_RAW, DATA_PROCESSED, OUT_FIG, OUT_TAB, OUT_MODEL):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- dataset
DATASET_NAME = "Metro Interstate Traffic Volume"
DATASET_SOURCE = "UCI Machine Learning Repository (id=492)"
DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/492/"
    "metro+interstate+traffic+volume.zip"
)
DATASET_DOI = "10.24432/C5X60B"
DATASET_LICENSE = "CC BY 4.0"

RAW_CSV = DATA_RAW / "Metro_Interstate_Traffic_Volume.csv"
CLEAN_CSV = DATA_PROCESSED / "traffic_hourly.csv"

# ---------------------------------------------------------------- experiment
RANDOM_STATE = 42

# Parallelism is applied at exactly ONE level (the hyperparameter search).
# The estimators themselves run single-threaded, otherwise every CV worker
# spawns its own thread pool and the machine is oversubscribed.
# 4 keeps the CPU around 40% on a 10-core Mac. Raise to 8 for speed.
N_JOBS = 4

# Chronological boundary. Everything before this trains, everything after
# tests. Never randomised: the rows are ordered in time and shuffling would
# let the model peek at the future.
TEST_START = "2018-01-01"

N_CV_SPLITS = 5          # TimeSeriesSplit folds inside the training period
N_SEARCH_ITER = 20       # random draws per model during tuning

# ---------------------------------------------------------------- domain rules
# Physical sanity limits used during cleaning.
#   0 K is absolute zero -> the sensor clearly reported garbage.
#   9831 mm of rain in an hour would be ~10 metres -> impossible.
TEMP_ABSOLUTE_ZERO = 0.0
RAIN_MAX_MM = 100.0

# Congestion tiers in vehicles/hour, used only to translate a numeric
# prediction into a human label. Chosen from the I-94 lane capacity and the
# observed distribution, not fitted to the test set.
CONGESTION_FREE_MAX = 1500
CONGESTION_MODERATE_MAX = 4000
