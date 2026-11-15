"""Project-wide settings: which apps to study and where data lives."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW_REVIEWS = DATA / "raw" / "reviews.csv"
LLM_LABELS = DATA / "processed" / "llm_labels.csv"
HUMAN_LABELS = DATA / "labels" / "human_labels.csv"
FIGURES = ROOT / "figures"

# Target app first, then the competitors it is compared against.
TARGET_APP = "build.designand.riselife"
APPS = {
    "build.designand.riselife": "Life Reset",
    "com.kryptoniteappstudio.kaizen": "Kaizen AI",
    "com.habitrpg.android.habitica": "Habitica",
}

# Only keep reviews from the window in which all three apps were live, so an
# app with a decade of history (Habitica) is compared on the same period.
SINCE = "2024-10-01"

# Local model served by Ollama (https://ollama.com). Pull it with `ollama pull qwen3:14b`.
LLM_MODEL = "qwen3:14b"

# Fixed so topic models and samples are identical on every run.
SEED = 42
