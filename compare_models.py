import json
import os
from pathlib import Path

import pandas as pd

from config import METRICS_DIR, MODELS_DIR, REPORTS_DIR

REPORTS_DIR.mkdir(exist_ok=True)

rows = []

for json_file in METRICS_DIR.glob("*.json"):

    with open(json_file, "r", encoding="utf-8") as f:
        history = json.load(f)

    if len(history) == 0:
        continue

    best = max(history, key=lambda x: x["accuracy"])

    model_name = json_file.stem

    model_file = MODELS_DIR / f"{model_name}_best.pth"

    if model_file.exists():
        size = round(model_file.stat().st_size / 1024 / 1024, 2)
    else:
        size = 0

    rows.append({

        "Model": model_name,

        "Accuracy": round(best["accuracy"] * 100, 2),

        "Precision": round(best["precision"] * 100, 2),

        "Recall": round(best["recall"] * 100, 2),

        "F1-score": round(best["f1"] * 100, 2),

        "Inference (ms)": round(best["inference_time"], 2),

        "Epoch time (s)": round(best["epoch_time"], 2),

        "Model size (MB)": size

    })

df = pd.DataFrame(rows)

df = df.sort_values(
    by="Accuracy",
    ascending=False
)

print("\n")
print("=" * 90)
print(df.to_string(index=False))
print("=" * 90)

excel_path = REPORTS_DIR / "model_comparison.xlsx"

df.to_excel(excel_path, index=False)

print(f"\nСохранено:\n{excel_path}")

best_model = df.sort_values(by="Accuracy", ascending=False).iloc[0]["Model"]

best_path = Path("metrics") / "best_model.txt"

with open(best_path, "w") as f:
    f.write(best_model)

print(f"\nЛучшая модель: {best_model}")
print(f"Сохранено в {best_path}")