import pandas as pd
import sqlite3
from pathlib import Path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
DB_PATH = project_root / "history" / "history.db"

def export_to_excel(output_path="reports/classification_report.xlsx"):

    conn = sqlite3.connect(DB_PATH)

    query = """
        SELECT id, created_at, filename, prediction, confidence
        FROM history
        ORDER BY created_at DESC
    """

    df = pd.read_sql_query(query, conn)

    conn.close()

    df.columns = [
        "ID",
        "Дата",
        "Файл",
        "Предсказание",
        "Уверенность"
    ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    df.to_excel(output_path, index=False)

    return output_path