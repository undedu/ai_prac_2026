import sqlite3
from datetime import datetime

from config import DB_PATH


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            prediction TEXT,
            confidence REAL,
            created_at TEXT
        )
    """)

    conn.commit()
    conn.close()


def save_history(filename, predictions):
    """
    сохраняем только TOP-1
    """

    top1 = predictions[0]

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO history (filename, prediction, confidence, created_at)
        VALUES (?, ?, ?, ?)
    """, (
        filename,
        top1["class"],
        top1["probability"],
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()


def load_history():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, filename, prediction, confidence, created_at
        FROM history
        ORDER BY id DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "id": r[0],
            "filename": r[1],
            "prediction": r[2],
            "confidence": r[3],
            "datetime": r[4]
        }
        for r in rows
    ]