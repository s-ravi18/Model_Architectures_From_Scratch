import sqlite3
import pandas as pd

DB_PATH = "ecommerce.db"


def run_query(query: str):
    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql(query, conn)

    conn.close()

    return df.to_dict(orient="records")