import sqlite3
import pandas as pd
from pathlib import Path

# Project root
ROOT_DIR = Path(__file__).resolve().parents[2]

# Data folder
DATA_DIR = ROOT_DIR / "data"

# Database location
DB_PATH = ROOT_DIR / "ecommerce.db"


def load_csv_to_sqlite():

    conn = sqlite3.connect(DB_PATH)

    print("Connected to SQLite")

    # ------------------------------
    # 1 Order Items
    # ------------------------------
    df_order_items = pd.read_csv(DATA_DIR / "order_items_dataset.csv")

    df_order_items.to_sql(
        "ecommerce_order_items",
        conn,
        if_exists="replace",
        index=False
    )

    print("Loaded: ecommerce_order_items")

    # ------------------------------
    # 2 Order Payments
    # ------------------------------
    df_payments = pd.read_csv(DATA_DIR / "order_payments_dataset.csv")

    df_payments.to_sql(
        "ecommerce_order_payments",
        conn,
        if_exists="replace",
        index=False
    )

    print("Loaded: ecommerce_order_payments")

    # ------------------------------
    # 3 Orders / Customer Status
    # ------------------------------
    df_orders = pd.read_csv(DATA_DIR / "orders_dataset.csv")

    df_orders.to_sql(
        "ecommerce_order_customer_status",
        conn,
        if_exists="replace",
        index=False
    )

    print("Loaded: ecommerce_order_customer_status")

    # ------------------------------
    # 4 Product Info
    # ------------------------------
    df_products = pd.read_csv(DATA_DIR / "products_dataset.csv")

    df_products.to_sql(
        "ecommerce_product_info",
        conn,
        if_exists="replace",
        index=False
    )

    print("Loaded: ecommerce_product_info")

    conn.close()

    print("\nDatabase created successfully:", DB_PATH)


if __name__ == "__main__":
    load_csv_to_sqlite()