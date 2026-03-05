from langchain.tools import tool
from src.utils.db import run_query


@tool
def order_status_tool(order_id: str):
    """
    Fetch order status and delivery timestamps using order_id
    """

    query = f"""
    SELECT
        order_id,
        customer_id,
        order_status,
        order_purchase_timestamp,
        order_delivered_customer_date
    FROM ecommerce_order_customer_status
    WHERE order_id = '{order_id}'
    """

    return run_query(query)


@tool
def payment_info_tool(order_id: str):
    """
    Fetch payment information for an order
    """

    query = f"""
    SELECT
        order_id,
        payment_type,
        payment_installments,
        payment_value
    FROM ecommerce_order_payments
    WHERE order_id = '{order_id}'
    """

    return run_query(query)


@tool
def order_items_tool(order_id: str):
    """
    Fetch products contained in an order
    """

    query = f"""
    SELECT
        order_id,
        product_id,
        price,
        freight_value
    FROM ecommerce_order_items
    WHERE order_id = '{order_id}'
    """

    return run_query(query)


@tool
def product_info_tool(product_id: str):
    """
    Fetch product category and description
    """

    query = f"""
    SELECT *
    FROM ecommerce_product_info
    WHERE product_id = '{product_id}'
    """

    return run_query(query)