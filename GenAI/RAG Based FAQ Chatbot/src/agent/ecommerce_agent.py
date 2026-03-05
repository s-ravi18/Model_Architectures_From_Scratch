from langchain.agents import initialize_agent
from langchain.agents import AgentType

from src.llm.model import get_llm

from src.tools.ecommerce_tools import (
    order_status_tool,
    payment_info_tool,
    order_items_tool,
    product_info_tool
)


def build_ecommerce_agent():

    llm = get_llm()

    tools = [
        order_status_tool,
        payment_info_tool,
        order_items_tool,
        product_info_tool
    ]

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True
    )

    return agent