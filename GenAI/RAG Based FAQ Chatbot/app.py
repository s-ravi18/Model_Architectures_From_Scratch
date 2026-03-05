import streamlit as st
from src.rag.chain import build_rag_chain
from src.agent.ecommerce_agent import build_ecommerce_agent

st.set_page_config(
    page_title="Ecommerce Support Portal",
    page_icon="🛒",
    layout="centered"
)

# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Home", "General FAQs", "Consignment FAQs", "Account Help"]
)

# -------------------------------
# HOME PAGE
# -------------------------------
if page == "Home":

    st.title("🛒 Ecommerce Support Portal")

    st.markdown("""
    Welcome to the **Ecommerce Customer Support Center**.

    Here you can quickly find answers to common questions related to:

    - 📦 Delivery and shipping
    - 💳 Payments and refunds
    - 🔁 Returns and cancellations
    - 📦 Order tracking

    ### How to use this portal

    **Customer FAQs**
    - Ask any question about orders, payments or returns.
    - Our AI assistant will search the knowledge base and provide answers.

    **Consignment Status**
    - Track the current status of your order.

    **Account Help**
    - Get assistance related to account settings and login.

    Use the **navigation panel on the left** to explore different sections.
    """)
    
    
    st.divider()

    st.subheader("Generate a random customer identification number")

    if "customer_id" not in st.session_state:
        st.session_state.customer_id = None

    if st.button("Generate"):
        st.session_state.customer_id = "CUST-482917"

    if st.session_state.customer_id:
        st.success(f"Customer ID: {st.session_state.customer_id}")    

# -------------------------------
# CUSTOMER FAQ PAGE
# -------------------------------
elif page == "General FAQs":

    st.title("🛒 Ecommerce FAQ Chatbot")
    st.caption("RAG powered by LangChain + ChromaDB")

    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = build_rag_chain()

    user_query = st.text_input(
        "Ask a question about payments, delivery, returns, etc."
    )

    if user_query:
        with st.spinner("Searching for the best answer..."):
            response = st.session_state.rag_chain(user_query)

        st.subheader("Answer")
        st.write(response["result"])

        with st.expander("Retrieved Context"):
            for i, doc in enumerate(response["source_documents"], start=1):
                st.markdown(f"**Document {i}**")
                st.write(doc.page_content)

# -------------------------------
# DELIVERY STATUS PAGE
# -------------------------------
elif page == "Consignment FAQs":

    st.title("📦 Consignment FAQs")

    st.markdown(
        "Ask questions about your order, payment, or product details."
    )

    if "delivery_agent" not in st.session_state:
        st.session_state.delivery_agent = build_ecommerce_agent()

    user_query = st.text_input("Enter your query about your order")

    if user_query:

        with st.spinner("Checking order details..."):

            response = st.session_state.delivery_agent.run(user_query)

        st.subheader("Response")
        st.write(response)

# -------------------------------
# ACCOUNT HELP PAGE
# -------------------------------
elif page == "Account Help":

    st.title("👤 Account Help")

    st.markdown("""
    This section will help customers with:

    - Resetting passwords
    - Updating profile information
    - Managing saved payment methods
    - Account security

    Feature coming soon...
    """)