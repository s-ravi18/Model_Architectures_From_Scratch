import streamlit as st
from src.rag.chain import build_rag_chain

st.set_page_config(
    page_title="Ecommerce FAQ Bot",
    page_icon="🛒",
    layout="centered"
)

st.title("🛒 Ecommerce Support Chatbot")
st.caption("RAG powered by LangChain + ChromaDB")

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = build_rag_chain()

user_query = st.text_input("Ask a question about payments, delivery, returns, etc.")

if user_query:
    with st.spinner("Searching for the best answer..."):
        response = st.session_state.rag_chain(user_query)

    st.subheader("Answer")
    st.write(response["result"])

    with st.expander("Retrieved Context"):
        for i, doc in enumerate(response["source_documents"], start=1):
            st.markdown(f"**Document {i}**")
            st.write(doc.page_content)