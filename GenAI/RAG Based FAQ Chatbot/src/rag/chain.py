from langchain.chains import RetrievalQA
from src.rag.retriever import get_retriever
from src.llm.model import get_llm
from src.utils.prompt import RAG_PROMPT


def build_rag_chain():
    retriever = get_retriever()
    llm = get_llm()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": RAG_PROMPT
        }
    )

    return qa_chain