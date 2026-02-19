
import logging
from typing import Dict, List, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ─── Research RAG (For AI Research Assistant tab) ─────────────────────────────

def build_rag_chain(vectorstore, api_key: str):
    """Build a standard RAG chain for conversational QA."""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=api_key, temperature=0.1)
    
    template = """
    You are an expert financial research analyst. Use the provided context (news chunks) to answer the user's question.
    If the context doesn't contain the answer, say "I don't have enough information in the ingested news to answer this."
    Provide a professional, objective, and detailed response.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Return tuple for app.py to unpack sources
    return (chain, retriever)

def query_rag(chain_tuple, question: str) -> Dict[str, Any]:
    """Execute the RAG chain and return answer + source documents."""
    chain, retriever = chain_tuple
    answer = chain.invoke(question)
    sources = retriever.get_relevant_documents(question)
    return {"answer": answer, "sources": sources}


# ─── Legacy/Quant Support (Keep for alpha_engine.py if needed) ────────────────

def build_quant_chain(vectorstore, api_key: str):
    """Legacy: Build the chain used for the original alpha signal generation."""
    # (Simplified for compatibility)
    return build_rag_chain(vectorstore, api_key)[0]

def query_alpha_signal(chain, technicals: str) -> str:
    """Legacy: Query for a structured signal."""
    # (Simplified for compatibility)
    return chain.invoke(f"Generate alpha signal based on technicals: {technicals}")

def generate_mock_signal() -> str:
    """Return a placeholder JSON signal for demo mode."""
    return """
    {
        "signal_strength": 3,
        "direction": "Neutral",
        "confidence_score": 0.5,
        "reasoning": "Demo mode: No Gemini API key provided. Showing neutral placeholder."
    }
    """
