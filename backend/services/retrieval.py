from langchain_core.documents import Document 
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import StrOutputParser 
from langchain_qdrant import QdrantVectorStore
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
import asyncio
# ---- HYBRID SEARCH ----
async def contextual_compression(vector_store: QdrantVectorStore, query: str, k: int, llm=None):
    base_retriever = vector_store.as_retriever()
    if llm is None:
        return []
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    results = await compression_retriever.ainvoke(query)
    results = results[:k]
    print(f"\nTop {k} compressed search results for: '{query}'")
    for idx, res in enumerate(results[:k]):
        print(f"\nResult {idx+1}:")
        print("Modality:", res.metadata.get("modality"))
        print("Compressed Content:", res.page_content[:200])
        print("Metadata:", res.metadata)
    return results[:k]

async def summarize_docs(retreived_docs: list[Document], llm, query):
    context = "\n\n".join([doc.page_content for doc in retreived_docs])
    prompt = PromptTemplate(
        template="Summarize the following content{query_part}:\n\n{context}\n\nSummary:",
        input_variables=["context", "query_part"],
    )

    prompt_vars = {"context": context}
    
    if query:
        prompt_vars["query_part"] = f" in response to the query: '{query}'"
    else:
        prompt_vars["query_part"] = ""

    chain = prompt | llm | StrOutputParser()
    summary = await chain.ainvoke(prompt_vars)
    return summary