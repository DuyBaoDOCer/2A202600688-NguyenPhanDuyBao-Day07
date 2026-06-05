from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        # 1. Retrieve the most relevant chunks
        results = self.store.search(question, top_k=top_k)

        # 2. Build context from retrieved chunks
        if results:
            context_parts = []
            for i, result in enumerate(results, start=1):
                source = result["metadata"].get("source", result["metadata"].get("doc_id", f"chunk-{i}"))
                context_parts.append(f"[{i}] (source: {source})\n{result['content']}")
            context = "\n\n".join(context_parts)
        else:
            context = "No relevant documents found."

        # 3. Build structured RAG prompt
        prompt = (
            "You are a helpful assistant. Answer the question using ONLY the context below.\n"
            "If the context does not contain the answer, say 'I don't have enough information.'\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        # 4. Call the LLM
        return self.llm_fn(prompt)
