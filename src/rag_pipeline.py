"""Task 3: RAG core — retriever + prompt + generator.

Loads the persisted ChromaDB vector store (built in Task 2, or swap in the
pre-built full-scale store), retrieves the most relevant complaint chunks for a
question, grounds an open-source LLM on them, and returns an evidence-backed
answer plus the source chunks.

Embedding model must match the one used to build the store
(all-MiniLM-L6-v2). The generator defaults to a small, CPU-friendly
instruction-tuned model (FLAN-T5) so it runs without a GPU or API key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.build_vector_store import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    VECTOR_STORE_DIR,
)

# --- Config ----------------------------------------------------------------

# flan-t5-small is instruction-tuned, CPU-friendly, and responsive (~1s/answer)
# on this 8 GB / no-GPU box. Swap to flan-t5-base for higher quality if a GPU is
# available.
GENERATOR_MODEL = "google/flan-t5-small"
DEFAULT_TOP_K = 5
MAX_NEW_TOKENS = 192

PROMPT_TEMPLATE = """You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints. Use the following retrieved complaint excerpts to formulate your answer. If the context doesn't contain the answer, state that you don't have enough information.

Context:
{context}

Question: {question}

Answer:"""


# --- Retrieval -------------------------------------------------------------


@dataclass
class RetrievedChunk:
    text: str
    metadata: dict
    distance: float


class Retriever:
    """Embeds a question and runs a similarity search over the vector store."""

    def __init__(
        self,
        persist_dir: str | Path = VECTOR_STORE_DIR,
        collection_name: str = COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
    ):
        import chromadb
        from sentence_transformers import SentenceTransformer

        self._embedder = SentenceTransformer(embedding_model)
        client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = client.get_collection(collection_name)

    def retrieve(
        self,
        question: str,
        k: int = DEFAULT_TOP_K,
        product_category: str | None = None,
    ) -> list[RetrievedChunk]:
        """Return the top-``k`` chunks, optionally filtered by product category."""
        query_embedding = self._embedder.encode([question]).tolist()
        where = {"product_category": product_category} if product_category else None
        res = self._collection.query(
            query_embeddings=query_embedding, n_results=k, where=where
        )
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res.get("distances", [[None] * len(docs)])[0]
        return [
            RetrievedChunk(text=d, metadata=m, distance=dist)
            for d, m, dist in zip(docs, metas, dists)
        ]


# --- Generation ------------------------------------------------------------


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks into a numbered, source-tagged context block."""
    lines = []
    for i, c in enumerate(chunks, 1):
        m = c.metadata
        tag = f"[{m.get('product_category', '?')} | {m.get('issue', '?')}]"
        lines.append(f"{i}. {tag} {c.text}")
    return "\n".join(lines)


class Generator:
    """Wraps a HuggingFace seq2seq (FLAN-T5) model for grounded generation.

    Loads the model directly rather than via ``pipeline("text2text-generation")``,
    which was removed in transformers 5.x.
    """

    def __init__(
        self,
        model_name: str = GENERATOR_MODEL,
        max_new_tokens: int = MAX_NEW_TOKENS,
        max_input_tokens: int = 1024,
    ):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self._max_new_tokens = max_new_tokens
        self._max_input_tokens = max_input_tokens

    def generate(self, question: str, context: str) -> str:
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_input_tokens,
        )
        output_ids = self._model.generate(
            **inputs, max_new_tokens=self._max_new_tokens
        )
        return self._tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        ).strip()


# --- Pipeline --------------------------------------------------------------


@dataclass
class RAGResult:
    question: str
    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)


class RAGPipeline:
    """End-to-end retrieve -> ground -> generate."""

    def __init__(
        self,
        persist_dir: str | Path = VECTOR_STORE_DIR,
        generator_model: str = GENERATOR_MODEL,
        top_k: int = DEFAULT_TOP_K,
    ):
        self.retriever = Retriever(persist_dir=persist_dir)
        self.generator = Generator(model_name=generator_model)
        self.top_k = top_k

    def answer(
        self, question: str, k: int | None = None, product_category: str | None = None
    ) -> RAGResult:
        chunks = self.retriever.retrieve(
            question, k=k or self.top_k, product_category=product_category
        )
        if not chunks:
            return RAGResult(question, "I don't have enough information.", [])
        context = format_context(chunks)
        answer = self.generator.generate(question, context)
        return RAGResult(question, answer, chunks)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ask the RAG pipeline a question.")
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--product", default=None)
    args = parser.parse_args()

    rag = RAGPipeline()
    result = rag.answer(args.question, k=args.k, product_category=args.product)
    print("\nANSWER:\n", result.answer)
    print("\nSOURCES:")
    for i, c in enumerate(result.sources, 1):
        print(f"  {i}. [{c.metadata.get('product_category')}] {c.text[:120]}...")
