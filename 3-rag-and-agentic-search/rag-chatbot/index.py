"""Builds the hybrid retriever once at import time from corpus.DOCUMENTS.

Fit order matters: TfidfEmbedder.fit() needs corpus-wide document frequencies, so
it must run before VectorIndex(embedding_fn=tfidf.embed) is constructed - the one
place this differs from 005_hybrid.ipynb's build order (VoyageAI needed no pre-fit
step, since it embeds per call over the network).
"""

from typing import Dict, List

from corpus import DOCUMENTS
from embedder import TfidfEmbedder
from retrieval import BM25Index, Retriever, VectorIndex, chunk_by_section


def _build_corpus() -> List[Dict[str, str]]:
    """Flatten DOCUMENTS into {"content", "source", "section"} chunk dicts.

    "source"/"section" ride along purely for citation/debug display - they're never
    passed to the embedding function, which only ever sees "content".
    """
    chunks = []
    for title, text in DOCUMENTS.items():
        sections = chunk_by_section(text)
        for i, section in enumerate(sections):
            section_name = "Overview" if i == 0 else section.split("\n", 1)[0].strip()
            chunks.append({"content": section.strip(), "source": title, "section": section_name})
    return chunks


CHUNKS = _build_corpus()

tfidf = TfidfEmbedder()
tfidf.fit([c["content"] for c in CHUNKS])  # must precede VectorIndex construction below

vector_index = VectorIndex(embedding_fn=tfidf.embed)
bm25_index = BM25Index()
retriever = Retriever(bm25_index, vector_index)
retriever.add_documents(CHUNKS)
