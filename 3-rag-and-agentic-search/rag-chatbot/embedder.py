"""TF-IDF embedder - replaces VoyageAI's generate_embedding.

VectorIndex only needs a callable: str -> vector, or list[str] -> list[vector],
with a fixed output dimension. VoyageAI supplied that over the network; this
supplies it with plain TF-IDF, fitted once over the whole corpus up front (unlike
VoyageAI, TF-IDF needs corpus-wide document frequencies before it can embed
anything, so fit() must run before VectorIndex(embedding_fn=...) is constructed).
"""

import math
import re
from collections import Counter
from typing import Callable, Dict, List, Optional


def default_tokenizer(text: str) -> List[str]:
    """Lowercase + split on non-word chars, drop empties.

    Deliberately duplicates BM25Index's own default tokenizer rather than sharing
    it, so BM25Index in retrieval.py stays byte-for-byte as ported from the
    notebook while both indexes still tokenize identically for a fair
    lexical/semantic comparison.
    """
    text = text.lower()
    return [token for token in re.split(r"\W+", text) if token]


class TfidfEmbedder:
    """Fits a fixed vocabulary + IDF table on a corpus, then embeds text as
    L2-normalized TF-IDF vectors of that fixed dimension - a plug-in replacement
    for VectorIndex's embedding_fn.
    """

    def __init__(self, tokenizer: Optional[Callable[[str], List[str]]] = None):
        self._tokenizer = tokenizer or default_tokenizer
        self._vocab: Dict[str, int] = {}
        self._idf: List[float] = []

    def fit(self, documents: List[str]) -> "TfidfEmbedder":
        doc_freq: Dict[str, int] = {}
        for doc in documents:
            for term in set(self._tokenizer(doc)):
                doc_freq[term] = doc_freq.get(term, 0) + 1

        # Sorted for a deterministic column order - dict/set iteration order would
        # otherwise make the vector layout depend on incidental hash ordering.
        vocab_terms = sorted(doc_freq)
        self._vocab = {term: i for i, term in enumerate(vocab_terms)}

        n_docs = len(documents)
        # Smoothed IDF (sklearn's TfidfVectorizer(smooth_idf=True) default): the +1s
        # keep every term's weight positive, even one appearing in every document.
        self._idf = [math.log((1 + n_docs) / (1 + doc_freq[term])) + 1 for term in vocab_terms]

        return self

    def embed(self, text):
        is_list = isinstance(text, list)
        texts = text if is_list else [text]
        vectors = [self._vectorize(t) for t in texts]
        return vectors if is_list else vectors[0]

    def _vectorize(self, text: str) -> List[float]:
        counts = Counter(self._tokenizer(text))
        vector = [0.0] * len(self._vocab)

        for term, count in counts.items():
            idx = self._vocab.get(term)
            if idx is None:
                continue  # out-of-vocabulary term - skip, never resize the vector
            # Sublinear (log-scaled) TF dampens a term repeated many times in one
            # chunk from dominating the vector.
            tf = 1 + math.log(count)
            vector[idx] = tf * self._idf[idx]

        # L2-normalize. Redundant with VectorIndex's default cosine distance (already
        # magnitude-invariant) but kept for conventional TF-IDF correctness, and so
        # this embedder behaves sanely if ever paired with distance_metric="euclidean".
        norm = math.sqrt(sum(x * x for x in vector))
        if norm > 0:
            vector = [x / norm for x in vector]

        return vector
