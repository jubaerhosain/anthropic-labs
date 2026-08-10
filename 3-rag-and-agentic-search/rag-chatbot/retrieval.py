"""Hybrid retrieval primitives - ported from 005_hybrid.ipynb, unmodified.

chunk_by_section splits a document on markdown H2 headers. VectorIndex (semantic)
and BM25Index (lexical) are pure-Python indexes sharing an add_document /
add_documents / search interface (see SearchIndex). Retriever fuses any number of
such indexes via Reciprocal Rank Fusion.

The notebook's generate_embedding (VoyageAI) is dropped - embedder.TfidfEmbedder
fills the same embedding_fn slot in VectorIndex without an API call.
"""

import math
import re
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple


def chunk_by_section(document_text: str) -> List[str]:
    pattern = r"\n## "
    return re.split(pattern, document_text)


class SearchIndex(Protocol):
    def add_document(self, document: Dict[str, Any]) -> None: ...

    def add_documents(self, documents: List[Dict[str, Any]]) -> None: ...

    def search(self, query: Any, k: int = 1) -> List[Tuple[Dict[str, Any], float]]: ...


class VectorIndex:
    def __init__(
        self,
        distance_metric: str = "cosine",
        embedding_fn=None,
    ):
        self.vectors: List[List[float]] = []
        self.documents: List[Dict[str, Any]] = []
        self._vector_dim: Optional[int] = None
        if distance_metric not in ["cosine", "euclidean"]:
            raise ValueError("distance_metric must be 'cosine' or 'euclidean'")
        self._distance_metric = distance_metric
        self._embedding_fn = embedding_fn

    def add_document(self, document: Dict[str, Any]):
        if not self._embedding_fn:
            raise ValueError("Embedding function not provided during initialization.")
        if not isinstance(document, dict):
            raise TypeError("Document must be a dictionary.")
        if "content" not in document:
            raise ValueError("Document dictionary must contain a 'content' key.")

        content = document["content"]
        if not isinstance(content, str):
            raise TypeError("Document 'content' must be a string.")

        vector = self._embedding_fn(content)
        self.add_vector(vector=vector, document=document)

    def add_documents(self, documents: List[Dict[str, Any]]):
        if not self._embedding_fn:
            raise ValueError("Embedding function not provided during initialization.")

        if not isinstance(documents, list):
            raise TypeError("Documents must be a list of dictionaries.")

        if not documents:
            return

        contents = []
        for i, doc in enumerate(documents):
            if not isinstance(doc, dict):
                raise TypeError(f"Document at index {i} must be a dictionary.")
            if "content" not in doc:
                raise ValueError(f"Document at index {i} must contain a 'content' key.")
            if not isinstance(doc["content"], str):
                raise TypeError(f"Document 'content' at index {i} must be a string.")
            contents.append(doc["content"])

        vectors = self._embedding_fn(contents)

        for vector, document in zip(vectors, documents):
            self.add_vector(vector=vector, document=document)

    def search(self, query: Any, k: int = 1) -> List[Tuple[Dict[str, Any], float]]:
        if not self.vectors:
            return []

        if isinstance(query, str):
            if not self._embedding_fn:
                raise ValueError("Embedding function not provided for string query.")
            query_vector = self._embedding_fn(query)
        elif isinstance(query, list) and all(isinstance(x, (int, float)) for x in query):
            query_vector = query
        else:
            raise TypeError("Query must be either a string or a list of numbers.")

        if self._vector_dim is None:
            return []

        if len(query_vector) != self._vector_dim:
            raise ValueError(
                f"Query vector dimension mismatch. Expected {self._vector_dim}, got {len(query_vector)}"
            )

        if k <= 0:
            raise ValueError("k must be a positive integer.")

        if self._distance_metric == "cosine":
            dist_func = self._cosine_distance
        else:
            dist_func = self._euclidean_distance

        distances = []
        for i, stored_vector in enumerate(self.vectors):
            distance = dist_func(query_vector, stored_vector)
            distances.append((distance, self.documents[i]))

        distances.sort(key=lambda item: item[0])

        return [(doc, dist) for dist, doc in distances[:k]]

    def add_vector(self, vector, document: Dict[str, Any]):
        if not isinstance(vector, list) or not all(isinstance(x, (int, float)) for x in vector):
            raise TypeError("Vector must be a list of numbers.")
        if not isinstance(document, dict):
            raise TypeError("Document must be a dictionary.")
        if "content" not in document:
            raise ValueError("Document dictionary must contain a 'content' key.")

        if not self.vectors:
            self._vector_dim = len(vector)
        elif len(vector) != self._vector_dim:
            raise ValueError(
                f"Inconsistent vector dimension. Expected {self._vector_dim}, got {len(vector)}"
            )

        self.vectors.append(list(vector))
        self.documents.append(document)

    def _euclidean_distance(self, vec1: List[float], vec2: List[float]) -> float:
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same dimension")
        return math.sqrt(sum((p - q) ** 2 for p, q in zip(vec1, vec2)))

    def _dot_product(self, vec1: List[float], vec2: List[float]) -> float:
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same dimension")
        return sum(p * q for p, q in zip(vec1, vec2))

    def _magnitude(self, vec: List[float]) -> float:
        return math.sqrt(sum(x * x for x in vec))

    def _cosine_distance(self, vec1: List[float], vec2: List[float]) -> float:
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same dimension")

        mag1 = self._magnitude(vec1)
        mag2 = self._magnitude(vec2)

        if mag1 == 0 and mag2 == 0:
            return 0.0
        elif mag1 == 0 or mag2 == 0:
            return 1.0

        dot_prod = self._dot_product(vec1, vec2)
        cosine_similarity = dot_prod / (mag1 * mag2)
        cosine_similarity = max(-1.0, min(1.0, cosine_similarity))

        return 1.0 - cosine_similarity

    def __len__(self) -> int:
        return len(self.vectors)

    def __repr__(self) -> str:
        has_embed_fn = "Yes" if self._embedding_fn else "No"
        return f"VectorIndex(count={len(self)}, dim={self._vector_dim}, metric='{self._distance_metric}', has_embedding_fn='{has_embed_fn}')"


class BM25Index:
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
    ):
        self.documents: List[Dict[str, Any]] = []
        self._corpus_tokens: List[List[str]] = []
        self._doc_len: List[int] = []
        self._doc_freqs: Dict[str, int] = {}
        self._avg_doc_len: float = 0.0
        self._idf: Dict[str, float] = {}
        self._index_built: bool = False

        self.k1 = k1
        self.b = b
        self._tokenizer = tokenizer if tokenizer else self._default_tokenizer

    def _default_tokenizer(self, text: str) -> List[str]:
        text = text.lower()
        tokens = re.split(r"\W+", text)
        return [token for token in tokens if token]

    def _update_stats_add(self, doc_tokens: List[str]):
        self._doc_len.append(len(doc_tokens))

        seen_in_doc = set()
        for token in doc_tokens:
            if token not in seen_in_doc:
                self._doc_freqs[token] = self._doc_freqs.get(token, 0) + 1
                seen_in_doc.add(token)

        self._index_built = False

    def _calculate_idf(self):
        N = len(self.documents)
        self._idf = {}
        for term, freq in self._doc_freqs.items():
            idf_score = math.log(((N - freq + 0.5) / (freq + 0.5)) + 1)
            self._idf[term] = idf_score

    def _build_index(self):
        if not self.documents:
            self._avg_doc_len = 0.0
            self._idf = {}
            self._index_built = True
            return

        self._avg_doc_len = sum(self._doc_len) / len(self.documents)
        self._calculate_idf()
        self._index_built = True

    def add_document(self, document: Dict[str, Any]):
        if not isinstance(document, dict):
            raise TypeError("Document must be a dictionary.")
        if "content" not in document:
            raise ValueError("Document dictionary must contain a 'content' key.")

        content = document.get("content", "")
        if not isinstance(content, str):
            raise TypeError("Document 'content' must be a string.")

        doc_tokens = self._tokenizer(content)

        self.documents.append(document)
        self._corpus_tokens.append(doc_tokens)
        self._update_stats_add(doc_tokens)

    def add_documents(self, documents: List[Dict[str, Any]]):
        if not isinstance(documents, list):
            raise TypeError("Documents must be a list of dictionaries.")

        if not documents:
            return

        for i, doc in enumerate(documents):
            if not isinstance(doc, dict):
                raise TypeError(f"Document at index {i} must be a dictionary.")
            if "content" not in doc:
                raise ValueError(f"Document at index {i} must contain a 'content' key.")
            if not isinstance(doc["content"], str):
                raise TypeError(f"Document 'content' at index {i} must be a string.")

            content = doc["content"]
            doc_tokens = self._tokenizer(content)

            self.documents.append(doc)
            self._corpus_tokens.append(doc_tokens)
            self._update_stats_add(doc_tokens)

        self._index_built = False

    def _compute_bm25_score(self, query_tokens: List[str], doc_index: int) -> float:
        score = 0.0
        doc_term_counts = Counter(self._corpus_tokens[doc_index])
        doc_length = self._doc_len[doc_index]

        for token in query_tokens:
            if token not in self._idf:
                continue

            idf = self._idf[token]
            term_freq = doc_term_counts.get(token, 0)

            numerator = idf * term_freq * (self.k1 + 1)
            denominator = term_freq + self.k1 * (1 - self.b + self.b * (doc_length / self._avg_doc_len))
            score += numerator / (denominator + 1e-9)

        return score

    def search(
        self,
        query: Any,
        k: int = 1,
        score_normalization_factor: float = 0.1,
    ) -> List[Tuple[Dict[str, Any], float]]:
        if not self.documents:
            return []

        if isinstance(query, str):
            query_text = query
        else:
            raise TypeError("Query must be a string for BM25Index.")

        if k <= 0:
            raise ValueError("k must be a positive integer.")

        if not self._index_built:
            self._build_index()

        if self._avg_doc_len == 0:
            return []

        query_tokens = self._tokenizer(query_text)
        if not query_tokens:
            return []

        raw_scores = []
        for i in range(len(self.documents)):
            raw_score = self._compute_bm25_score(query_tokens, i)
            if raw_score > 1e-9:
                raw_scores.append((raw_score, self.documents[i]))

        raw_scores.sort(key=lambda item: item[0], reverse=True)

        normalized_results = []
        for raw_score, doc in raw_scores[:k]:
            normalized_score = math.exp(-score_normalization_factor * raw_score)
            normalized_results.append((doc, normalized_score))

        normalized_results.sort(key=lambda item: item[1])

        return normalized_results

    def __len__(self) -> int:
        return len(self.documents)

    def __repr__(self) -> str:
        return f"BM25VectorStore(count={len(self)}, k1={self.k1}, b={self.b}, index_built={self._index_built})"


class Retriever:
    def __init__(self, *indexes: SearchIndex):
        if len(indexes) == 0:
            raise ValueError("At least one index must be provided")
        self._indexes = list(indexes)

    def add_document(self, document: Dict[str, Any]):
        for index in self._indexes:
            index.add_document(document)

    def add_documents(self, documents: List[Dict[str, Any]]):
        for index in self._indexes:
            index.add_documents(documents)

    def search(self, query_text: str, k: int = 1, k_rrf: int = 60) -> List[Tuple[Dict[str, Any], float]]:
        if not isinstance(query_text, str):
            raise TypeError("Query text must be a string.")
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        if k_rrf < 0:
            raise ValueError("k_rrf must be non-negative.")

        all_results = [index.search(query_text, k=k * 5) for index in self._indexes]

        doc_ranks = {}
        for idx, results in enumerate(all_results):
            for rank, (doc, _) in enumerate(results):
                doc_id = id(doc)
                if doc_id not in doc_ranks:
                    doc_ranks[doc_id] = {
                        "doc_obj": doc,
                        "ranks": [float("inf")] * len(self._indexes),
                    }
                doc_ranks[doc_id]["ranks"][idx] = rank + 1

        def calc_rrf_score(ranks: List[float]) -> float:
            return sum(1.0 / (k_rrf + r) for r in ranks if r != float("inf"))

        scored_docs: List[Tuple[Dict[str, Any], float]] = [
            (ranks["doc_obj"], calc_rrf_score(ranks["ranks"])) for ranks in doc_ranks.values()
        ]

        filtered_docs = [(doc, score) for doc, score in scored_docs if score > 0]
        filtered_docs.sort(key=lambda x: x[1], reverse=True)

        return filtered_docs[:k]
