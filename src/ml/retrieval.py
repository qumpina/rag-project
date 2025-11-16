# src/ml/retrieval.py
from typing import List, Tuple, Any, Dict
import numpy as np
import faiss
import pandas as pd
from contracts import RetrievalContract, RetrievalResult, EmbeddingContract

class RetrievalFAISS(RetrievalContract):
    def __init__(self, chunks_df: pd.DataFrame = None, metric: str = "cosine"):
        """
        chunks_df: DataFrame с колонками ['chunk_id', 'web_id', 'text_chunk', ...]
        metric: "cosine" or "l2"
        """
        self.chunks_df = chunks_df
        self.chunk_ids: List[str] = []
        self.chunk_to_web: Dict[str, str] = {}
        self.index = None
        self.emb_dim = None
        self.metric = metric

    def build_vector_index(self, embeddings: np.ndarray, chunk_ids: List[str]) -> Any:
        """
        embeddings: np.ndarray shape (N, D)
        chunk_ids: list of length N
        """
        if embeddings is None or len(embeddings) == 0:
            raise ValueError("Empty embeddings")

        # Save metadata
        self.chunk_ids = list(chunk_ids)
        if self.chunks_df is not None:
            # build mapping chunk_id -> web_id (fall back to chunk_id parsing if missing)
            df = self.chunks_df.set_index('chunk_id')
            self.chunk_to_web = df['web_id'].to_dict()
        else:
            self.chunk_to_web = {}

        # dimension
        N, D = embeddings.shape
        self.emb_dim = D

        # Normalization for cosine
        if self.metric == "cosine":
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1e-12
            embeddings = embeddings / norms

        # Build FAISS index (IndexFlatIP for cosine after normalization, or IndexFlatL2)
        if self.metric == "cosine":
            index = faiss.IndexFlatIP(D)
        else:
            index = faiss.IndexFlatL2(D)

        # convert to float32
        embeddings = embeddings.astype('float32')
        index.add(embeddings)
        self.index = index
        return index

    def search_similar(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        if self.index is None:
            raise RuntimeError("Index not built. Call build_vector_index first.")
        if query_embedding is None: 
            return []

        q = query_embedding.reshape(1, -1).astype('float32')
        # normalize query if cosine
        if self.metric == "cosine":
            norm = np.linalg.norm(q, axis=1, keepdims=True)
            norm[norm == 0] = 1e-12
            q = q / norm

        D, I = self.index.search(q, top_k)
        scores = D[0].tolist()
        idxs = I[0].tolist()

        results = []
        for idx, score in zip(idxs, scores):
            if idx < 0 or idx >= len(self.chunk_ids):
                continue
            chunk_id = self.chunk_ids[idx]
            results.append((chunk_id, float(score)))
        return results

    def _aggregate_chunks_to_webids(self, chunk_results: List[Tuple[str, float]], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        chunk_results: list of (chunk_id, score)
        returns list of (web_id, aggregated_score) sorted desc, length up to top_k
        Aggregation strategy: take max score per web_id
        """
        web_scores: Dict[str, float] = {}
        for chunk_id, score in chunk_results:
            web_id = self.chunk_to_web.get(chunk_id, chunk_id)  # fallback to chunk_id if mapping missing
            max_score = web_scores.get(web_id, -999)
            web_scores[web_id] = max(max_score, score) + 0.05
        sorted_webs = sorted(web_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_webs[:top_k]

    def retrieve_for_question(self, question: str, embedding_model: EmbeddingContract, top_k: int = 5) -> List[Tuple[str, float]]:
        # get embedding for question
        q_emb = embedding_model.get_embeddings([question])
        if q_emb is None or len(q_emb) == 0:
            return []
        chunk_hits = self.search_similar(q_emb[0], top_k=50)  # search more chunks to increase chance разных web_id
        web_hits = self._aggregate_chunks_to_webids(chunk_hits, top_k=top_k)
        return web_hits

    def batch_retrieve(self, questions: List[Dict], embedding_model: EmbeddingContract, top_k: int = 5) -> List[RetrievalResult]:
        """
        questions: list of dicts or objects with fields 'q_id' and 'query'.
        Returns list of RetrievalResult
        """
        # Build all query embeddings in batch for speed
        queries = [q['query'] if isinstance(q, dict) else q.query for q in questions]
        q_ids = [q['q_id'] if isinstance(q, dict) else q.q_id for q in questions]
        q_embs = embedding_model.get_embeddings(queries)  # shape (M, D)

        results = []
        for q_id, q_emb in zip(q_ids, q_embs):
            chunk_hits = self.search_similar(q_emb, top_k=50)
            web_hits = self._aggregate_chunks_to_webids(chunk_hits, top_k=top_k)
            top_docs = [wid for wid, score in web_hits]
            scores = [score for wid, score in web_hits]
            results.append(RetrievalResult(q_id=q_id, top_docs=top_docs, scores=scores))
        return results

def generate_submission_csv(results, out_path="submission.csv"):
    """
    Формат:
    q_id, web_list
    1, "[1, 22, 55, 99, 101]"
    """
    rows = []
    for r in results:
        web_list = r.top_docs[:5] + [""] * max(0, 5 - len(r.top_docs))

        web_list_str = "[" + ", ".join(str(x) for x in web_list if x != "") + "]"

        rows.append([r.q_id, web_list_str])

    df = pd.DataFrame(rows, columns=["q_id", "web_list"])
    df.to_csv(out_path, index=False)
    return df
