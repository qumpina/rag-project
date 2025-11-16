# run_pipeline.py

import os
import pandas as pd
import numpy as np

from src.data.data_loader import AlfabankDataProcessor
from src.nlp.embeddings import EmbeddingModel
from src.ml.retrieval import RetrievalFAISS, generate_submission_csv


def main():

    print("\n### STEP 1 — Load and preprocess data")
    processor = AlfabankDataProcessor()

    bundle = processor.build_data_bundle()

    print("Questions:", len(bundle.questions))
    print("Websites:", len(bundle.websites))
    print("Chunks:", len(bundle.chunks))
    print(bundle.chunks.head())


    print("\n### STEP 2 — Load embedding model")
    # !!! Убедитесь, что модель существует локально или качается с HF
    emb_model = EmbeddingModel(
        model_path="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    print("Embedding dimension:", emb_model.get_embedding_dimension())


    print("\n### STEP 3 — Build embeddings for chunks")
    chunk_texts = bundle.chunks["text_chunk"].tolist()
    print(f"Encoding {len(chunk_texts)} chunks…")

    chunk_embeddings = emb_model.get_embeddings(chunk_texts)
    chunk_ids = bundle.chunks["chunk_id"].tolist()

    print("Embeddings shape:", chunk_embeddings.shape)

    # optional: save embeddings for reuse
    np.save("chunk_embeddings.npy", chunk_embeddings)
    bundle.chunks.to_csv("chunks_used.csv", index=False)


    print("\n### STEP 4 — Build FAISS index")
    retriever = RetrievalFAISS(
        chunks_df=bundle.chunks,
        metric="cosine"
    )
    retriever.build_vector_index(chunk_embeddings, chunk_ids)

    print("FAISS index ready!")


    print("\n### STEP 5 — Batch retrieval for all questions")
    questions = bundle.questions.to_dict("records")
    results = retriever.batch_retrieve(questions, emb_model, top_k=5)

    print("Example result:", vars(results[0]))


    print("\n### STEP 6 — Generate submission.csv")
    df_sub = generate_submission_csv(results, "submission.csv")

    print(df_sub.head())
    print("\nSaved → submission.csv")


    print("\n### FINISHED SUCCESSFULLY ###")


if __name__ == "__main__":
    main()
