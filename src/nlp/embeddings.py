import numpy as np
from typing import List,Dict,Any
from sentence_transformers import SentenceTransformer
from contracts import EmbeddingContract
from src.data.data_loader import DataLoader


class EmbeddingModel(EmbeddingContract):
    def __init__(self, model_path: str = "./sentence-transformer"):
        data_loader = DataLoader()
        self.model: SentenceTransformer = data_loader.get_free_embedding_model(model_path)

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        embeddings = self.model.encode(texts)
        return embeddings

    def get_embedding_dimension(self) -> int:
        if not hasattr(self, 'model') or self.model is None:
            raise ValueError("Model loading error")
        return self.model.get_sentence_embedding_dimension()

    def get_model_info(self) -> Dict[str, Any]:
        if not hasattr(self, 'model') or self.model is None:
            raise ValueError("Model loading error")

        try:
            model = self.model
            model_path = getattr(model, 'model_path', None)

            info = {
                # ������� ����������
                "model_type": "SentenceTransformer",
                "model_name": getattr(model, 'model_name', 'Unknown'),
                "model_path": model_path,

                # ����������� ��������������
                "embedding_dimension": model.get_sentence_embedding_dimension(),
                "max_sequence_length": model.get_max_seq_length(),
                "normalize_embeddings": getattr(model, '_modules', {}).get('2', None) is not None,
                # Check if Normalize layer exists

                # ����������� ������
                "modules": [],
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "total_parameters": sum(p.numel() for p in model.parameters()),

                # ���������� � ������� �������
                "device": str(model.device),
                "dtype": str(next(model.parameters()).dtype),
                "model_size_mb": self._get_model_size(model_path) if model_path else 'Unknown',

                # ���������� � �����������
                "tokenizer_info": {
                    "type": type(model.tokenizer).__name__,
                    "vocab_size": len(model.tokenizer),
                    "padding_token": getattr(model.tokenizer, 'pad_token', None),
                    "unknown_token": getattr(model.tokenizer, 'unk_token', None),
                } if hasattr(model, 'tokenizer') else {},

                # ���������� � ��������������
                "supported_tasks": ["text-embeddings", "semantic-similarity", "clustering"],
                "sklearn_compatible": True,
            }
            return info
        except Exception as e:
            return {"error": f"Failed to get model info: {str(e)}"}


