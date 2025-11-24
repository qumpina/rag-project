
import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import time
 
# ==================== DATA STRUCTURES ====================
@dataclass
class RetrievalResult:
    """Результат поиска релевантных фрагментов"""
    question: str
    retrieved_chunks: List[str]
    relevant_chunks: List[str]
    scores: List[float]
    chunk_ids: List[str]
 
# ==================== OPTIMIZED EVALUATION ====================
class Evaluation:
    """Оптимизированная реализация оценки качества"""
   
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        if self.use_gpu:
            try:
                import cupy as cp
                self.cp = cp
                print(" CuPy доступен, используем GPU")
            except ImportError:
                self.use_gpu = False
                print(" CuPy не установлен, используем CPU")
        else:
            print("ℹ Используем CPU для вычислений")
   
    def calculate_hit_at_k(self,
                          true_relevant: List[List[str]],
                          predicted: List[List[str]],
                          k: int = 5) -> float:
        """Оптимизированный Hit@K"""
        print(f"Calculating Hit@{k}...")
       
        if self.use_gpu:
            return self._hit_at_k_gpu(true_relevant, predicted, k)
        else:
            return self._hit_at_k_cpu(true_relevant, predicted, k)
   
    def _hit_at_k_cpu(self, true_relevant, predicted, k):
        """CPU-оптимизированная версия"""
        hits = 0
        total = len(true_relevant)
       
        for i in range(total):
            true_set = set(true_relevant[i])
            pred_top_k = set(predicted[i][:k])
            if true_set & pred_top_k:
                hits += 1
       
        return hits / total if total > 0 else 0.0
   
    def _hit_at_k_gpu(self, true_relevant, predicted, k):
        """GPU-оптимизированная версия"""
        # Векторизованная обработка на GPU
        hit_matrix = []
       
        for i in range(len(true_relevant)):
            true_set = set(true_relevant[i])
            hit_row = [1 if doc in true_set else 0 for doc in predicted[i][:k]]
            hit_matrix.append(hit_row)
       
        # Конвертируем всю матрицу сразу
        hit_tensor = self.cp.array(hit_matrix)
        # Суммируем по строкам (хотя бы один хит)
        row_max = self.cp.max(hit_tensor, axis=1)
        hit_rate = float(self.cp.mean(row_max))
       
        return hit_rate
   
    def calculate_ndcg_at_k(self,
                           true_relevant: List[List[str]],
                           predicted: List[List[str]],
                           k: int = 5) -> float:
        """Оптимизированный NDCG@K"""
        if self.use_gpu:
            return self._ndcg_at_k_gpu(true_relevant, predicted, k)
        else:
            return self._ndcg_at_k_cpu(true_relevant, predicted, k)
   
    def _ndcg_at_k_cpu(self, true_relevant, predicted, k):
        """CPU версия NDCG"""
        ndcg_scores = []
       
        for i in range(len(true_relevant)):
            true_set = set(true_relevant[i])
            pred_list = predicted[i][:k]
           
            dcg = 0.0
            for j, doc in enumerate(pred_list):
                if doc in true_set:
                    dcg += 1.0 / np.log2(j + 2)
           
            # Идеальный DCG
            num_relevant = min(len(true_set), k)
            ideal_dcg = sum(1.0 / np.log2(j + 2) for j in range(num_relevant))
           
            ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
            ndcg_scores.append(ndcg)
       
        return np.mean(ndcg_scores)
   
    def _ndcg_at_k_gpu(self, true_relevant, predicted, k):
        """GPU версия NDCG"""
        ndcg_scores = []
       
        for i in range(len(true_relevant)):
            true_set = set(true_relevant[i])
            pred_list = predicted[i][:k]
           
            # Векторизованное вычисление DCG
            relevance = [1.0 if doc in true_set else 0.0 for doc in pred_list]
            positions = np.arange(1, len(relevance) + 1)
            discounts = 1.0 / np.log2(positions + 1)
           
            dcg = np.sum(np.array(relevance) * discounts)
           
            # Идеальный DCG
            num_relevant = min(len(true_set), k)
            ideal_dcg = np.sum(1.0 / np.log2(np.arange(1, num_relevant + 1) + 1))
           
            ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
            ndcg_scores.append(ndcg)
       
        # Конвертируем все scores сразу
        ndcg_gpu = self.cp.array(ndcg_scores)
        return float(self.cp.mean(ndcg_gpu))
   
    def calculate_mrr(self,
                     true_relevant: List[List[str]],
                     predicted: List[List[str]]) -> float:
        """Оптимизированный MRR"""
        if self.use_gpu:
            return self._mrr_gpu(true_relevant, predicted)
        else:
            return self._mrr_cpu(true_relevant, predicted)
   
    def _mrr_cpu(self, true_relevant, predicted):
        """CPU версия MRR"""
        reciprocal_ranks = []
       
        for i in range(len(true_relevant)):
            true_set = set(true_relevant[i])
            for rank, doc in enumerate(predicted[i], 1):
                if doc in true_set:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                reciprocal_ranks.append(0.0)
       
        return np.mean(reciprocal_ranks)
   
    def _mrr_gpu(self, true_relevant, predicted):
        """GPU версия MRR"""
        reciprocal_ranks = []
       
        for i in range(len(true_relevant)):
            true_set = set(true_relevant[i])
            for rank, doc in enumerate(predicted[i], 1):
                if doc in true_set:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                reciprocal_ranks.append(0.0)
       
        # Конвертируем все ranks сразу
        rr_gpu = self.cp.array(reciprocal_ranks)
        return float(self.cp.mean(rr_gpu))
 

class EvaluationContract:
    """Оптимизированный контракт для оценки качества"""
   
    def __init__(self, use_gpu: bool = True):
        self.evaluator = Evaluation(use_gpu)
       
    def calculate_comprehensive_metrics(self,
                                      results: List[RetrievalResult],
                                      k_values: List[int] = None) -> Dict[str, float]:
        """Вычисление всех метрик за один проход"""
        if k_values is None:
            k_values = [1, 3, 5, 10]
       
        # Подготовка данных один раз
        true_relevant = [result.relevant_chunks for result in results]
        predicted = [result.retrieved_chunks for result in results]
       
        metrics = {}
       
        # Hit@K для разных K
        for k in k_values:
            metrics[f'hit_at_{k}'] = self.evaluator.calculate_hit_at_k(
                true_relevant, predicted, k
            )
       
        # Другие метрики
        metrics['ndcg_at_5'] = self.evaluator.calculate_ndcg_at_k(
            true_relevant, predicted, 5
        )
        metrics['mrr'] = self.evaluator.calculate_mrr(true_relevant, predicted)
       
        return metrics
   
    def generate_optimized_submission(self,
                                    results: List[RetrievalResult],
                                    output_path: str = "submission.csv") -> pd.DataFrame:
        """Оптимизированная генерация submission файла"""
        print(f"Generating optimized submission: {output_path}")
       
        # Векторизованное создание данных
        submission_rows = []
       
        for result in results:
            # Ограничиваем количество чанков для экономии памяти
            top_chunks = result.retrieved_chunks[:10]  # Только топ-10
            top_scores = result.scores[:10]
            top_ids = result.chunk_ids[:10] if result.chunk_ids else []
           
            relevant_set = set(result.relevant_chunks)
           
            for i, (chunk, score) in enumerate(zip(top_chunks, top_scores)):
                chunk_id = top_ids[i] if i < len(top_ids) else f"chunk_{i}"
                is_relevant = 1 if chunk in relevant_set else 0
               
                submission_rows.append({
                    'question': result.question,
                    'retrieved_chunk': chunk[:150] + "..." if len(chunk) > 150 else chunk,
                    'chunk_id': chunk_id,
                    'score': round(score, 4),
                    'is_relevant': is_relevant,
                    'retrieval_rank': i + 1
                })
       
        # Создаем DataFrame одним вызовом
        df = pd.DataFrame(submission_rows)
        df.to_csv(output_path, index=False)
       
        print(f"Submission created: {len(df)} records")
        return df
   
    def analyze_retrieval_quality(self,
                                results: List[RetrievalResult]) -> Dict[str, Any]:
        """Оптимизированный анализ качества"""
        print("Starting retrieval analysis...")
        start_time = time.time()
       
        # Вычисляем все метрики за один проход
        metrics = self.calculate_comprehensive_metrics(results)
       
        # Статистика scores
        all_scores = [score for result in results for score in result.scores[:10]]  # Только топ-10
       
        if self.evaluator.use_gpu:
            scores_tensor = self.evaluator.cp.array(all_scores)
            score_stats = {
                'mean_score': float(self.evaluator.cp.mean(scores_tensor)),
                'std_score': float(self.evaluator.cp.std(scores_tensor)),
                'min_score': float(self.evaluator.cp.min(scores_tensor)),
                'max_score': float(self.evaluator.cp.max(scores_tensor)),
            }
        else:
            score_stats = {
                'mean_score': np.mean(all_scores),
                'std_score': np.std(all_scores),
                'min_score': np.min(all_scores),
                'max_score': np.max(all_scores),
            }
       
        analysis_result = {
            'metrics': metrics,
            'score_statistics': score_stats,
            'computation_time_seconds': time.time() - start_time,
            'total_queries': len(results),
            'gpu_accelerated': self.evaluator.use_gpu,
        }
       
        print(f" Analysis completed in {analysis_result['computation_time_seconds']:.2f}s")
        return analysis_result
