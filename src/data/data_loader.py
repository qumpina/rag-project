import pandas as pd
import numpy as np
import hashlib
from typing import Dict, Any, List
from contracts import DataContract, DataBundle, Document, Question, Chunk

from typing import List

from sentence_transformers import SentenceTransformer
from contracts import EmbeddingContract
class AlfabankDataProcessor(DataContract):
    def __init__(self, data_path: str = "."):
        self.data_path = data_path

    def load_questions(self, file_path: str = "src/data/raw/questions_clean.csv") -> pd.DataFrame:
        try:
            if self.data_path == ".":
                full_path = file_path
            else:
                full_path = f"{self.data_path}/{file_path}"

            df = pd.read_csv(full_path)
            print(f" questions_clean : {df.shape}")
            return df
        except FileNotFoundError:
            print(f"  {file_path}  ")
            return pd.DataFrame()
        except Exception as e:
            print(f"   {file_path}: {e}")
            return pd.DataFrame()

    def load_websites(self, file_path: str = "src/data/raw/websites_updated.csv") -> pd.DataFrame:
        try:
            if self.data_path == ".":
                full_path = file_path
            else:
                full_path = f"{self.data_path}/{file_path}"

            df = pd.read_csv(full_path)
            print(f" websites_updated : {df.shape}")
            return df
        except FileNotFoundError:
            print(f"  {file_path}  ")
            return pd.DataFrame()
        except Exception as e:
            print(f"   {file_path}: {e}")
            return pd.DataFrame()

    def preprocess_text(self, text: str) -> str:
        if pd.isna(text):
            return ""

        text_str = str(text)
        #  
        text_clean = text_str.strip()
        return text_clean

    def chunk_documents(self, df: pd.DataFrame, chunk_size: int = 512, overlap: int = 50) -> pd.DataFrame:
        if df.empty or 'text' not in df.columns:
            print("    ")
            return pd.DataFrame()

        chunks_data = []
        chunk_counter = 0

        for idx, row in df.iterrows():
            text_content = str(row.get('text', ''))
            text_length = len(text_content)
            web_id = row.get('web_id', f'doc_{idx}')

            if text_length == 0:
                continue

            #    
            if text_length <= chunk_size:
                chunks = [text_content]
            else:
                chunks = []
                start = 0
                while start < text_length:
                    end = start + chunk_size
                    chunk = text_content[start:end]
                    chunks.append(chunk)
                    start += chunk_size - overlap
                    if start >= text_length:
                        break

            #  
            for i, chunk_text in enumerate(chunks):
                chunk_id = f"{web_id}_chunk_{i}"
                chunks_data.append({
                    'chunk_id': chunk_id,
                    'web_id': web_id,
                    'text_chunk': chunk_text,
                    'chunk_length': len(chunk_text),
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                })
                chunk_counter += 1

        print(f"  : {chunk_counter}")
        return pd.DataFrame(chunks_data)

    def build_data_bundle(self, questions_path: str = "src/data/raw/questions_clean.csv",
                          websites_path: str = "src/data/raw/websites_updated.csv") -> DataBundle:
        print("  DataBundle  RAG ")

        #  
        questions_df = self.load_questions(questions_path)
        websites_df = self.load_websites(websites_path)

        if questions_df.empty or websites_df.empty:
            print("    ")
            return DataBundle(
                questions=pd.DataFrame(),
                websites=pd.DataFrame(),
                chunks=pd.DataFrame()
            )

        #  
        questions_df = self._preprocess_questions(questions_df)
        websites_df = self._preprocess_websites(websites_df)

        #  
        chunks_df = self.chunk_documents(websites_df)

        print(f" DataBundle :")
        print(f"   - : {len(questions_df)}")
        print(f"   - : {len(websites_df)}")
        print(f"   - : {len(chunks_df)}")

        return DataBundle(
            questions=questions_df,
            websites=websites_df,
            chunks=chunks_df
        )

    def validate_data_quality(self) -> Dict[str, Any]:
        print("\n   ")
        print("=" * 50)

        quality_report = {}

        #    
        questions_df = self.load_questions()
        websites_df = self.load_websites()

        #  questions_clean
        if not questions_df.empty:
            q_issues = self._validate_questions(questions_df)
            quality_report['questions'] = q_issues

        #  websites_updated
        if not websites_df.empty:
            w_issues = self._validate_websites(websites_df)
            quality_report['websites'] = w_issues

        #  
        total_issues = sum(len(issues) for issues in quality_report.values())
        quality_report['summary'] = {
            'total_issues': total_issues,
            'status': 'PASS' if total_issues == 0 else 'WARNING'
        }

        print(f"  : {quality_report['summary']['status']}")
        print(f"   -  : {total_issues}")

        return quality_report

    def _preprocess_questions(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()

        #  
        df_clean = df_clean.fillna('')

        #   
        if 'query' in df_clean.columns:
            df_clean['query'] = df_clean['query'].apply(self.preprocess_text)

        #  
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        final_rows = len(df_clean)

        if initial_rows != final_rows:
            print(f"   -   : {initial_rows - final_rows}")

        return df_clean

    def _preprocess_websites(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()

        #  
        df_clean = df_clean.fillna('')

        #   
        text_columns = ['text', 'title', 'url', 'kind']
        for col in text_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].apply(self.preprocess_text)

        #  
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        final_rows = len(df_clean)

        if initial_rows != final_rows:
            print(f"   -   : {initial_rows - final_rows}")

        return df_clean

    def _validate_questions(self, df: pd.DataFrame) -> Dict[str, Any]:
        issues = {}

        #   
        required_columns = ['q_id', 'query']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            issues['missing_columns'] = missing_columns

        #   
        missing_values = df.isnull().sum().to_dict()
        if any(missing_values.values()):
            issues['missing_values'] = missing_values

        #  
        duplicate_q_ids = df['q_id'].duplicated().sum() if 'q_id' in df.columns else 0
        duplicate_queries = df['query'].duplicated().sum() if 'query' in df.columns else 0

        if duplicate_q_ids > 0:
            issues['duplicate_q_ids'] = duplicate_q_ids
        if duplicate_queries > 0:
            issues['duplicate_queries'] = duplicate_queries

        #  
        print(f"  questions_clean:")
        print(f"   -  : {sum(missing_values.values())}")
        print(f"   -  q_id: {duplicate_q_ids}")
        print(f"   -  query: {duplicate_queries}")

        return issues

    def _validate_websites(self, df: pd.DataFrame) -> Dict[str, Any]:
        issues = {}

        #   
        required_columns = ['web_id', 'text']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            issues['missing_columns'] = missing_columns

        #   
        missing_values = df.isnull().sum().to_dict()
        if any(missing_values.values()):
            issues['missing_values'] = missing_values

        #  
        duplicate_web_ids = df['web_id'].duplicated().sum() if 'web_id' in df.columns else 0

        #     
        if 'text' in df.columns:
            df_temp = df.copy()
            df_temp['text_hash'] = df_temp['text'].apply(
                lambda x: hashlib.md5(str(x).encode()).hexdigest()
            )
            duplicate_texts = df_temp['text_hash'].duplicated().sum()
            if duplicate_texts > 0:
                issues['duplicate_texts'] = duplicate_texts

        if duplicate_web_ids > 0:
            issues['duplicate_web_ids'] = duplicate_web_ids

        #   
        if 'text' in df.columns:
            text_lengths = df['text'].str.len()
            empty_texts = (text_lengths == 0).sum()
            if empty_texts > 0:
                issues['empty_texts'] = empty_texts

            print(f"  websites_updated:")
            print(f"   -  : {sum(missing_values.values())}")
            print(f"   -  web_id: {duplicate_web_ids}")
            print(f"   -  : {issues.get('duplicate_texts', 0)}")
            print(f"   -  : {empty_texts}")
            print(f"   -   : {text_lengths.mean():.1f} ")

        return issues

class DataLoader(EmbeddingContract):
    # ??????????????, ??? ????? ? ??????? ?????????? sentence-transformer ? ??????????? ? ????? ???????
    def get_free_embedding_model(self, model_path="./sentence-transformer"):
        return SentenceTransformer(model_path)

    def get_available_models(self) -> List[str]:
        return ["ai-forever/sbert_large_nlu_ru"]