import numpy as np
import pandas as pd
import warnings
import torch
from utils import make_batches
from typing import List, Dict, Any, Optional
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm


class SemanticSearch:

    def __init__(self, text_fields: List[str], batch_size: int = 4):
        """
        Args:
            text_fields: List of text fields to embed and search over.
            batch_size: Number of documents per batch. Keep low (4 or less) on CPU.
        """
        self.text_fields = text_fields
        self.batch_size = batch_size
        self.matrices = {}
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.eval()
        warnings.warn("SemanticSearch is slow on CPU. Consider using a GPU for larger datasets.")

    def fit(self, records: List[Dict[str, Any]]) -> None:
        """
        Compute BERT embeddings for all documents.
        Warning: slow on CPU for large datasets.

        Args:
            records: List of dictionaries with text fields.
        """
        self.df = pd.DataFrame(records)
        if 'answer' in self.df.columns:
            self.df = self.df.rename(columns={'answer': 'text'})
        for field in self.text_fields:
            text_batches = make_batches(self.df[field].tolist(), self.batch_size)
            all_embeddings = []
            for batch in tqdm(text_batches, desc=f"Embedding '{field}'"):
                encoded_input = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt')

                with torch.no_grad():
                    outputs = self.model(**encoded_input)
                    hidden_states = outputs.last_hidden_state
                    batch_embeddings = hidden_states.mean(dim=1)
                    batch_embeddings_np = batch_embeddings.cpu().numpy()
                    all_embeddings.append(batch_embeddings_np)
            self.matrices[field] = np.vstack(all_embeddings)       


    def search(self, query: str, n_results: int = 10, boost: Optional[Dict] = None, filters: Optional[Dict] = None) -> List[Dict]:
        """
        Search using cosine similarity over BERT embeddings.

        Args:
            query: Search query string.
            n_results: Number of results to return.
            boost: Field boost weights (e.g., {'question': 2.0}).
            filters: Field-value pairs to filter results (e.g., {'course': 'de-zoomcamp'}).

        Returns:
            List of matching records sorted by relevance.
        """
        boost = boost or {}
        filters = filters or {}
        score = np.zeros(len(self.df))

        encoded_q = self.tokenizer([query], padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**encoded_q)
            Q_emb = outputs.last_hidden_state.mean(dim=1).numpy()

        for field in self.text_fields:
            scores = cosine_similarity(self.matrices[field], Q_emb).flatten()
            score = score + boost.get(field, 1.0) * scores

        for field, value in filters.items():
            mask = (self.df[field] == value).values
            score = score * mask

        idx = np.argsort(-score)[:n_results]
        results = self.df.iloc[idx]
        return results.to_dict(orient='records')
    

if __name__ == "__main__":
    sample_records = [
        {"question": "When does the course start?", "text": "The course starts in January.", "course": "de-zoomcamp"},
        {"question": "What are the prerequisites?", "text": "You need basic Python knowledge.", "course": "de-zoomcamp"},
        {"question": "Is it free?", "text": "Yes, the course is completely free.", "course": "ml-zoomcamp"},
    ]

    engine_semantic = SemanticSearch(text_fields=["question", "text"], batch_size=4)
    engine_semantic.fit(sample_records)

    results_semantic = engine_semantic.search(
        query="how much does it cost?",
        n_results=2,
        boost={"question": 2.0},
        filters={"course": "de-zoomcamp"}
    )

    assert len(results_semantic) <= 2
    assert all(r["course"] == "de-zoomcamp" for r in results_semantic)
    print("All checks passed.")
    print("BERT results:", results_semantic)