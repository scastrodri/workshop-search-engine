import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TextSearch:

    def __init__(self, text_fields: List[str]):
        self.text_fields = text_fields
        self.matrices = {}
        self.vectorizers = {}

    def fit(self, records: List[Dict[str, Any]], vectorizer_params: Optional[Dict] = None):
        """
        Fit the search engine with a list of records.

        Args:
            records: A list of dictionaries, where each dictionary represents a record with text fields.
            vectorizer_params: A dictionary of parameters to pass to TfidfVectorizer (e.g., {'max_features': 1000}).
        
        Returns:
            None
        """

        vectorizer_params = vectorizer_params or {}

        self.df = pd.DataFrame(records)
        if 'answer' in self.df.columns:
            self.df = self.df.rename(columns={'answer': 'text'})

        for f in self.text_fields:
            cv = TfidfVectorizer(**vectorizer_params)
            X = cv.fit_transform(self.df[f])
            self.matrices[f] = X
            self.vectorizers[f] = cv

    def search(self, query: str, n_results: int = 10, boost: Optional[Dict] = None, filters: Optional[Dict] = None) -> List[Dict]:
        """
        Search for relevant records based on a query.
        
        Args:
            query: The search query (string).
            n_results: Number of top results to return.
            boost: A dictionary mapping text fields to boost factors (e.g., {'title': 2.0}).
            filters: A dictionary of field-value pairs to filter results (e.g., {'category': 'books'}).
        
        Returns:
            A list of dictionaries representing the top matching records, sorted by relevance.
        """
        boost = boost or {}
        filters = filters or {}
        score = np.zeros(len(self.df))

        for f in self.text_fields:
            b = boost.get(f, 1.0)
            q = self.vectorizers[f].transform([query])
            s = cosine_similarity(self.matrices[f], q).flatten()
            score = score + b * s

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

    engine = TextSearch(text_fields=["question", "text"])
    engine.fit(sample_records, vectorizer_params={"stop_words": "english"})

    results = engine.search(
        query="how much does it cost?",
        n_results=2,
        boost={"question": 2.0},
        filters={"course": "de-zoomcamp"}
    )

    assert len(results) <= 2
    assert all(r["course"] == "de-zoomcamp" for r in results)
    print("All checks passed.")
    print(results)