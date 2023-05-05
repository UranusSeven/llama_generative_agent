import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import chromadb
from chromadb.errors import NoIndexException
from langchain.schema import Document
from langchain.vectorstores import Chroma

logger = logging.getLogger(__name__)


def default_relevance_score_fn(score: float):
    import math

    return 1 / (1 + math.exp(-score / 100000)) - 0.5


def _results_to_docs_and_scores(results: Any) -> List[Tuple[Document, float]]:
    return [
        # TODO: Chroma can do batch querying,
        # we shouldn't hard code to the 1st result
        (Document(page_content=result[0], metadata=result[1] or {}), result[2])
        for result in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


class EnhancedChroma(Chroma):
    def __init__(
        self,
        relevance_score_fn: Callable[[float], float] = default_relevance_score_fn,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.relevance_score_fn = relevance_score_fn

    def _similarity_search_with_relevance_scores(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Return docs and their similarity scores on a scale from 0 to 1."""
        if self.relevance_score_fn is None:
            raise ValueError("a relevance_score_fn is required.")
        try:
            docs_and_scores = self.similarity_search_with_score(query, k=k)
            return [
                (doc, self.relevance_score_fn(score)) for doc, score in docs_and_scores
            ]
        except NoIndexException:
            return []

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Run similarity search with Chroma with distance.

        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List[Tuple[Document, float]]: List of documents most similar to the query
                text with distance in float.
        """
        if self._embedding_function is None:
            results = self.__query_collection(
                query_texts=[query], n_results=k, where=filter
            )
        else:
            query_embedding = self._embedding_function.embed_query(query)
            results = self.__query_collection(
                query_embeddings=[query_embedding], n_results=k, where=filter
            )

        return _results_to_docs_and_scores(results)

    def __query_collection(
        self,
        query_texts: Optional[List[str]] = None,
        query_embeddings: Optional[List[List[float]]] = None,
        n_results: int = 4,
        where: Optional[Dict[str, str]] = None,
    ) -> List[Document]:
        """Query the chroma collection."""
        for i in range(n_results, 0, -1):
            try:
                return self._collection.query(
                    query_texts=query_texts,
                    query_embeddings=query_embeddings,
                    n_results=i,
                    where=where,
                )
            except chromadb.errors.NotEnoughElementsException:
                logger.warning(
                    f"Chroma collection {self._collection.name} "
                    f"contains fewer than {i} elements."
                )
        return []
