from datetime import datetime
from typing import List, Any, Optional
from copy import deepcopy

from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.schema import Document


def _get_hours_passed(time: datetime, ref_time: datetime | str) -> float:
    """Get the hours passed between two datetime objects."""
    if isinstance(ref_time, str):
        ref_time = datetime.fromisoformat(ref_time)
    return (time - ref_time).total_seconds() / 3600


class LlamaTimeWeightedVectorStoreRetriever(TimeWeightedVectorStoreRetriever):
    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """Add documents to vectorstore."""
        current_time = kwargs.get("current_time", datetime.now())
        # Avoid mutating input documents
        dup_docs = [deepcopy(d) for d in documents]
        for i, doc in enumerate(dup_docs):
            if "last_accessed_at" not in doc.metadata:
                doc.metadata["last_accessed_at"] = str(current_time)
            if "created_at" not in doc.metadata:
                doc.metadata["created_at"] = str(current_time)
            doc.metadata["buffer_idx"] = len(self.memory_stream) + i
        self.memory_stream.extend(dup_docs)
        return self.vectorstore.add_documents(dup_docs, **kwargs)

    def _get_combined_score(
        self,
        document: Document,
        vector_relevance: Optional[float],
        current_time: datetime,
    ) -> float:
        """Return the combined score for a document."""
        hours_passed = _get_hours_passed(
            current_time,
            document.metadata["last_accessed_at"],
        )
        score = (1.0 - self.decay_rate) ** hours_passed
        for key in self.other_score_keys:
            if key in document.metadata:
                score += document.metadata[key]
        if vector_relevance is not None:
            score += vector_relevance
        return score
