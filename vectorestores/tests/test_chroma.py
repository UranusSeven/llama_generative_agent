import pytest
from chroma import EnhancedChroma
from langchain.embeddings import LlamaCppEmbeddings


def test():
    local_path = "/Users/jon/Downloads/ggml-vicuna-13b-1.1-q4_2.bin"
    embeddings = LlamaCppEmbeddings(model_path=local_path)

    vs = EnhancedChroma.from_texts([], embedding=embeddings)

    docs = vs.similarity_search_with_score("how does tommie feel?", k=1)
    print(docs)


def test_default_relevance_score_fn():
    print(default_relevance_score_fn(14000.0))
    print(default_relevance_score_fn(0))
    print(default_relevance_score_fn(20000.0))
    print(default_relevance_score_fn(200000.0))
    print(default_relevance_score_fn(2000000.0))
