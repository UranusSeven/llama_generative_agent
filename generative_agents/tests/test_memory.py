import pytest
from chroma import EnhancedChroma
from langchain import LlamaCpp
from langchain.embeddings import LlamaCppEmbeddings
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from llama_memory import LlamaGenerativeAgentMemory

local_path = "/Users/jon/Downloads/ggml-vicuna-13b-1.1-q4_2.bin"


def new_memory_retriever():
    embeddings_model = LlamaCppEmbeddings(model_path=local_path)
    vs = EnhancedChroma(embedding_function=embeddings_model)
    return TimeWeightedVectorStoreRetriever(
        vectorstore=vs, other_score_keys=["importance"], k=15
    )


def test():
    llm = LlamaCpp(model_path=local_path, verbose=True)
    memory = LlamaGenerativeAgentMemory(
        llm=llm,
        memory_retriever=new_memory_retriever(),
        verbose=True,
        reflection_threshold=8
        # we will give this a relatively low number to show how reflection works
    )
    print(
        memory._score_memory_importance(
            "Tommie remembers his dog, Bruno, from when he was a kid"
        )
    )
