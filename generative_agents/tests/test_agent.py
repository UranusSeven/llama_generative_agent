import pytest
from langchain import LlamaCpp
from langchain.embeddings import LlamaCppEmbeddings

from retrivers.llama_time_weighted_retriever import LlamaTimeWeightedVectorStoreRetriever
from vectorestores.chroma import EnhancedChroma
from ..llama_generative_agent import LlamaGenerativeAgent
from ..llama_memory import LlamaGenerativeAgentMemory


@pytest.fixture
def model_path():
    return "/Users/jon/Downloads/ggml-vicuna-13b-1.1-q4_2.bin"
    # return "/Users/jon/Documents/models/stable-vicuna-13B.ggml.q5_1.bin"


@pytest.fixture
def llm(model_path):
    return LlamaCpp(model_path=model_path, verbose=True, n_batch=256, temperature=0.3, n_ctx=2048,
                    use_mmap=False, stop=["###"])


@pytest.fixture
def retriever(model_path, llm):
    embeddings_model = LlamaCppEmbeddings(model_path=model_path)
    vs = EnhancedChroma(embedding_function=embeddings_model)
    return LlamaTimeWeightedVectorStoreRetriever(
        vectorstore=vs,
        other_score_keys=["importance"], k=15
    )


@pytest.fixture
def memory(llm, retriever):
    return LlamaGenerativeAgentMemory(
        llm=llm,
        memory_retriever=retriever,
        reflection_threshold=8,
        # we will give this a relatively low number to show how reflection works
        verbose=True,
    )


@pytest.fixture
def agent(retriever, memory, llm) -> LlamaGenerativeAgent:
    return LlamaGenerativeAgent(
        name="Tommie",
        age=25,
        traits="anxious, likes design, talkative",  # You can add more persistent traits here
        status="looking for a job",
        # When connected to a virtual world, we can have the characters update their status
        memory_retriever=retriever,
        llm=llm,
        memory=memory,
        verbose=True,
    )


def test_compute_agent_summary(agent):
    # expected: No statements were provided about Tommie's core characteristics.
    for _ in range(5):
        print(agent.get_summary(force_refresh=True))


def test_get_entity_from_observation(agent):
    # expected: Jon
    print(
        agent._get_entity_from_observation("Jon says What are you most worried about today?")
    )


def test_get_entity_action(agent):
    # expected: Jon is asking what the person is most worried about on that day.
    print(
        agent._get_entity_action(
            "Jon says What are you most worried about today?", "Jon")
    )


def test_summarize_related_memories(agent):
    print(
        agent.summarize_related_memories(
            "Jon says What are you most worried about today?")
    )


def test_summarize_speaker_memories(agent):
    """
    What is the relationship between Tommie and Person A?
    Context from memory:
    - May 04, 2023, 10:58 PM: Tommie sees the new home
    - May 04, 2023, 10:58 PM: Tommie remembers his dog, Bruno, from when he was a kid
    - May 04, 2023, 10:59 PM: Tommie is hungry
    - May 04, 2023, 11:00 PM: Tommie tries to get some rest.
    - May 04, 2023, 10:58 PM: Tommie feels tired from driving so far
    - May 04, 2023, 10:59 PM: The new neighbors have a cat
    - May 04, 2023, 10:59 PM: The road is noisy at night
    Relevant context:
    """
    print(
        agent.summarize_speaker_memories("Jon", "What are you most worried about today?")
    )


def test_generate_dialogue(agent):
    print(
        agent.generate_dialogue("Jon", "What are you most worried about today?")
    )
