from datetime import datetime
from typing import Dict, Any

from langchain import PromptTemplate, LLMChain
from langchain.experimental.generative_agents.generative_agent import \
    GenerativeAgent


class LlamaGenerativeAgent(GenerativeAgent):

    system_prompt: str = (
        "A chat between a curious user and an artificial intelligence assistant. The assistant "
        "gives helpful, detailed, and polite answers to the user's questions.\n"
        "###USER: %s\n"
        "###ASSISTANT: ")

    def chain(self, prompt: PromptTemplate) -> LLMChain:
        return LLMChain(
            llm=self.llm, prompt=prompt, verbose=self.verbose, memory=self.memory
        )

    def _get_entity_from_observation(self, observation: str) -> str:
        # TODO: better prompt for conversations.
        instruction = (
            f"Extract the entity from the following observation without explanation.\n"
            f"Observation: {observation}\n"
        )
        prompt = PromptTemplate.from_template(
            self.system_prompt % instruction
        )
        return self.chain(prompt).run(observation=observation).strip()

    def _get_entity_action(self, observation: str, entity_name: str) -> str:
        instruction = (
            f"What is the {entity_name} doing in the following observation?\n"
            f"Observation: {observation}\n"
        )
        prompt = PromptTemplate.from_template(
            self.system_prompt % instruction
        )
        return (
            self.chain(prompt).run(entity=entity_name, observation=observation).strip()
        )

    def summarize_related_memories(self, observation: str) -> str:
        """Summarize memories that are most relevant to an observation."""
        prompt = PromptTemplate.from_template(
            "{q1}?\n"
            "Context from memory:\n"
            "{relevant_memories}\n"
            "Relevant context:"
        )
        entity_name = self._get_entity_from_observation(observation)
        entity_action = self._get_entity_action(observation, entity_name)
        q1 = f"What is the relationship between {self.name} and {entity_name}"
        q2 = f"{entity_name} is {entity_action}"
        return self.chain(prompt=prompt).run(q1=q1, queries=[q1, q2]).strip()

    def summarize_speaker_memories(self, speaker: str, observation: str) -> str:
        instruction = (
            f"what is the most possible relationship between {self.name} and {speaker} in the"
            f" following observation? Do not embellish if you don't know. Do not return a list.\n"
            "Observation: {relevant_memories}\n"
        )
        prompt = PromptTemplate.from_template(
            self.system_prompt % instruction
        )
        return self.chain(prompt=prompt).run(me=self.name, speaker=speaker, queries=[f"{speaker}"]).strip()

    def _compute_agent_summary(self) -> str:
        instruction = (
            f"Summarize {self.name}'s core characteristics given the following input. Do not "
            f"embellish if you don't know. Do not return a list.\n"
            "Input: {relevant_memories}\n"
        )
        prompt = PromptTemplate.from_template(
            self.system_prompt % instruction
        )
        # The agent seeks to think about their core characteristics.
        return (
            self.chain(prompt)
            .run(name=self.name, queries=[f"{self.name}'s core characteristics"])
            .strip()
        )

    def _generate_dialogue_reaction(self, speaker: str, observation: str, suffix: str) -> str:
        """React to a given observation or dialogue act."""
        prompt = PromptTemplate.from_template(
            "{agent_summary_description}"
            + "\nIt is {current_time}."
            + "\n{agent_name}'s status: {agent_status}"
            + "\nSummary of relevant context from {agent_name}'s memory:"
            + "\n{relevant_memories}"
            + "\nMost recent observations: {most_recent_memories}"
            + "\nObservation: {observation}"
            + "\n\n"
            + suffix
        )
        agent_summary_description = self.get_summary()
        relevant_memories_str = self.summarize_speaker_memories(speaker, observation)
        current_time_str = datetime.now().strftime("%B %d, %Y, %I:%M %p")
        kwargs: Dict[str, Any] = dict(
            agent_summary_description=agent_summary_description,
            current_time=current_time_str,
            relevant_memories=relevant_memories_str,
            agent_name=self.name,
            observation= speaker + " says " + observation,
            agent_status=self.status,
        )
        consumed_tokens = self.llm.get_num_tokens(
            prompt.format(most_recent_memories="", **kwargs)
        )
        kwargs[self.memory.most_recent_memories_token_key] = consumed_tokens
        return self.chain(prompt=prompt).run(**kwargs).strip()

    def generate_dialogue(self, speaker: str, observation: str):
        """React to a given observation."""
        call_to_action_template = (
            "What would {agent_name} say? To end the conversation, write:"
            ' GOODBYE: "what to say". Otherwise to continue the conversation,'
            ' write: SAY: "what to say next"\n\n'
        )
        full_result = self._generate_dialogue_reaction(
            speaker,
            observation,
            call_to_action_template
        )
        result = full_result.strip().split("\n")[0]
        if "GOODBYE:" in result:
            farewell = self._clean_response(result.split("GOODBYE:")[-1])
            self.memory.save_context(
                {},
                {
                    self.memory.add_memory_key: f"{self.name} observed "
                                                f"{observation} and said {farewell}"
                },
            )
            return False, f"{self.name} said {farewell}"
        if "SAY:" in result:
            response_text = self._clean_response(result.split("SAY:")[-1])
            self.memory.save_context(
                {},
                {
                    self.memory.add_memory_key: f"{self.name} observed "
                                                f"{observation} and said {response_text}"
                },
            )
            return True, f"{self.name} said {response_text}"
        else:
            return False, result
