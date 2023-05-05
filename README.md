
![image](https://user-images.githubusercontent.com/109661872/236526585-acc7b215-8181-4385-aadf-e908b6279251.png)



# LLaMA Generative Agent
A generative agent implementation for LLaMA based models, derived from langchain's implementation.

**This project is still in its early stages. The generative agent's inference is currently quite slow and may not produce reasonable answers. Any suggestions or advice on improving its performance would be greatly appreciated!**



# Demo
To run the demo, you need to download a LLaMA based model from huggingface, for example:

https://huggingface.co/eachadea/ggml-vicuna-13b-1.1/tree/main

And replace the model path before runing the following notebook:

https://github.com/UranusSeven/llama_generative_agent/blob/main/llama_generative_agent.ipynb

## Summary of a agent's core characteristics
|   | OpenAI  | LLaMA  |
|---|---|---|
| Without observations  | No statements were provided about Tommie's core characteristics.  |  I'm sorry, I do not have enough information about "Tommie" to provide a summary of their core characteristics. Could you please provide more context or details about who or what Tommie refers to? |
| With observations  | Tommie is a tired, hungry person who is trying to get some rest after seeing a new home. He remembers his dog from when he was a kid and notices the new neighbors have a cat. The road noise at night may be bothering him.  | 1. Tommie remembers his dog Bruno from when he was a kid. 2. Tommie sees the new home. 3. The road is noisy at night. 4. The new neighbors have a cat. 5. Tommie tries to get some rest. 6. Tommie feels tired from driving so far. 7. Tommie is hungry.  |

## Memory’s importance score
| Observation  | importance score  |
|---|---|
| Tommie remembers his dog, Bruno, from when he was a kid.   | 8 |
| Tommie feels tired from driving so far.  | 1 |
| Tommie sees the new home.  | 8 |
| The new neighbors have a cat.  | 2 |
| The road is noisy at night.  | 2 |
| Tommie is hungry.  | 1 |
| Tommie tries to get some rest.  | 1 |

## Reaction
```
Observation: Tommie sees his neighbor's cat
Reaction: Tommie might be curious about the cat and ask where it came from, or he might simply acknowledge its presence without saying anything.
```

## Dialogue
```
Dad: Have you got a new job?
Tommie: No, I haven't found one yet.
```
