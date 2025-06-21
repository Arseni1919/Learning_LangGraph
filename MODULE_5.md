# Long-Turn Memory (Module 5)

## LangGraph Store

Here, we'll introduce the [LangGraph Memory Store](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore) as a way to save and retrieve long-term memories.

We'll build a chatbot that uses both `short-term (within-thread)` and `long-term (across-thread)` memory.
 
We'll focus on long-term [semantic memory](https://langchain-ai.github.io/langgraph/concepts/memory/#semantic-memory), which will be facts about the user. 

These long-term memories will be used to create a personalized chatbot that can remember facts about the user.

It will save memory ["in the hot path"](https://langchain-ai.github.io/langgraph/concepts/memory/#writing-memories), as the user is chatting with it.

```python
from langgraph.store.memory import InMemoryStore
in_memory_store = InMemoryStore()
```

When storing objects (e.g., memories) in the [Store](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore), we provide:

- The `namespace` for the object, a tuple (similar to directories)
- the object `key` (similar to filenames)
- the object `value` (similar to file contents)

We use the [put](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore.put) method to save an object to the store by `namespace` and `key`.

![langgraph_store.png](attachment:6281b4e3-4930-467e-83ce-ba1aa837ca16.png)

TODO: There were bugs, I need to reimplement it on demand.

## Chatbot with Profile Schema  


It is possible to get store the memory as `schema` with the `with_structured_output` method of a model.

If the schema is too complicated, the model struggles with the output. The `trustcall` library comes to help to avoid errors while creating the schemas.

## Chatbot with Collection Schema 

Same thing as in the previous section but with collections in memory and not a single datapoint.

























## Credits

- [LangChain | persistence](https://langchain-ai.github.io/langgraph/how-tos/persistence/)
- 