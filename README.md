![LangChain Academy](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba1020525eea7873f96_LCA-big-green%20(2).svg)

# LangGraph Course


Here, I list all the important points that I've learned from the course.

The initial setup instructions can be found in [SETUP.md](SETUP.md).

# Basics (Modules 1 and 2)

## Tavily

A nice specialized for LLMs tool for a websearch.
* Tavily Search API is a search engine optimized for LLMs and RAG, aimed at efficient, 
quick, and persistent search results. 
* You can sign up for an API key [here](https://tavily.com/). 
It's easy to sign up and offers a very generous free tier. Some lessons (in Module 4) will use Tavily. 

* Set `TAVILY_API_KEY` in your environment.

## Set up LangGraph Studio

* LangGraph Studio is a custom IDE for viewing and testing agents.
* Studio can be run locally and opened in your browser on Mac, Windows, and Linux.
* See documentation [here](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/#local-development-server) on the local Studio development server and [here](https://langchain-ai.github.io/langgraph/how-tos/local-studio/#run-the-development-server). 
* Graphs for LangGraph Studio are in the `module-x/studio/` folders.
* To start the local development server, run the following command in your terminal in the `/studio` directory each module:

```
langgraph dev
```

You should see the following output:
```
- 🚀 API: http://127.0.0.1:2024
- 🎨 Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- 📚 API Docs: http://127.0.0.1:2024/docs
```

Open your browser and navigate to the Studio UI: `https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`.

* To use Studio, you will need to create a .env file with the relevant API keys
* Run this from the command line to create these files for module 1 to 6, as an example:
```
for i in {1..6}; do
  cp module-$i/studio/.env.example module-$i/studio/.env
  echo "OPENAI_API_KEY=\"$OPENAI_API_KEY\"" > module-$i/studio/.env
done
echo "TAVILY_API_KEY=\"$TAVILY_API_KEY\"" >> module-4/studio/.env
```


## Agent with Memory

Use `InMemorySaver`: 

```python
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver() 


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[get_weather],
    checkpointer=checkpointer 
)

# Run the agent
config = {
    "configurable": {
        "thread_id": "1"  
    }
}

sf_response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
    config
)

# Continue the conversation using the same thread_id
ny_response = agent.invoke(
    {"messages": [{"role": "user", "content": "what about new york?"}]},
    config 
)
```


## Intro to Production

- `LangGraph` - a library
- `LangGraph API` - bundles the graph code - enables to access your code within a client
- `LangGraph Cloud` - hosts the API
- `LangGraph Studio` - play with the graph
- `LangGraph SDK` - programmatically interact with your API

## State Schema

There are several options to define the state.

The simplest one is with the `TypedDict`.

```python
from typing import Literal

class TypedDictState(TypedDict):
    name: str
    mood: Literal["happy","sad"]
```

Next, with `dataclasses` we can use the `state.name` notation:

```python
from dataclasses import dataclass

@dataclass
class DataclassState:
    name: str
    mood: Literal["happy","sad"]
```

With `pydantic`, we can actually raise errors when the type is unmatched:

```python
from pydantic import BaseModel

class PydanticState(BaseModel):
    name: str
    mood: Literal['happy', 'sad'] # "happy" or "sad"
```


## State Reducers

Reducers specify how state updates are performed on specific keys / channels in the state schema.

- The default is to just overwrite the state with the new values.
- If there is a branching, and we try to update the state at the same time, the `InvalidUpdateError` will occur.
- We can use the `Annotated` type to specify a reducer function.
```python
from operator import add
from typing import Annotated

class State(TypedDict):
    foo: Annotated[list[int], add]
```
- We can also define custom reducers:
```python
def reduce_list(left: list | None, right: list | None) -> list:
    """Safely combine two lists, handling cases where either or both inputs might be None.

    Args:
        left (list | None): The first list to combine, or None.
        right (list | None): The second list to combine, or None.

    Returns:
        list: A new list containing all elements from both input lists.
               If an input is None, it's treated as an empty list.
    """
    if not left:
        left = []
    if not right:
        right = []
    return left + right

class DefaultState(TypedDict):
    foo: Annotated[list[int], add]

class CustomReducerState(TypedDict):
    foo: Annotated[list[int], reduce_list]
```
- You can use our `MessageState`:
```python
from typing import Annotated
from langgraph.graph import MessagesState
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

# Define a custom TypedDict that includes a list of messages with add_messages reducer
class CustomMessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    added_key_1: str
    added_key_2: str
    # etc

# Use MessagesState, which includes the messages key with add_messages reducer
class ExtendedMessagesState(MessagesState):
    # Add any keys needed beyond messages, which is pre-built 
    added_key_1: str
    added_key_2: str
    # etc
```

### `add_messages`:

To add:
```python
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage

# Initial state
initial_messages = [AIMessage(content="Hello! How can I assist you?", name="Model"),
                    HumanMessage(content="I'm looking for information on marine biology.", name="Lance")
                   ]

# New message to add
new_message = AIMessage(content="Sure, I can help with that. What specifically are you interested in?", name="Model")

# Test
add_messages(initial_messages , new_message)
```

To rewrite:
```python
# Initial state
initial_messages = [AIMessage(content="Hello! How can I assist you?", name="Model", id="1"),
                    HumanMessage(content="I'm looking for information on marine biology.", name="Lance", id="2")
                   ]

# New message to add
new_message = HumanMessage(content="I'm looking for information on whales, specifically", name="Lance", id="2")

# Test
add_messages(initial_messages , new_message)
```

To remove:
```python
from langchain_core.messages import RemoveMessage

# Message list
messages = [AIMessage("Hi.", name="Bot", id="1")]
messages.append(HumanMessage("Hi.", name="Lance", id="2"))
messages.append(AIMessage("So you said you were researching ocean mammals?", name="Bot", id="3"))
messages.append(HumanMessage("Yes, I know about whales. But what others should I learn about?", name="Lance", id="4"))

# Isolate messages to delete
delete_messages = [RemoveMessage(id=m.id) for m in messages[:-2]]
print(delete_messages)
#%%
add_messages(messages , delete_messages)
```

## Multiple Schemas

- you can use multiple schemas between nodes
  * Internal nodes may pass information that is *not required* in the graph's input / output.

  * We may also want to use different input / output schemas for the graph. The output might, for example, only contain a single relevant output key.


```python
from typing_extensions import TypedDict
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END

class OverallState(TypedDict):
    foo: int

class PrivateState(TypedDict):
    baz: int

def node_1(state: OverallState) -> PrivateState:
    print("---Node 1---")
    return {"baz": state['foo'] + 1}

def node_2(state: PrivateState) -> OverallState:
    print("---Node 2---")
    return {"foo": state['baz'] + 1}

# Build graph
builder = StateGraph(OverallState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", END)
graph = builder.compile()

graph.invoke({"foo" : 1})
```


## Filtering and Trimming Messages

- we want to avoid long context because of latency and costs

We can use the `RemoveMEssage` class:
```python
from langchain_core.messages import RemoveMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState

# Nodes
def filter_messages(state: MessagesState):
    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"messages": delete_messages}

def chat_model_node(state: MessagesState):    
    return {"messages": [llm.invoke(state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("filter", filter_messages)
builder.add_node("chat_model", chat_model_node)
builder.add_edge(START, "filter")
builder.add_edge("filter", "chat_model")
builder.add_edge("chat_model", END)
graph = builder.compile()

# View
display(Image(graph.get_graph().draw_mermaid_png()))
```

You can just pass the last message (to filter out other messages):
```python
def chat_model_node(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"][-1:])]}
```

You can trim the messages using `trim_messages()` function:
```python
trim_messages(
        messages,
        max_tokens=100,
        strategy="last",
        token_counter=ChatOpenAI(model="gpt-4o"),
        allow_partial=False
    )
```

## Chatbot with message summarization

```python
import os
from typing import *

from dotenv import load_dotenv
load_dotenv()
from IPython.display import Image, display


from langchain_ollama import ChatOllama
from langchain_together import ChatTogether
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage, RemoveMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver
#%%
# ------------------------------------------------------ #
# GLOBALS
# ------------------------------------------------------ #
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
#%%
# ------------------------------------------------------ #
# MODELS & TOOLS
# ------------------------------------------------------ #
# chat_llm = ChatTogether(model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", api_key=TOGETHER_API_KEY)
chat_llm = ChatOllama(model='llama3.2:latest')
#%%
chat_llm.invoke('hi').content
#%%
# ------------------------------------------------------ #
# STATE
# ------------------------------------------------------ #
class State(MessagesState):
    summary: str
#%%
# ------------------------------------------------------ #
# NODES
# ------------------------------------------------------ #
def call_model(state: State):
    # summary = state['summary']
    summary = state.get('summary', '')
    if summary:
        # the summary exists
        system_message = f'Summary of conversation earlier: {summary}'
        messages = [SystemMessage(content=system_message)] + state['messages']
    else:
        # no summary
        messages =  state['messages']
    response = chat_llm.invoke(messages)
    return {'messages': response}


def summarize_conversation(state: State):
    # summary = state['summary']
    summary = state.get('summary', '')
    if summary:
        # a summary already exists
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            f"Extend the summary by taking into the new messages above:"
        )
    else:
        # no summary
        summary_message = "Create a summary of the conversation above:"
    messages = state['messages'] + [HumanMessage(content=summary_message)]
    response = chat_llm.invoke(messages)

    delete_messages = [RemoveMessage(id=m.id) for m in state['messages'][:-2]]
    return {'summary': response.content, 'messages': delete_messages}
#%%
# ------------------------------------------------------ #
# CONDITIONAL EDGES
# ------------------------------------------------------ #
def should_continue(state: State) -> Literal["summarize_conversation", "__end__"]:
    messages = state['messages']
    if len(messages) > 6:
        return 'summarize_conversation'
    return END
#%%
# ------------------------------------------------------ #
# MEMORY
# ------------------------------------------------------ #
memory = MemorySaver()
#%%
# ------------------------------------------------------ #
# WORKFLOWS
# ------------------------------------------------------ #
def workflow_default():
    # define workflow
    i_workflow = StateGraph(State)
    # nodes
    i_workflow.add_node('conversation', call_model)
    i_workflow.add_node('summarize_conversation', summarize_conversation)
    # edges
    i_workflow.add_edge(START, 'conversation')
    # i_workflow.add_edge('conversation', 'summarize_conversation')
    i_workflow.add_conditional_edges('conversation', should_continue)
    i_workflow.add_edge('summarize_conversation', END)
    return i_workflow
#%%
# ------------------------------------------------------ #
# COMPILE GRAPH
# ------------------------------------------------------ #
workflow = workflow_default()
graph = workflow.compile(checkpointer=memory)
# print(graph.get_graph().draw_ascii())
display(Image(graph.get_graph().draw_mermaid_png()))
#%%
# ------------------------------------------------------ #
# THREAD
# ------------------------------------------------------ #
config = {"configurable": {"thread_id": "1"}}
#%%
input_message = HumanMessage(content='hi! I am Lance')
output = graph.invoke({'messages': [input_message]}, config=config)
for m in output['messages']:
    m.pretty_print()
#%%
input_message = HumanMessage(content="what's my name?")
output = graph.invoke({"messages": [input_message]}, config=config)
for m in output['messages']:
    m.pretty_print()
#%%
input_message = HumanMessage(content="i like the 49ers!")
output = graph.invoke({"messages": [input_message]}, config=config)
for m in output['messages']:
    m.pretty_print()
#%%
state = graph.get_state(config)
state.values.get("summary","")
#%%
input_message = HumanMessage(content="i like Nick Bosa, isn't he the highest paid defensive player?")
output = graph.invoke({"messages": [input_message]}, config=config)
for m in output['messages']:
    m.pretty_print()
#%%
print(graph.get_state(config).values.get("summary",""))
```

## Chatbot Summarization with External DB Memory

Like the previous section, but just change the `MEMORY` part to this:

```python
# ------------------------------------------------------ #
# MEMORY
# ------------------------------------------------------ #
# memory = MemorySaver()

import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
# In memory
# conn = sqlite3.connect(":memory:", check_same_thread = False)
# outer source
db_path = "chat.db"
conn = sqlite3.connect(db_path, check_same_thread=False)
memory = SqliteSaver(conn)
```


# Human in the Loop (Module 3)

Let's put some humans in the loop.


## Streaming

LangGraph supports a few [different streaming modes](https://langchain-ai.github.io/langgraph/how-tos/stream-values/) for [graph state](https://langchain-ai.github.io/langgraph/how-tos/stream-values/):
 
* `values`: This streams the full state of the graph after each node is called.
* `updates`: This streams updates to the state of the graph after each node is called.

![values_vs_updates.png](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dbaf892d24625a201744e5_streaming1.png)

Stream in the `updates` mode (the output of the current node):

```python
config = {"configurable": {"thread_id": "1"}}
# Start conversation
chunks = []
for chunk in graph.stream({"messages": [HumanMessage(content="hi! I'm Lance")]}, config, stream_mode="updates"):
    chunks.append(chunk)
    # print(chunk)
    chunk['conversation']["messages"].pretty_print()
chunks
```

Stream in the `values` mode (the output of the current node + all the previous nodes):

```python
config = {"configurable": {"thread_id": "2"}}
# Start conversation
input_message = HumanMessage(content="hi! I'm Lance")
events = []
for event in graph.stream({"messages": [input_message]}, config, stream_mode="values"):
    events.append(event)
    # print(chunk)
    print('vvv' * 25)
    for m in event['messages']:
        m.pretty_print()
    print('---' * 25)
events
```

The streaming of the tokens within the current node with the `astream_events` method:

```python
config = {'configurable': {'thread_id': '3'}}
input_message = HumanMessage(content='Tell me about the current conflicts that Israel is involved in.')
async for event in graph.astream_events({'messages': [input_message]}, config, version='v2'):
    # print('vvv')
    # print(f"Node: {event['metadata'].get('langgraph_node', '')}\n"
    #       f"Type: {event['event']}\n"
    #       f"Name: {event['name']}")
    # print('^^^')
    if event['event'] == 'on_chat_model_stream':
        data = event['data']
        print(data['chunk'].content, end='')
```


## Breakpoints


`human-in-the-loop' approach brings several advantages:
- _approval_ of the inputs
- _debugging_ 
- _editing_ the current state of the graph

To start the strimming do:
```python
for event in graph.stream(initial_input, config, stream_mode="values"):
    event['messages'][-1].pretty_print()
```

User approval:
```python
user_approval = input('Do you want to call the tool? yes or no')
```

The continuation:
```python
if user_approval.lower() == 'yes':
    for event in graph.stream(None, config, stream_mode='values'):
        event['messages'][-1].pretty_print()
else:
    print('operation cancelled by user')
```

When we invoke the graph with `None`, it will just continue from the last state checkpoint!

![breakpoints.jpg](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dbae7985b747dfed67775d_breakpoints1.png)

For clarity, LangGraph will re-emit the current state, which contains the `AIMessage` with tool call.

And then it will proceed to execute the following steps in the graph, which start with the tool node.

We see that the tool node is run with this tool call, and it's passed back to the chat model for our final answer.


## Editing Graph State

The trick is to create a dummy node:
```python
def human_feedback(state: MessagesState):
    pass
```

To update a workflow:
```python
def workflow_hil():
    # define workflow
    i_workflow = StateGraph(State)
    # nodes
    i_workflow.add_node('assistant', assistant)
    i_workflow.add_node('tools', ToolNode(tools))
    i_workflow.add_node('human_feedback', human_feedback)
    # edges
    i_workflow.add_edge(START, 'human_feedback')
    i_workflow.add_edge('human_feedback', 'assistant')
    i_workflow.add_conditional_edges('assistant', tools_condition)
    i_workflow.add_edge('tools', 'human_feedback')
    return i_workflow

workflow = workflow_hil()
graph = workflow.compile(interrupt_before=['human_feedback'], checkpointer=memory)
# graph = workflow.compile()
# print(graph.get_graph().draw_ascii())
display(Image(graph.get_graph().draw_mermaid_png()))
```

![mod_3_output.png](pics/mod_3_output.png)

To update state with the `as_node` parameter:
```python
thread = {"configurable": {"thread_id": "5"}}
initial_input = {"messages": "Multiply 2 and 3"}

# Run the graph until the first interruption
for event in graph.stream(initial_input, thread, stream_mode="values"):
    event["messages"][-1].pretty_print()

# Get user input
user_input = input("Tell me how you want to update the state: ")

# We now update the state as if we are the human_feedback node
graph.update_state(thread, {"messages": user_input}, as_node="human_feedback")

# Continue the graph execution
for event in graph.stream(None, thread, stream_mode="values"):
    event["messages"][-1].pretty_print()

# to continue after tools
for event in graph.stream(None, thread, stream_mode="values"):
    event["messages"][-1].pretty_print()
```

## Dynamic Breakpoints

Sometimes it is helpful to allow the graph dynamically interrupt itself.

This is an internal breakpoint, and can be achieved using `NodeInterrupt`. This has a few specific benefits: 

(1) you can do it conditionally (from inside a node based on developer-defined logic).

(2) you can communicate to the user why its interrupted (by passing whatever you want to the `NodeInterrupt`).


```python
from langgraph.errors import NodeInterrupt

def human_feedback(state: State):
    if len(state['messages'][-1].content) == 0:
        raise NodeInterrupt('Please, provide the question')
    return state
```

```python
state = graph.get_state(thread)
print(state.next)
print(state.interrupts[0].value)
```


## Time Travel

Let's see how LangGraph supports _debugging_ by using **time travel**.

We can use `get_state_history` to look at the whole history of states that were in the execution.

```python
all_states = [s for s in graph.get_state_history(thread)]
```

We can re-run our agent from any of the prior steps.

![fig2.jpg](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dbb038a0bd34b541c78fb8_time-travel2.png)

It does not _re-executes_ the graph (just replays) because it knows it already executed the state previously.

What if we want to run from that same step, but with a different input.

This is **forking**.

![fig3.jpg](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dbb038f89f2d847ee5c336_time-travel3.png)

```python
to_fork = all_states[-2]
to_fork.values["messages"]
to_fork.config
```

Let's modify the state at this checkpoint.

We can just run `update_state` with the `checkpoint_id` supplied. 

Remember how our reducer on `messages` works: 

* It will append, unless we supply a message ID.
* We supply the message ID to overwrite the message, rather than appending to state!

So, to overwrite the the message, we just supply the message ID, which we have `to_fork.values["messages"].id`.

```python
fork_config = graph.update_state(
    to_fork.config,
    {"messages": [HumanMessage(content='Multiply 5 and 3', 
                               id=to_fork.values["messages"][0].id)]},
)
```

This creates a new, forked checkpoint.
 
But, the metadata - e.g., where to go next - is perserved! 

We can see the current state of our agent has been updated with our fork.

Now, when we stream, the graph knows this checkpoint has never been executed.

So, the graph runs, rather than simply re-playing.

```python
for event in graph.stream(None, fork_config, stream_mode="values"):
    event['messages'][-1].pretty_print()
```

Now, we can see the current state is the end of our agent run.

```python
graph.get_state({'configurable': {'thread_id': '1'}})
```

# Building Your Assistant (Module 4)

Here, we will bring together all the concepts we've learned previously into a useful project where we will build an assistant that we can customize as we wish for whatever use-cases. 

But first, let's introduce several important concepts.

## Parallel Node Execution








