from typing import Annotated, List, TypedDict
from typing import List
import requests
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.runnables import Runnable, RunnableConfig
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph, START
import uuid
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
import json

from langgraph.prebuilt import ToolNode
from atlassian import Confluence

token = "xyz..."

@tool
def read_confluence_by_space_key(space_key: str, query_str:str) -> str:
    """Read the confluence page by space key.
    
    Sample input:
        space_key = "PD"
        query_str = "Bende"
    """
    confluence = Confluence(
    url="http://localhost:8090",
    token=token,
    )

    # Construct the CQL query
    cql = f"space={space_key} and text~'{query_str}'"

    # Make the GET request
    response = confluence.cql(cql, limit=100)

    # Check if the request was successful
    if response:
        results = response.get("results", [])
        content = "\n".join([json.dumps(result["excerpt"]) for result in results])  
        cleaned_content = content.replace("@@@hl@@@", "").replace("@@@endhl@@@", "").encode().decode('unicode_escape')
        print(cleaned_content)  
    else:
        print( "Failed to retrieve content")

    return cleaned_content

@tool
def get_all_spaces() -> str:
    """Get all the spaces in the confluence.
    """
    confluence = Confluence(
    url="http://localhost:8090",
    token=token,
    )

    return json.dumps(confluence.get_all_spaces(start=0, limit=500, expand=None))

def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
                if(msg_repr in "The prefix "):
                    msg_repr = msg_repr[:200] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)




class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            configuration = config.get("configurable", {})
            passenger_id = configuration.get("passenger_id", None)
            state = {**state, "user_info": passenger_id}
            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}


llm = ChatOpenAI(model="gpt-4o-mini")

primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful customer support assistant for Confluence management. "
            " Use the provided tools to search for tables and columns to answer the user's questions. "
            " If a search comes up empty, expand your search before giving up."
        ),
        ("placeholder", "{messages}"),
    ]
)


confluence_tools = [
    read_confluence_by_space_key,
    get_all_spaces
]
datamodel_assistant_runnable = primary_assistant_prompt | llm.bind_tools(confluence_tools)


builder = StateGraph(State)


builder.add_node("assistant", Assistant(datamodel_assistant_runnable))
builder.add_node("tools", create_tool_node_with_fallback(confluence_tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "assistant")

memory = MemorySaver()
part_1_graph = builder.compile(checkpointer=memory)

tutorial_questions = [
    "What is the Bende project about?"
]

thread_id = str(uuid.uuid4())

config = {
    "configurable": {
        "thread_id": thread_id,
    }
}

_printed = set()
for question in tutorial_questions:
    events = part_1_graph.stream(
        {"messages": ("user", question)}, config, stream_mode="values"
    )
    for event in events:
        _print_event(event, _printed)