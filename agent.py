import os
import json
from typing import List, TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, ToolMessage
from langgraph.graph import StateGraph, END
from tools import TOOL_KIT

load_dotenv()


class Agent:
    def __init__(self, instructions:str, model:str="gpt-4o-mini"):
        class AgentState(TypedDict):
            messages: List[BaseMessage]

        self.AgentState = AgentState

        # Initialize the LLM
        llm = ChatOpenAI(
            model=model,
            temperature=0.0,
            base_url="https://openai.vocareum.com/v1",
            api_key=os.getenv("VOCAREUM_API_KEY")
        )

        self.system_message = SystemMessage(content=instructions)
        # Bind tools to the model for function calling
        self.model = llm.bind_tools(TOOL_KIT)
        self.tool_map = {t.name: t for t in TOOL_KIT}

        # Build an explicit LangGraph with schema, nodes, and edges
        graph = StateGraph(self.AgentState)
        graph.add_node("chat", self._call_model)
        graph.add_node("tools", self._call_tools)
        graph.add_edge("tools", "chat")
        graph.add_conditional_edges(
            "chat",
            self._route_from_chat,
            {"tools": "tools", "__end__": END},
        )
        graph.set_entry_point("chat")
        self.graph = graph.compile()

    def invoke(self, question: str, context:str=None) -> str:
        """
        Ask the Energy Advisor a question about energy optimization.
        
        Args:
            question (str): The user's question about energy optimization
            context (str): Optional additional context (e.g., location)
        
        Returns:
            str: The advisor's response with recommendations
        """
        
        messages = [self.system_message]
        if context:
            # Add some context to the question as a system message
            messages.append(SystemMessage(content=context))

        messages.append(HumanMessage(content=question))
        
        # Get response from the agent
        return self.graph.invoke({"messages": messages})

    def get_agent_tools(self):
        """Get list of available tools for the Energy Advisor"""
        return [t.name for t in TOOL_KIT]

    def _call_model(self, state: dict):
        """LLM node: take messages, call model, append the AI message."""
        ai_message = self.model.invoke(state["messages"])
        return {"messages": state["messages"] + [ai_message]}

    def _call_tools(self, state: dict):
        """Tool node: execute requested tools and append ToolMessages."""
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", []) or []
        tool_messages: List[ToolMessage] = []

        for call in tool_calls:
            name = call["function"]["name"]
            args_raw = call["function"].get("arguments") or "{}"
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except Exception:
                args = {}
            tool = self.tool_map.get(name)
            if not tool:
                result = {"error": f"Tool '{name}' not found"}
            else:
                try:
                    result = tool.invoke(args)
                except Exception as e:
                    result = {"error": str(e)}

            tool_messages.append(
                ToolMessage(
                    content=json.dumps(result),
                    name=name,
                    tool_call_id=call.get("id"),
                )
            )

        return {"messages": state["messages"] + tool_messages}

    def _route_from_chat(self, state: dict) -> str:
        """Decide whether to call tools or end based on model output."""
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return "__end__"
