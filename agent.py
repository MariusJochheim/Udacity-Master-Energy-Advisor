import os
from typing import List, TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
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

        # Build an explicit LangGraph with schema, nodes, and edges
        graph = StateGraph(self.AgentState)
        graph.add_node("chat", self._call_model)
        graph.add_node("tools", ToolNode(TOOL_KIT))
        graph.add_edge("tools", "chat")
        graph.add_conditional_edges(
            "chat",
            tools_condition,
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
