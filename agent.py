import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt.chat_agent_executor import create_function_calling_executor
from tools import TOOL_KIT

load_dotenv()


class Agent:
    def __init__(self, instructions:str, model:str="gpt-4o-mini"):

        # Initialize the LLM
        llm = ChatOpenAI(
            model=model,
            temperature=0.0,
            base_url="https://openai.vocareum.com/v1",
            api_key=os.getenv("VOCAREUM_API_KEY")
        )

        self.system_message = SystemMessage(content=instructions)
        # Build a graph that uses OpenAI function calling to route to tools
        self.graph = create_function_calling_executor(model=llm, tools=TOOL_KIT)

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
        response = self.graph.invoke(
            input= {
                "messages": messages
            }
        )
        
        return response

    def get_agent_tools(self):
        """Get list of available tools for the Energy Advisor"""
        return [t.name for t in TOOL_KIT]
