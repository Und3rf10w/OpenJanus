from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.agents.conversational_chat.base import ConversationalChatAgent

from openjanus.toolkits.toolkit import get_openjanus_tools


if __name__ == "__main__":
    chat_llm = ChatOpenAI(
        model="gpt-3.5-turbo-1106",
        temperature=0.5
    )
    memory = ConversationSummaryBufferMemory(return_messages=True)
    chat_agent = ConversationalChatAgent.from_llm_and_tools(
        llm=chat_llm,
        tools=get_openjanus_tools(llm=chat_llm),
        # system_message=  # TODO:
        memory=memory,
    )
    agent_chain = AgentExecutor.from_agent_and_tools(
        tools=get_openjanus_tools(llm=chat_llm),
        llm=chat_llm,
        agent=chat_agent,
        memory=memory,
    )
    # TODO: Run app here