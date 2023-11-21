import asyncio
import logging

from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.agents.conversational_chat.base import ConversationalChatAgent

from openjanus.toolkits.toolkit import get_openjanus_tools
from openjanus.chains.prompt import BASE_AGENT_SYSTEM_PROMPT_PREFIX


logging.basicConfig(level=logging.DEBUG)


async def test_invoke(agent_chain):
    await agent_chain.ainvoke({"input":"Hello ATC, this is John Smith, on approach to Seraphim Station. Requesting permission to land, over.", "chat_history": []})


if __name__ == "__main__":
    chat_llm = ChatOpenAI(
        model="gpt-3.5-turbo-1106",
        temperature=0.5,
        streaming=True,
        verbose=True
    )
    memory = ConversationSummaryBufferMemory(llm=chat_llm, return_messages=True)
    chat_agent = ConversationalChatAgent.from_llm_and_tools(
        llm=chat_llm,
        tools=get_openjanus_tools(llm=chat_llm),
        system_message=BASE_AGENT_SYSTEM_PROMPT_PREFIX,
        verbose=True
    )
    agent_chain = AgentExecutor.from_agent_and_tools(
        tools=get_openjanus_tools(llm=chat_llm),
        llm=chat_llm,
        agent=chat_agent,
        verbose=True
    )
    # TODO: Run app here
    # Async invoking the chain:
    asyncio.run(test_invoke(agent_chain))
    # Invoking the chain:
    # agent_chain.invoke({"input":"Hello ATC, this is John Smith, on approach to Seraphim Station. Requesting permission to land, over.", "chat_history": []})
    print("done")