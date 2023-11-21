import asyncio

from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.agents.conversational_chat.base import ConversationalChatAgent

from openjanus.toolkits.toolkit import get_openjanus_tools, get_elevenlabs_tts_tool


if __name__ == "__main__":
    chat_llm = ChatOpenAI(
        model="gpt-3.5-turbo-1106",
        temperature=0.5,
        verbose=True
    )
    memory = ConversationSummaryBufferMemory(llm=chat_llm, return_messages=True)
    chat_agent = ConversationalChatAgent.from_llm_and_tools(
        llm=chat_llm,
        tools=get_openjanus_tools(llm=chat_llm),
        # system_message=  # TODO:
        verbose=True
    )
    agent_chain = AgentExecutor.from_agent_and_tools(
        tools=get_openjanus_tools(llm=chat_llm),
        llm=chat_llm,
        agent=chat_agent,
        verbose=True
    )
    # TODO: Run app here
    stream = agent_chain.invoke({"input":"Hello ATC, this is John Smith, on approach to Seraphim Station. Requesting permission to land, over.", "chat_history": []})
    tts = get_elevenlabs_tts_tool()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(tts.arun({"stream": stream['output']}))
    print("done")