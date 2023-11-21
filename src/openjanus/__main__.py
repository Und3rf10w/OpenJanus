import asyncio
import logging
import threading
from pynput import keyboard

from langchain.agents import AgentExecutor
from langchain.agents.conversational_chat.base import ConversationalChatAgent
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.document_loaders.parsers import OpenAIWhisperParser


from openjanus.toolkits.toolkit import get_openjanus_tools
from openjanus.chains.prompt import BASE_AGENT_SYSTEM_PROMPT_PREFIX
from openjanus.stt.whisper.recorder import Recorder


logging.basicConfig(level=logging.DEBUG)


# async def test_invoke(agent_chain: AgentExecutor):
#     await agent_chain.ainvoke({"input":"Hello ATC, this is John Smith, on approach to Seraphim Station. Requesting permission to land, over.", "chat_history": []})


async def main():
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
    recorder = Recorder()
    # # TODO: Make the selected key selectable from the config file
    # with keyboard.Listener(
    #     on_press=lambda event: recorder.start_recording() if isinstance(event, keyboard.KeyCode) and event.char == 'r' and not recorder.is_recording else None,
    #     on_release=lambda event: threading.Thread(target=recorder.stop_recording).start() if isinstance(event, keyboard.KeyCode) and event.char == 'r' else None
    # ) as listener:
    #     try:
    #         while True:
    #             await asyncio.sleep(0.1)  # Allows asyncio to yield control to other tasks
    #     except KeyboardInterrupt:
    #         logging.WARN("User requested exit")
    #     listener.join()
    def on_release(event):
        if isinstance(event, keyboard.KeyCode) and event.char == 'r':
            threading.Thread(target=recorder.stop_recording).start()
            # After stopping, transcribe and invoke using a separate thread
            threading.Thread(target=recorder.transcribe_and_invoke, args=(agent_chain,)).start()     

    with keyboard.Listener(
        on_press=lambda event: recorder.start_recording() if isinstance(event, keyboard.KeyCode) and event.char == 'r' and not recorder.is_recording else None,
        on_release=on_release
    ) as listener:
        try:
            while True:
                await asyncio.sleep(0.1)  # Allows asyncio to yield control to other tasks
        except KeyboardInterrupt:
            logging.WARN("User requested exit")
        listener.join()


if __name__ == "__main__":
    # chat_llm = ChatOpenAI(
    #     model="gpt-3.5-turbo-1106",
    #     temperature=0.5,
    #     streaming=True,
    #     verbose=True
    # )
    # memory = ConversationSummaryBufferMemory(llm=chat_llm, return_messages=True)
    # chat_agent = ConversationalChatAgent.from_llm_and_tools(
    #     llm=chat_llm,
    #     tools=get_openjanus_tools(llm=chat_llm),
    #     system_message=BASE_AGENT_SYSTEM_PROMPT_PREFIX,
    #     verbose=True
    # )
    # agent_chain = AgentExecutor.from_agent_and_tools(
    #     tools=get_openjanus_tools(llm=chat_llm),
    #     llm=chat_llm,
    #     agent=chat_agent,
    #     verbose=True
    # )
    # TODO: Run app here
    # Async invoking the chain:
    # asyncio.run(test_invoke(agent_chain))
    # Invoking the chain:
    # agent_chain.invoke({"input":"Hello ATC, this is John Smith, on approach to Seraphim Station. Requesting permission to land, over.", "chat_history": []})
    asyncio.run(main())
    print("done")