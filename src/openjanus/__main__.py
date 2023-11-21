import asyncio
from csv import excel
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


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


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
    # TODO: Make the selected key selectable from the config file
    class KeyListener:
        def __init__(self, recorder: Recorder, agent_chain: AgentExecutor):
            self.recorder = recorder
            self.agent_chain = agent_chain
            self.record_key_pressed = False

        def on_press(self, key):
            try:
                if key == keyboard.Key.f12 and not self.recorder.is_recording:
                    LOGGER.info("Record button pressed")
                    self.recorder.start_recording()
            except AttributeError:
                pass

        def on_release(self, key):
            try:
                if key == keyboard.Key.f12 and self.recorder.is_recording:
                    LOGGER.info("Recording button released")
                    recording_path = self.recorder.stop_recording()
                    threading.Thread(target=self.recorder.transcribe_and_invoke, args=(self.agent_chain,recording_path,)).start()
                    self.record_key_pressed = False
            except Exception as e:
                LOGGER.error("Raised an error when stopping recording", exc_info=e)



    key_listener = KeyListener(recorder, agent_chain)
    with keyboard.Listener(
        on_press=key_listener.on_press,
        on_release=key_listener.on_release,
        suppress=False
    ) as listener:
        listener.join()


if __name__ == "__main__":
   
    # TODO: Run app here
    # Async invoking the chain:
    # asyncio.run(test_invoke(agent_chain))
    # Invoking the chain:
    # agent_chain.invoke({"input":"Hello ATC, this is John Smith, on approach to Seraphim Station. Requesting permission to land, over.", "chat_history": []})
    # asyncio.run(main())
    # print("done")
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()