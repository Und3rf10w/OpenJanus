import asyncio

import concurrent.futures
import logging
import threading
from pynput import keyboard

from langchain.agents import AgentExecutor
from langchain.agents.conversational_chat.base import ConversationalChatAgent
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.document_loaders.parsers import OpenAIWhisperParser


import openjanus.app.config as openjanus_config
from openjanus.chains.prompt import BASE_AGENT_SYSTEM_PROMPT_PREFIX
from openjanus.toolkits.toolkit import get_openjanus_tools
from openjanus.stt.whisper.recorder import Recorder
from openjanus.utils.exceptions import ListenKeyNotSupportedException


logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)


async def main():
    config = openjanus_config.load_config()
    chat_llm = ChatOpenAI(
        model="gpt-3.5-turbo-1106",
        # model="gpt-4-1106-preview",
        temperature=0.5,
        streaming=True,
        verbose=True
    )
    # memory = ConversationSummaryBufferMemory(llm=chat_llm, return_messages=True, memory_key="chat_history")
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
    listen_key = config["openjanus"]["listen_key"]
    # TODO: Make the selected key selectable from the config file
    class KeyListener:
        def __init__(self, recorder: Recorder, agent_chain: AgentExecutor):
            self.recorder = recorder
            self.agent_chain = agent_chain
            self.record_key_pressed = False
            self.listen_key = self.get_key(listen_key)

        def get_key(self, key: str):
            try:
                # Try to get a key for special keys
                return keyboard.Key[key]
            except KeyError:
                try:
                    # If that fails, the key is not a special key, then it must be a character
                    return keyboard.KeyCode.from_char(key)
                except KeyError:
                    LOGGER.error(f"The key {key} is not a valid key that can be used")
                    raise ListenKeyNotSupportedException(key)

        def on_press(self, key):
            try:
                if key == self.listen_key and not self.recorder.is_recording:
                    self.record_key_pressed = True
                    LOGGER.info("Record button pressed")
                    self.recorder.start_recording()
            except AttributeError:
                pass

        def on_release(self, key):
            try:
                if key == self.listen_key and self.recorder.is_recording:
                    LOGGER.info("Recording button released")
                    recording_path = self.recorder.stop_recording()
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        executor.submit(self.recorder.transcribe_and_invoke, self.agent_chain, recording_path)
                    self.record_key_pressed = False
            except Exception as e:
                LOGGER.error("Raised an error when stopping recording", exc_info=e)



    key_listener = KeyListener(recorder, agent_chain)
    with keyboard.Listener(
        on_press=key_listener.on_press,
        on_release=key_listener.on_release,
        suppress=False,
        listen_key=listen_key
    ) as listener:
        listener.join()


if __name__ == "__main__":

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()