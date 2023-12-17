import asyncio
from cgitb import reset

import concurrent.futures
import logging
from pynput import keyboard
import threading

from langchain.agents import AgentExecutor
from langchain.agents.conversational_chat.base import ConversationalChatAgent
from langchain.chat_models import ChatOpenAI

import openjanus.app.config as openjanus_config
from openjanus.app.banner import banner
from openjanus.chains.prompt import BASE_AGENT_SYSTEM_PROMPT_PREFIX
from openjanus.toolkits.toolkit import get_openjanus_tools
from openjanus.stt.whisper.recorder import Recorder
from openjanus.utils.exceptions import ListenKeyNotSupportedException


logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)


class KeyListener:
        def __init__(self, recorder: Recorder, agent_chain: AgentExecutor, listen_key: str):
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
            if key == self.listen_key and self.recorder.is_recording:
                LOGGER.info("Recording button released")
                self.record_key_pressed = False
                recording_path = self.recorder.stop_recording()
                
                def run_async_process():
                    yellow = '\033[93m'
                    reset = '\033[0m'
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(self.recorder.transcribe_and_invoke(self.agent_chain, recording_path))
                    print(yellow + "Ready to record again" + reset)
                    # loop.close()

                if recording_path:
                    thread = threading.Thread(target=run_async_process)
                    thread.start()



if __name__ == "__main__":
    print(banner())
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
        verbose=True,
    )
    recorder = Recorder()
    listen_key = config["openjanus"]["listen_key"]

    key_listener = KeyListener(recorder, agent_chain, listen_key)
    with keyboard.Listener(
        on_press=key_listener.on_press,
        on_release=key_listener.on_release,
        suppress=False,
        listen_key=listen_key
    ) as listener:
        listener.join()

    while True:
        try:
            pass
        except KeyboardInterrupt:
            break


