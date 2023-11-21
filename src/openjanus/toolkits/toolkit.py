from typing import Generator

from langchain.chat_models.base import BaseChatModel
from langchain.memory import ConversationSummaryBufferMemory, ReadOnlySharedMemory
from langchain.pydantic_v1 import BaseModel, Field
from langchain.schema import BaseMemory
from langchain.schema.language_model import BaseLanguageModel
from langchain.tools import Tool

from openjanus.chains.atc.base import AtcChain
from openjanus.chains.onboardia.base import OnboardIaChain
from openjanus.tts.elevenlabs.chat import get_tool as get_elevenlabs_tts_tool


def atc_chain_tool(llm: BaseLanguageModel, memory: BaseMemory, **kwargs) -> Tool:
    """
    Generate a tool to expose the ATC

    :param llm: The LLM object to use
    :param memory: A memory object to use
    :return: A tool with an Air Traffic Controller
    """
    atc_chain = AtcChain(
        llm=llm,
        memory=memory
    )
    atc_tool = Tool(
        name="Reply_ATC",
        description="Use this tool when assuming the role of an Air Traffic Controller",
        func=atc_chain.stream,
        coroutine=atc_chain.astream,
        return_direct=True,
        **kwargs
    )
    return atc_tool


def onboard_ia_chain_tool(llm: BaseLanguageModel, memory: BaseMemory, **kwargs) -> Tool:
    """
    Generate a tool to expose the onboard IA

    :param llm: The LLM object to use
    :param memory: the memory object to use
    :return: A tool with the onboard ship IA
    """
    onboard_ia_chain = OnboardIaChain(
        llm=llm,
        memory=memory
    )
    onboard_ia_tool = Tool(
        name="Reply_Onboard_IA",
        description="Use this tool when assuming the role of an Onboard-Ship IA",
        func=onboard_ia_chain.stream,
        coroutine=onboard_ia_chain.astream,
        return_direct=True,
        **kwargs
    )
    return onboard_ia_tool


def tts_agent_tool(**kwargs) -> Tool:
    """
    Speak an output. Always run this tool last

    :return: A tool to use with text to speech
    """
    from openjanus.tts.elevenlabs.async_patch import DEFAULT_VOICE
    imported_tts_tool = get_elevenlabs_tts_tool() # TODO: Set voice from config
    tts_tool = Tool(
        name=imported_tts_tool.name,
        description=imported_tts_tool.description,
        func=imported_tts_tool.play,
        coroutine=imported_tts_tool.astream_speech_from_stream,
        **kwargs
    )
    return tts_tool


def get_openjanus_tools(llm:BaseChatModel) -> list:
    """
    Return a list of tools for the base chat agent to use

    :param llm: A chat llm with a context window
    :return: A list object of tools (i.e. a toolkit)
    """
    tools = []
    try:
        tools = [
            atc_chain_tool(llm=llm, memory=ConversationSummaryBufferMemory(llm=llm, return_messages=True, memory_key="chat_history")),
            onboard_ia_chain_tool(llm=llm, memory=ConversationSummaryBufferMemory(llm=llm, return_messages=True, memory_key="chat_history")),
            # tts_agent_tool()
        ]
    except ImportError:
        pass
    return tools
