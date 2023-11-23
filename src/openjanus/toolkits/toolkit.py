from gc import callbacks
from typing import Generator

from langchain.chat_models.base import BaseChatModel
from langchain.memory import ConversationSummaryBufferMemory, ReadOnlySharedMemory
from langchain.pydantic_v1 import BaseModel, Field
from langchain.schema import BaseMemory
from langchain.schema.language_model import BaseLanguageModel
from langchain.tools import Tool

from openjanus.chains.base import AsyncOpenJanusChainCallbackHandler, OpenJanusChainCallbackHandler
from openjanus.chains.atc.base import AtcChain
from openjanus.chains.onboardia.base import OnboardIaChain, create_parallel_onboard_ia_chain


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
        description="Use this tool to assume the role of an Air Traffic Controller. Pass the user's entire question unaltertered to this tool.",
        func=atc_chain.process,
        coroutine=atc_chain.aprocess,
        return_direct=True,
        verbose=True,
        **kwargs
    )
    return atc_tool


# def onboard_ia_chain_tool(llm: BaseLanguageModel, memory: BaseMemory, **kwargs) -> Tool:
#     """
#     Generate a tool to expose the onboard IA

#     :param llm: The LLM object to use
#     :param memory: the memory object to use
#     :return: A tool with the onboard ship IA
#     """
#     onboard_ia_chain = OnboardIaChain(
#         llm=llm,
#         memory=memory
#     )
#     onboard_ia_tool = Tool(
#         name="Reply_Onboard_IA",
#         description="Use this tool to assume the role of an Onboard-Ship IA. Pass the user's entire question unaltertered to this tool.",
#         func=onboard_ia_chain.process,
#         coroutine=onboard_ia_chain.aprocess,
#         return_direct=True,
#         verbose=True,
#         **kwargs
#     )
#     return onboard_ia_tool

def onboard_ia_chain_tool(llm: BaseLanguageModel, memory: BaseMemory, **kwargs) -> Tool:
    """
    Generate a tool to expose the onboard IA

    :param llm: The LLM object to use
    :param memory: the memory object to use
    :return: A tool with the onboard ship IA
    """
    onboard_ia_chain = create_parallel_onboard_ia_chain(llm=llm, memory=memory, verbose=True, callbacks=[AsyncOpenJanusChainCallbackHandler, OpenJanusChainCallbackHandler])
    onboard_ia_tool = Tool(
        name="Reply_Onboard_IA",
        description="Use this tool to assume the role of an Onboard-Ship IA. Pass the user's entire question unaltertered to this tool.",
        func=onboard_ia_chain.invoke,
        coroutine=onboard_ia_chain.ainvoke,
        return_direct=True,
        verbose=True,
        **kwargs
    )
    return onboard_ia_tool

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
            # onboard_ia_chain_tool(llm=llm, memory=ConversationSummaryBufferMemory(llm=llm, return_messages=True, memory_key="chat_history")),
            # tts_agent_tool()
        ]
    except ImportError:
        pass
    return tools
