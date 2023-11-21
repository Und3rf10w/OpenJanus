from langchain.chat_models.base import BaseChatModel
from langchain.memory import ConversationSummaryBufferMemory, ReadOnlySharedMemory
from langchain.schema import BaseMemory
from langchain.schema.language_model import BaseLanguageModel
from langchain.tools import Tool

from openjanus.chains.atc.base import AtcChain
from openjanus.chains.onboardia.base import OnboardIaChain


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
        name="Reply as an Air-Traffic-Controller",
        description="Use this tool when assuming the role of an Air Traffic Controller",
        func=atc_chain.call,
        coroutine=atc_chain.arun,
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
        name="Reply as the ship's onboard intelligent assistant",
        description="Use this tool when assuming the role of an Onboard-Ship IA",
        func=onboard_ia_chain.run,
        coroutine=onboard_ia_chain.arun,
        return_direct=True,
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
        tools.append(
            atc_chain_tool(llm=llm, memory=ConversationSummaryBufferMemory(return_messages=True)),
            onboard_ia_chain_tool(llm=llm, memory=ConversationSummaryBufferMemory(return_messages=True))
        )
    except ImportError:
        pass
    return tools
