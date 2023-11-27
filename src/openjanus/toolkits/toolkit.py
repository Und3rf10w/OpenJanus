from typing import Dict

from langchain.chains import SequentialChain
from langchain.chains.llm import LLMChain
from langchain.chat_models.base import BaseChatModel
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langchain.schema import BaseMemory
from langchain.schema.language_model import BaseLanguageModel
from langchain.tools import Tool

from openjanus.chains.base import AsyncOpenJanusChainCallbackHandler, OpenJanusChainCallbackHandler
from openjanus.chains.atc.base import AtcChain
from openjanus.chains.onboardia.base import (
    OnboardIaChain,
    _parse_json_markdown, langchain_json_parser
)
from openjanus.chains.onboardia.prompt import (
    ONBOARD_IA_KEYMAP_USER_PROMPT,
    ONBOARD_IA_KEYMAP_PROMPT
)


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


class InnerInputModel(BaseModel):
    input: str


class OnboardIAChainInputSchema(BaseModel):
    """What actions to perform that correspond with an action that the onboard computer is tasked with performing"""
    input: Dict[str, str]


def onboard_ia_chain_tool(llm: BaseLanguageModel, memory: BaseMemory, **kwargs) -> Tool:
    """
    Generate a tool to expose the onboard IA

    :param llm: The LLM object to use
    :param memory: the memory object to use
    :return: A tool with the onboard ship IA
    """
    keypress_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(template=ONBOARD_IA_KEYMAP_PROMPT),
            HumanMessagePromptTemplate.from_template(ONBOARD_IA_KEYMAP_USER_PROMPT)
        ])
    keypress_prompt.input_variables = ['input']
    langchain_json_parser.parse_json_markdown = _parse_json_markdown
    PatchedSimpleJsonOutputParser = langchain_json_parser.SimpleJsonOutputParser
    keypress_chain = LLMChain(memory=None, prompt=keypress_prompt, llm=llm)
    keypress_chain.output_parser = PatchedSimpleJsonOutputParser()
    chains = [keypress_chain,
              OnboardIaChain(memory=memory, llm=llm, verbose=True, callbacks=[AsyncOpenJanusChainCallbackHandler(), OpenJanusChainCallbackHandler()])]
    onboard_ia_chain = SequentialChain(memory=memory, verbose=True, chains=chains, input_variables=['input'], output_variables=['response'])
    onboard_ia_tool = Tool(
        name="Reply_Onboard_IA",
        description="Use this tool to assume the role of an Onboard Ship-AI/Computer. This tool can be used to perform actions related to this ship. Pass the user's entire question unaltertered to this tool within the input schema.",
        func=onboard_ia_chain.invoke,
        coroutine=onboard_ia_chain.ainvoke,
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
            onboard_ia_chain_tool(llm=llm, memory=ConversationSummaryBufferMemory(llm=llm, return_messages=True, memory_key="chat_history", output_key="response", input_key="input")),
        ]
    except ImportError:
        pass
    return tools
