from abc import ABC
from operator import itemgetter
from typing import Any, Optional, List, Union, Dict

from langchain.agents import BaseSingleActionAgent
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationChain
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.openai_functions.utils import (
    _convert_schema,
    _resolve_schema_references,
    get_llm_kwargs,
)
from langchain.memory import SimpleMemory
from langchain.output_parsers.openai_functions import (
    JsonKeyOutputFunctionsParser,
    PydanticAttrOutputFunctionsParser,
)
from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, BasePromptTemplate, HumanMessagePromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.schema import BasePromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.memory import BaseMemory
from langchain.schema.messages import SystemMessage
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import (
    RunnableParallel, 
    RunnableSequence, 
    RunnablePassthrough, 
    RunnableMap, 
    Runnable
)
from langchain.tools import Tool, StructuredTool
from langchain.utils.openai_functions import convert_pydantic_to_openai_tool

from openjanus.chains.base import BaseOpenJanusConversationChain, BaseOpenJanusOpenAIFunctionsAgent
from openjanus.chains.onboardia.prompt import (
    ONBOARD_IA_KEYMAP_USER_PROMPT,
    ONBOARD_IA_SYSTEM_PROMPT,
    ONBOARD_IA_USER_PROMPT,
    ONBOARD_IA_KEYMAP_PROMPT
)
from openjanus.chains.onboardia.keypress import(
    run as keypress_run,
    arun as keypress_arun
)

class KeypressSchema(BaseModel):
    """What actions to perform that correspond with an action that the onboard computer is tasked with performing"""
    input: str
    keypresses: Union[Dict, List]


class OnboardIaChain(BaseOpenJanusConversationChain):
    """
    A chain for acting like an Onboard Ship IA
    """
    prompt: BasePromptTemplate = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(ONBOARD_IA_SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template(ONBOARD_IA_USER_PROMPT)
    ])


class OnboardIaAgent(BaseOpenJanusOpenAIFunctionsAgent, ABC):
    """
    An agent to act like an Onboard Ship IA with button press support
    """


def _get_keypress_function(entity_schema: dict) -> dict:
    return {
        "name": "ship_ia_keypress",
        "description": "Select the key to press based on the desired action or actions",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "array", "items": _convert_schema(schema=entity_schema)}
            },
            "required": ["action"]
        },
    }

def get_ship_actions(input):
    ship_action= Tool.from_function(
        name="Perform_Action",
        description="Use this tool next to perform actions. Pass the entire original output from Retrieve_keypresses directly to this tool.",
        func=keypress_run,
        coroutine=keypress_arun,
        verbose=True
    )
    output = ship_action.run(input)
    return {"actions": output}

def create_parallel_onboard_ia_chain(
    llm: BaseLanguageModel,
    memory: BaseMemory,
    tags: Optional[List[str]] = None,
    verbose: bool = False,
    **kwargs
# ) -> (OnboardIaChain, List[Tool]):
) -> Runnable:
    """
    Which key button to press

    :param llm: An llm to use for the chains and agent
    :param memory: The memory object to use for the OnboardIAAgent
    :param tags: Tags to associate with the Chain, defaults to None
    :param verbose: Whether to run the chain in verbose mode, defaults to False
    :param kwargs: kwargs to pass to the agent executor
    :return: An agent executor
    """
    # function = _get_keypress_function(entity_schema=KeypressSchema)
    # keypress_prompt = ChatPromptTemplate.from_messages(messages={
    #     ("system", ONBOARD_IA_KEYMAP_PROMPT),
    #     ("user", "{input}")
    # })
    # output_parser = JsonKeyOutputFunctionsParser(key_name="action")
    # llm_kwargs = get_llm_kwargs(function)
    # chain = LLMChain(
    #     llm=llm,
    #     prompt=keypress_prompt,
    #     llm_kwargs=llm_kwargs,
    #     output_parser=output_parser,
    #     tags=tags,
    #     verbose=verbose
    # )
    # keypress_tools = [convert_pydantic_to_openai_tool(KeypressSchema)]
    # model = llm.bind(tools=keypress_tools)
    # keypress_runnable_chain = keypress_prompt | model | PydanticToolsParser(tools=[KeypressSchema])
    # keypress_tool = Tool.from_function(
    #     func=keypress_runnable_chain.invoke,
    #     coroutine=keypress_runnable_chain.ainvoke,
    #     name="Perform_Ship_Action",
    #     description="Use first to convert an input request to an action that needs to be perormed on the ship",
    #     verbose=verbose,
    #     args_schema=KeypressSchema,
    # )
    onboard_ia_chain = OnboardIaChain(memory=memory, llm=llm, verbose=verbose, **kwargs)

    # keypress_mapgen_chain = ConversationChain(
    #     llm=llm,
    #     memory=SimpleMemory(memories={"chat_history": ""}),
    #     verbose=verbose,
    #     tags=tags,
    #     prompt=ChatPromptTemplate.from_messages([
    #         SystemMessagePromptTemplate.from_template(ONBOARD_IA_KEYMAP_PROMPT),
    #         HumanMessagePromptTemplate.from_template(ONBOARD_IA_USER_PROMPT)
    #     ]),
    # )
    keypress_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(template=ONBOARD_IA_KEYMAP_PROMPT),
            HumanMessagePromptTemplate.from_template(ONBOARD_IA_KEYMAP_USER_PROMPT)
        ])
    
    # Input: "Turn on the lights and deploy the landing gear"
    # Output: { "Toggle Ship Lights": { "keys": ["L"], "hold": false }, "Landing Gear": { "keys": ["N"], "hold": false } }
    # keypress_tool = Tool.from_function(
    #     name="Retrieve_Keypresses",
    #     description="Use this tool first to retrieve a keypress mapping that you need to perform actions onboard the ship on behalf of the user",
    #     func=keypress_mapgen_chain.invoke,
    #     coroutine=keypress_mapgen_chain.ainvoke,
    #     verbose=verbose,
    # )
    # map_ = RunnableMap(input=RunnablePassthrough())
    ship_action= StructuredTool.from_function(
        name="Perform_Action",
        description="Use this tool next to perform actions. Pass the entire original output from Retrieve_keypresses directly to this tool.",
        func=keypress_run,
        coroutine=keypress_arun,
        verbose=True,
        args_schema=KeypressSchema
    )
    
    keypress_chain = {"input": RunnablePassthrough()} | RunnablePassthrough.assign(action_map=itemgetter("input")) | keypress_prompt | llm | SimpleJsonOutputParser()
    # ship_action_chain = Runnable(get_ship_actions)
    chain = (
        RunnablePassthrough.assign(keypresses=keypress_chain)
        | ship_action
        | {"input": RunnablePassthrough()}
        # | StrOutputParser()
        | onboard_ia_chain
    )
    # ship_action = {RunnablePassthrough.assign(actions=keypress_chain) | StrOutputParser()}

    # chain = RunnableSequence(keypress_chain | 
    #                         # {"actions": ship_action} | 
    #                         RunnablePassthrough.assign(input=itemgetter("input"), actions=ship_action) |onboard_ia_chain
    #                         )
    return chain
