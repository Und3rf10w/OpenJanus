from typing import Optional, List

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.openai_functions.utils import (
    _convert_schema,
    _resolve_schema_references,
    get_llm_kwargs,
)
from langchain.output_parsers.openai_functions import (
    JsonKeyOutputFunctionsParser,
    PydanticAttrOutputFunctionsParser,
)
from langchain.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, BasePromptTemplate, HumanMessagePromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.schema import BasePromptTemplate
from langchain.schema.language_model import BaseLanguageModel

from openjanus.chains.base import BaseOpenJanusConversationChain
from openjanus.chains.onboardia.prompt import (
    ONBOARD_IA_SYSTEM_PROMPT,
    ONBOARD_IA_USER_PROMPT
)


class Keypress(BaseModel):
    """What actions to perform that correspond with an action that the onboard computer is tasked with performing"""
    action: List[str]


class OnboardIaChain(BaseOpenJanusConversationChain):
    """
    A chain for acting like an Onboard Ship IA
    """
    prompt: BasePromptTemplate = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(ONBOARD_IA_SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template(ONBOARD_IA_USER_PROMPT)
    ])


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


def create_keypress_chain(
    schema: dict,
    llm: BaseLanguageModel,
    prompt: Optional[BasePromptTemplate],
    tags: Optional[List[str]] = None,
    verbose: bool = False,
) -> Chain:
    """
    Creates a chain that executes key presses on behalf on the onboard IA

    :param Chain: _description_
    :param llm: _description_
    :param prompt: _description_
    :param tags: _description_, defaults to None
    :param verbose: _description_, defaults to False
    :return: _description_
    """
    function = _get_keypress_function(schema)
    keypress_prompt = prompt # or ChatPromptTemplate.from_template(_KEYPRESS_TEMPLATE)
    output_parser = JsonKeyOutputFunctionsParser(key_name="action")
    llm_kwargs = get_llm_kwargs(function)
    chain = LLMChain(
        llm=llm,
        prompt=keypress_prompt,
        llm_kwargs=llm_kwargs,
        output_parser=output_parser,
        tags=tags,
        verbose=verbose
    )
    return chain