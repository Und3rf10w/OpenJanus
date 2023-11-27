import json
import re
from typing import Any, Callable

# from langchain.output_parsers.json import SimpleJsonOutputParser
import langchain.output_parsers.json as langchain_json_parser
from langchain.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, BasePromptTemplate, HumanMessagePromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.schema import BasePromptTemplate

from openjanus.chains.base import BaseOpenJanusConversationChain
from openjanus.chains.onboardia.prompt import (
    ONBOARD_IA_SYSTEM_PROMPT,
    ONBOARD_IA_USER_PROMPT,
)
from openjanus.chains.onboardia.keypress import(
    run as keypress_run,
    arun as keypress_arun
)



class OnboardIaChain(BaseOpenJanusConversationChain):
    """
    A chain for acting like an Onboard Ship IA
    """
    prompt: BasePromptTemplate = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(ONBOARD_IA_SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template(ONBOARD_IA_USER_PROMPT)
    ])
    # input_key = "input"
    return_final_only = True


class testModel(BaseModel):
    input: str


def _replace_new_line(match: re.Match[str]) -> str:
    value = match.group(2)
    value = re.sub(r"\n", r"\\n", value)
    value = re.sub(r"\r", r"\\r", value)
    value = re.sub(r"\t", r"\\t", value)
    value = re.sub(r'(?<!\\)"', r"\"", value)

    return match.group(1) + value + match.group(3)


def _custom_parser(multiline_string: str) -> str:
    """
    The LLM response for `action_input` may be a multiline
    string containing unescaped newlines, tabs or quotes. This function
    replaces those characters with their escaped counterparts.
    (newlines in JSON must be double-escaped: `\\n`)
    """
    if isinstance(multiline_string, (bytes, bytearray)):
        multiline_string = multiline_string.decode()

    multiline_string = re.sub(
        r'{\s*("actions"\:\s*)(({.*},\s*)({.*"\n\s*})*\s}$)',
        _replace_new_line,
        multiline_string,
        flags=re.DOTALL,
    )

    return multiline_string


def _parse_json_markdown(
    json_string: str, *, parser: Callable[[str], Any] = json.loads
) -> str | None:
    """
    Parse a JSON string from a Markdown string.

    Args:
        json_string: The Markdown string.

    Returns:
        The parsed JSON object as a Python dictionary.
    """
    # Try to find JSON string within triple backticks - patched
    match = re.search(r"```(?:json)?\n(.*)```", json_string, re.DOTALL)

    # If no match found, assume the entire string is a JSON string
    if match is None:
        json_str = json_string
    else:
        # If match found, use the content within the backticks
        json_str = match.group(1)

    # Strip whitespace and newlines from the start and end
    json_str = json_str.strip()

    # handle newlines and other special characters inside the returned value
    json_str = _custom_parser(json_str)

    # Parse the JSON string into a Python dictionary
    parsed = parser(json_str)

    output = keypress_run(parsed)
    return output
