from langchain.agents.tools import Tool
from langchain.agents.agent import Agent, AgentOutputParser
from langchain.agents.conversational_chat.prompt import SUFFIX
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains import LLMChain
from langchain.pydantic_v1 import Field
from langchain.schema.memory import BaseMemory
from langchain.schema.language_model import BaseLanguageModel
from langchain.tools.base import BaseTool
from typing import Any, List, Optional, Sequence

from openjanus.integrations.cornerstone.planetary_survey import PlanetarySurvey
from openjanus.chains.base import BaseOpenJanusConversationAgent
from openjanus.chains.planetary_survey.prompt import (
    PLANETARY_SURVEY_SYTEM_PROMPT,
    PLANETARY_SURVEY_USER_PROMPT,
)


def _get_tools() -> List[Tool]:
        planetary_survey = PlanetarySurvey()
        search_location_tool = Tool.from_function(name="Search",
            description="Use this tool to search survey data for a location. Pass the location/system/planetary object's name as `search_name`, and one of `('SystemsV2', 'PlanetsV2', 'LocationsV2')` as `category`, along with name to this tool within the input schema. Input should be a single string delimited by a comma, e.g. `Hurston, PlanetsV2` Run this tool.",
            func=planetary_survey._search_helper,
        )
        tools = [
            search_location_tool
        ]
        return tools


class PlanetarySurveyAgent(BaseOpenJanusConversationAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Optional[Sequence[BaseTool]] = None,
        callback_manager: Optional[BaseCallbackManager] = None,
        output_parser: Optional[AgentOutputParser] = None,
        system_message: str = PLANETARY_SURVEY_SYTEM_PROMPT,
        human_message: str = SUFFIX,
        input_variables: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Agent:
        """Construct an agent from an LLM and tools."""
        if tools is None:
            tools = _get_tools()
        cls._validate_tools(tools)
        _output_parser = output_parser or cls._get_default_output_parser()
        prompt = cls.create_prompt(
            tools,
            system_message=system_message,
            human_message=human_message,
            input_variables=input_variables,
            output_parser=_output_parser,
        )
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            callback_manager=callback_manager,
        )
        tool_names = [tool.name for tool in tools]
        return cls(
            llm_chain=llm_chain,
            allowed_tools=tool_names,
            output_parser=_output_parser,
            **kwargs,
        )
