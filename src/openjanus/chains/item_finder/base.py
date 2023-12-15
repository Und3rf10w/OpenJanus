from langchain.agents import Tool
from langchain.agents.agent import Agent, AgentOutputParser
from langchain.agents.conversational_chat.prompt import SUFFIX
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains import LLMChain
from langchain.pydantic_v1 import Field
from langchain.schema.memory import BaseMemory
from langchain.schema.language_model import BaseLanguageModel
from langchain.tools.base import BaseTool
from typing import Any, List, Optional, Sequence

from openjanus.integrations.cornerstone.item_finder import ItemFinder
from openjanus.chains.base import BaseOpenJanusConversationAgent
from openjanus.chains.item_finder.prompt import (
    ITEM_FINDER_SYTEM_PROMPT,
    ITEM_FINDER_USER_PROMPT
)


def _get_tools() -> List[Tool]:
        item_finder = ItemFinder()
        search_item_tool = Tool(
            name="Search_Item",
            description="Use this tool to search for an item. Pass the item name to this tool within the input schema. Run this tool first.",
            func=item_finder.search_items,
        )
        get_item_tool = Tool(
            name="Get_Item",
            description="Use this tool to get the details of an item once you have a listing of items. Pass the item id to this tool within the input schema. Run this tool last.",
            func=item_finder.get_item_details_and_data,
        )
        tools = [
            search_item_tool,
            get_item_tool,
        ]
        return tools


class ItemFinderAgent(BaseOpenJanusConversationAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        memory: BaseMemory | None = None,
        tools: Optional[Sequence[BaseTool]] = None,
        callback_manager: Optional[BaseCallbackManager] = None,
        output_parser: Optional[AgentOutputParser] = None,
        system_message: str = ITEM_FINDER_SYTEM_PROMPT,
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
            # memory=memory,
        )
        tool_names = [tool.name for tool in tools]
        return cls(
            llm_chain=llm_chain,
            allowed_tools=tool_names,
            output_parser=_output_parser,
            **kwargs,
        )
