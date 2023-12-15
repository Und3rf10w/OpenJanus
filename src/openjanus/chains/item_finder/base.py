from langchain.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, BasePromptTemplate, HumanMessagePromptTemplate

from openjanus.chains.base import BaseOpenJanusConversationChain
from openjanus.chains.item_finder.prompt import (
    ITEM_FINDER_SYTEM_PROMPT,
    ITEM_FINDER_USER_PROMPT
)


class ItemFinderChain(BaseOpenJanusConversationChain):
    """
    A chain for acting like an Item Finder API
    """
    prompt: BasePromptTemplate = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(ITEM_FINDER_SYTEM_PROMPT),
        HumanMessagePromptTemplate.from_template(ITEM_FINDER_USER_PROMPT)
    ])
    input_key = "input"
    return_final_only = True