
from langchain.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, BasePromptTemplate, HumanMessagePromptTemplate

from openjanus.chains.base import BaseOpenJanusConversationChain
from openjanus.chains.atc.prompt import (
    ATC_SYSTEM_PROMPT,
    ATC_USER_PROMPT
)


class AtcChain(BaseOpenJanusConversationChain):
    """
    A chain for acting like an Air Traffic Controller
    """
    prompt: BasePromptTemplate = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(ATC_SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template(ATC_USER_PROMPT)
    ])