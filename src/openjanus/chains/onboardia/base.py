from langchain.chains import ConversationChain
from langchain.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, BasePromptTemplate, HumanMessagePromptTemplate

from openjanus.chains.onboardia.prompt import (
    ONBOARD_IA_SYSTEM_PROMPT,
    ONBOARD_IA_USER_PROMPT
)

class OnboardIaChain(ConversationChain):
    """
    A chain for acting like an Onboard Ship IA
    """
    prompt: BasePromptTemplate = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(ONBOARD_IA_SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template(ONBOARD_IA_USER_PROMPT)
    ])