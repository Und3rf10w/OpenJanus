from abc import ABC
import asyncio
import logging
from typing import Iterator, Optional, Any, AsyncIterator, List
from uuid import UUID

from langchain.agents.openai_functions_multi_agent.base import OpenAIMultiFunctionsAgent
from langchain.callbacks.base import BaseCallbackHandler, AsyncCallbackHandler
from langchain.chains import ConversationChain
from langchain.schema.agent import AgentAction, AgentFinish
from langchain.schema.runnable.utils import (
    Input,
    Output,
)
from langchain.schema.runnable.config import RunnableConfig
from langchain.tools import Tool

from openjanus.tts.elevenlabs.chat import get_tool
# from openjanus.tts.whisper.chat import get_tool


LOGGER = logging.getLogger(__name__)


class AsyncOpenJanusOpenAIFunctionsAgentCallbackHandler(AsyncCallbackHandler):
    async def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run on agent action."""
        LOGGER.debug(action.log)

    async def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run on agent end."""
        finish.return_values
        tts = get_tool()
        await tts.arun({"query": finish.return_values['response']})


class OpenJanusOpenAIFunctionsAgentCallbackHandler(BaseCallbackHandler):
    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run on agent action."""
        LOGGER.debug(action.log)

    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run on agent end."""
        finish.return_values
        tts = get_tool()
        tts.run({"query": finish.return_values['response']})


def get_openjanus_agent_callbacks() -> list:
    """
    Retrieve callbacks that are used for openjanus agents

    :return: a list of callbacks for openjanus agents
    """
    return [AsyncOpenJanusOpenAIFunctionsAgentCallbackHandler(), OpenJanusOpenAIFunctionsAgentCallbackHandler()]


class BaseOpenJanusOpenAIFunctionsAgent(OpenAIMultiFunctionsAgent, ABC):
    callbacks = get_openjanus_agent_callbacks()


class BaseOpenJanusConversationChain(ConversationChain):
    def process(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Iterator[Output]:
        """
        Default implementation of stream, which calls invoke.
        Subclasses should override this method if they support streaming output.
        """ 
        stream = self.invoke(input, config, **kwargs)
        tts = get_tool()
        tts.run({"query": stream['response']})

    async def aprocess(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[Output]:
        """
        Default implementation of astream, which calls ainvoke.
        Subclasses should override this method if they support streaming output.
        """
        stream = self.stream(input, config, **kwargs)
        tts = get_tool()
        response = await tts.arun({"stream": stream})


def tts_agent_tool(**kwargs) -> Tool:
    """
    Speak an output. Always run this tool last

    :return: A tool to use with text to speech
    """
    imported_tts_tool = get_tool() # TODO: Set voice from config
    tts_tool = Tool(
        name=imported_tts_tool.name,
        description=imported_tts_tool.description,
        func=imported_tts_tool.play,
        coroutine=imported_tts_tool.astream_speech_from_stream,
        **kwargs
    )
    return tts_tool
