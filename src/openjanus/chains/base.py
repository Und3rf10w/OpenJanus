from abc import ABC
import asyncio
import logging
from typing import Iterator, Optional, Any, AsyncIterator, List, Dict
from uuid import UUID

from langchain.agents.conversational_chat.base import ConversationalChatAgent
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

import openjanus.app.config as openjanus_config
from openjanus.utils.exceptions import TtsNotImplementedException


LOGGER = logging.getLogger(__name__)

config = openjanus_config.load_config()
tts_engine = openjanus_config.get_tts_engine()
if tts_engine.lower() == "elevenlabs":
    from openjanus.tts.elevenlabs.chat import get_tool
elif tts_engine.lower() == "whisper":
    from openjanus.tts.whisper.chat import get_tool
else:
    raise openjanus_config.TtsNotImplementedException(tts_engine)


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
        tts = get_tool()
        await tts.arun({"query": finish.return_values['output']})


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
        # finish.return_values
        tts = get_tool()
        tts.run({"query": finish.return_values['output']})


class OpenJanusChainCallbackHandler(BaseCallbackHandler):
    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when chain ends running."""
        # outputs['response']
        # tts = get_tool()
        # tts.run({"query": outputs['response']})


class AsyncOpenJanusChainCallbackHandler(BaseCallbackHandler):
    async def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when chain ends running."""
        outputs['response']
        tts = get_tool()
        tts.run({"query": outputs['response']})


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
        input: Dict,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> List[Dict[str, Any]]:
        """
        Default implementation of stream, which calls invoke.
        Subclasses should override this method if they support streaming output.
        """ 
        stream = self.invoke(input, config, **kwargs)
        tts = get_tool()
        response = tts.run({"query": stream['response']})
        return [{self.output_key: response}]

    async def aprocess(
        self,
        input: Dict,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> List[Dict[str, Any]]:
        """
        Default implementation of astream, which calls ainvoke.
        Subclasses should override this method if they support streaming output.
        """
        stream = self.stream(input, config, **kwargs)
        tts = get_tool()
        response = await tts.arun({"stream": stream})
        return [{self.output_key: response}]
    

class BaseOpenJanusConversationAgent(ConversationalChatAgent, ABC):
    callbacks = get_openjanus_agent_callbacks()
         


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
