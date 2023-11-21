import asyncio
import logging
from typing import Iterator, Optional, Any, AsyncIterator

from langchain.chains import ConversationChain
from langchain.schema.runnable.utils import (
    Input,
    Output,
)
from langchain.schema.runnable.config import RunnableConfig
from langchain.tools import Tool

from openjanus.tts.elevenlabs.chat import get_tool


LOGGER = logging.getLogger(__name__)


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
    from openjanus.tts.elevenlabs.async_patch import DEFAULT_VOICE
    imported_tts_tool = get_tool() # TODO: Set voice from config
    tts_tool = Tool(
        name=imported_tts_tool.name,
        description=imported_tts_tool.description,
        func=imported_tts_tool.play,
        coroutine=imported_tts_tool.astream_speech_from_stream,
        **kwargs
    )
    return tts_tool
