import logging
from os import getenv
from typing import Optional

from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.messages import BaseMessage

from openjanus.tts.whisper.tts import OpenAIWhisperSpeaker


LOGGER = logging.getLogger(__name__)


def get_tool(api_key: Optional[str] = None) -> OpenAIWhisperSpeaker:
    if api_key:
        tts = OpenAIWhisperSpeaker(api_key=api_key)
    else:
        tts = OpenAIWhisperSpeaker()
        
    return tts