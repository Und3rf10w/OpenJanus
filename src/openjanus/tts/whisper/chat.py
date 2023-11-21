import logging
from os import getenv

from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.messages import BaseMessage

from openjanus.tts.whisper.tts import OpenAIWhisperSpeaker


LOGGER = logging.getLogger(__name__)


def get_tool() -> OpenAIWhisperSpeaker:
    tts = OpenAIWhisperSpeaker()
    return tts