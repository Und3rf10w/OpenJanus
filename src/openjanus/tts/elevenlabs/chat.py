import asyncio
import logging
from os import getenv

from elevenlabs import set_api_key
from elevenlabs import Voice, VoiceSettings
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.messages import BaseMessage

from openjanus.tts.elevenlabs.tts import ElevenLabsText2SpeechTool
from openjanus.tts.elevenlabs.async_patch import DEFAULT_VOICE


LOGGER = logging.getLogger(__name__)


async def async_run_chat_messages(tts: ElevenLabsText2SpeechTool, chain: BaseLanguageModel, msg: BaseMessage, save_file: bool = False):
    await tts.astream_speech_from_stream(
        chain.stream(msg),
        chunk_size=100,
        save_message=save_file
    )


def run_chat_message(tts: ElevenLabsText2SpeechTool, chain: BaseLanguageModel, msg: BaseMessage, save_file: bool = False):
    output = chain.invoke(msg)
    LOGGER.debug(output)
    LOGGER.info("\n --> Generating voice...")
    tts.play(output, save_message=save_file)


def get_tool() -> ElevenLabsText2SpeechTool:
    set_api_key(getenv("ELEVEN_API_KEY"))
    voice_id = DEFAULT_VOICE.voice_id
    voice_settings = VoiceSettings(
        stability=0.5,
        similarity_boost=0.75,
        style=0,
        use_speaker_boost=False
    )
    voice = Voice(
        voice_id=voice_id,
        settings=voice_settings
    )
    tts = ElevenLabsText2SpeechTool(
        voice=voice
    )
    return tts
