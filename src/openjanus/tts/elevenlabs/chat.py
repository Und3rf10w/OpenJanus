import asyncio
import logging
from os import getenv

from elevenlabs import set_api_key
from elevenlabs import Voice, VoiceSettings
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.messages import BaseMessage

from openjanus.app.config import get_elevenlabs_config
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
    elevenlabs_config = get_elevenlabs_config()
    set_api_key(getenv("ELEVEN_API_KEY"))
    voice_id = elevenlabs_config['elevenlabs_voice_id']
    voice_settings = VoiceSettings(
        stability=elevenlabs_config['elevenlabs_stability'],
        similarity_boost=elevenlabs_config['elevenlabs_similarity_boost'],
        style=elevenlabs_config['elevenlabs_style'],
        use_speaker_boost=elevenlabs_config['elevenlabs_use_speaker_boost']
    )
    voice = Voice(
        voice_id=voice_id,
        settings=voice_settings
    )
    tts = ElevenLabsText2SpeechTool(
        voice=voice
    )
    return tts
