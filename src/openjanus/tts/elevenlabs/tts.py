from datetime import datetime
from enum import Enum
import logging
import tempfile
from typing import Any, Coroutine, Dict, Optional, Union, Iterator, Generator

import asyncio

from elevenlabs import Voice
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.pydantic_v1 import root_validator
from langchain.tools.base import BaseTool
from langchain.utils import get_from_dict_or_env
import openjanus.tts.elevenlabs.async_patch as eleven_labs_async_patch


LOGGER = logging.getLogger(__name__)


def _import_elevenlabs() -> Any:
    try:
        import elevenlabs
        elevenlabs.generate_stream_input_async = eleven_labs_async_patch.generate_stream_input_async
        elevenlabs.agenerate = eleven_labs_async_patch.agenerate
        elevenlabs.TTS.generate_stream_input_async = eleven_labs_async_patch.generate_stream_input_async

    except ImportError as e:
        raise ImportError(
            "Cannot import elevenlabs, please install `pip install elevenlabs`."
        ) from e
    return elevenlabs


class ElevenLabsModel(str, Enum):
    """Models available for Eleven Labs Text2Speech."""

    MULTI_LINGUAL = "eleven_multilingual_v1"
    MONO_LINGUAL = "eleven_monolingual_v1"
    MULTI_LINGUAL_V2 = "eleven_multilingual_v2"


class ElevenLabsText2SpeechTool(BaseTool):
    """Tool that queries the Eleven Labs Text2Speech API.

    In order to set this up, follow instructions at:
    https://docs.elevenlabs.io/welcome/introduction
    """

    model: Union[ElevenLabsModel, str] = ElevenLabsModel.MULTI_LINGUAL_V2

    name: str = "eleven_labs_text2speech"
    description: str = (
        "A wrapper around Eleven Labs Text2Speech. "
        "Useful for when you need to convert text to speech. "
        "It supports multiple languages, including English, German, Polish, "
        "Spanish, Italian, French, Portuguese, and Hindi. "
    )
    voice: Voice

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists in environment."""
        _ = get_from_dict_or_env(values, "eleven_api_key", "ELEVEN_API_KEY")

        return values

    def save_file(self, audio: Union[bytes, Iterator[bytes]]):
        if isinstance(audio, Iterator):
            raw_audio = iter(audio)
        else:
            raw_audio = audio
        elevenlabs = _import_elevenlabs()
        now = datetime.now()
        formatted_dt = now.strftime(format="%Y%m%d_%H%M%S")
        elevenlabs.save(raw_audio, f"{formatted_dt}_{self.voice.voice_id}_chat.mp3")

    def _run(
        self, query, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        elevenlabs = _import_elevenlabs()
        try:
            speech = elevenlabs.generate(text=query, model=self.model, voice=self.voice)
            elevenlabs.play(speech)
            # with tempfile.NamedTemporaryFile(
            #     mode="bx", suffix=".wav", delete=False
            # ) as f:
            #     f.write(speech)
            # return f.name
        except Exception as e:
            raise RuntimeError(f"Error while running ElevenLabsText2SpeechTool: {e}")
        
    async def _arun(self, stream, **kwargs: Any) -> Coroutine[Any, Any, Any]:
        """Play text to speech from a stream"""
        try:
            await self.astream_speech_from_stream(
                text_stream=stream,
                chunk_size=100,
                save_message=False,
            )
        except Exception as e:
            raise RuntimeError(f"Error while running ElevenLabsText2SpeechTool: {e}")
    



    def play(self, query: str, save_message: bool = False) -> None:
        """
        Play the speech as text

        :param query: The speech to play
        :param save_message: Whether to save the generated text, defaults to False
        """
        elevenlabs = _import_elevenlabs()
        audio = elevenlabs.generate(text=query, voice=self.voice, model=self.model, stream=False, latency=2)
        elevenlabs.play(audio)
        if save_message:
            self.save_file(audio)

    def stream_speech(self, text_stream) -> None:
        """Stream the text as speech as it is generated.
        Play the text in your speakers."""
        elevenlabs = _import_elevenlabs()

        for message in text_stream:
            query = message.content
            speech_stream = elevenlabs.generate(text=query, voice=self.voice, model=self.model, stream=True, latency=2)
            elevenlabs.stream(speech_stream)

    async def aprocess_message(self, query, save_message):
        elevenlabs = _import_elevenlabs()
        LOGGER.debug(query)
        # speech_stream = elevenlabs.generate(text=query, voice=self.voice, model=self.model, stream=True, latency=4)
        # audio = elevenlabs.stream(speech_stream)
        # if save_message:
        #     self.save_file(audio)
        audio_chunks = []
        async for chunk in elevenlabs.agenerate(query, voice=self.voice, model=self.model, stream=True, latency=0):
            audio_chunks.append(chunk)

        elevenlabs.stream(audio_chunks)

        if save_message:
            self.save_file(b''.join(audio_chunks))

    async def astream_speech(self, text_stream, save_message: bool = False) -> None:
        async def async_generator_to_list(async_generator):
            return [item async for item in async_generator]

        # Convert the async generator to a list of coroutines
        message_coroutines = await async_generator_to_list(text_stream)

        # Create tasks from the coroutines and execute them concurrently
        tasks = [self.aprocess_message(message, save_message) for message in message_coroutines]
        for future in asyncio.as_completed(tasks):
            result = await future  # result is not used in this case

    async def astream_speech_from_stream(self, text_stream, chunk_size: int = 1000, save_message: bool = False) -> None:
        """
        Play a text stream with TTS

        :param text_stream: The text stream generator object to use
        :param chunk_size: The size of chunks to generate audio for, defaults to 1000, you don't need to provide this
        :param save_message: Whether to save the message, defaults to False, you don't need to provide this
        """
        elevenlabs = _import_elevenlabs()
        # Define a function to process messages in chunks
        def chunk_messages(messages, chunk_size):
            chunk = []
            for message in messages:
                chunk.append(message['response'])
                # chunk.append(message)
                if len(chunk) == chunk_size:
                    yield ''.join(chunk)
                    chunk = []
            if chunk:
                yield ''.join(chunk)

        async def process_audio(queue: asyncio.Queue):
            while True:
                try:
                    audio_bytes, chunk_text, save_message_flag = await queue.get()
                    LOGGER.info("Playing audio...")
                    if audio_bytes is None:
                        break
                    if save_message_flag:
                        self.save_file(audio_bytes)
                    if chunk_text:
                        LOGGER.debug(chunk_text)
                    elevenlabs.stream(iter([audio_bytes]))
                    LOGGER.info("Ending audio stream")
                except TypeError:
                    break

        async def process_chunks(chunk_text):
            audio_bytes = []
            async for chunk in elevenlabs.agenerate(chunk_text, voice=self.voice, model=self.model, stream=True, latency=0):
                audio_bytes.append(chunk)
            audio = b''.join(audio_bytes)
            await queue.put((audio, chunk_text, save_message))

        queue = asyncio.Queue()
        audio_task = asyncio.create_task(process_audio(queue))

        tasks = []
        for combined_message in chunk_messages(text_stream, chunk_size):
            tasks.append(asyncio.create_task(process_chunks(combined_message)))

        await asyncio.gather(*tasks)
        await queue.put(None)
        await audio_task
