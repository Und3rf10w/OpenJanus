from datetime import datetime
import logging
import pathlib
import shutil
import subprocess
from typing import Any, Coroutine, Dict, Optional, Union, Iterator, Generator, Literal

from langchain.callbacks.manager import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
from langchain.document_loaders.blob_loaders import Blob
from langchain.pydantic_v1 import root_validator
from langchain.tools.base import BaseTool
from langchain.utils import get_from_dict_or_env
from langchain.schema import Document


LOGGER = logging.getLogger(__name__)


class OpenAIWhisperSpeaker(BaseTool):
    """Use OpenAI as a speech to text engine
    Speech generation is with the OpenAI Whisper model."""
    name: str = "OpenAI_Whisper_Speech_to_Text"
    description: str = """Use this tool to speak a response. Pass the entire input unaltered to this tool."""

    def __init__(
            self, 
            api_key: Optional[str] = None, 
            voice_id: Optional[str] = "nova",
            voice_model: Optional[Union[Literal["tts-1"], Literal["tts-1-hd"]]] = "tts-1"
        ) -> None:
        self.api_key = api_key
        self.voice_id = voice_id
        self.voice_model = voice_model
        self.output_file_path = ""

    def is_installed(self, lib_name: str) -> bool:
        lib = shutil.which(lib_name)
        if lib is None:
            return False
        return True

    def set_recording_path(self):
        # TODO: Clean this up, set from config, etc
        output_format = f"output.{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}".replace(' ','_')
        self.output_file_path = str(pathlib.PurePath(output_format))

    # Ripped from elevenlabs
    def stream_audio(self, audio_stream: Iterator[bytes]) -> bytes:
        if not self.is_installed("mpv"):
            message = (
                "mpv not found, necessary to stream audio. "
                "On mac you can install it with 'brew install mpv'. "
                "On linux and windows you can install it from https://mpv.io/"
            )
            raise ValueError(message)

        mpv_command = ["mpv", "--no-cache", "--no-terminal", "--", "fd://0"]
        mpv_process = subprocess.Popen(
            mpv_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        audio = b""

        for chunk in audio_stream:
            if chunk is not None:
                mpv_process.stdin.write(chunk)  # type: ignore
                mpv_process.stdin.flush()  # type: ignore
                audio += chunk

        if mpv_process.stdin:
            mpv_process.stdin.close()
        mpv_process.wait()

        return audio

    def _run(self, text: str, *args: Any, **kwargs: Any) -> Any:
        import io

        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package not found, please install it with "
                "`pip install openai`"
            )
        
        # Set the API key if provided
        if self.api_key:
            openai.api_key = self.api_key
        
        # TODO: Set from config
        self.set_recording_path()

        try:
            response = openai.audio.speech.create(
                model=self.voice_model,
                voice=self.voice_id,
                input=text
            )
            # Write the response to a file directly, not used, but left here for prosperity
            # response.stream_to_file(self.output_file_path)
            
            # Stream the response directly to mpv
            audio = self.stream_audio(audio_stream=response.iter_bytes())

            # Write the response to a file
            with open(self.output_file_path, 'wb') as f:
                f.write(audio)
            
            LOGGER.debug(f"Wrote response to {self.output_file_path}")

            return self.output_file_path
        except Exception as e:
            LOGGER.error("Error received while doing OpenAI Whisper TTS", exc_info=e)

    async def _arun(self, text: str, *args: Any, **kwargs: Any) -> Any:
        # It's going to be the same either way
        # TODO: See if we need to consume the stream from an iterator here? Perhaps text should be a union.
        output_path = self._run(text=text, *args, **kwargs)
        return output_path
