import asyncio
from datetime import datetime
import logging
import pathlib
import shutil
import subprocess
from typing import Any, Optional, Union, Iterator, Literal

from langchain.tools.base import BaseTool

from openjanus.app.config import get_recordings_dir


LOGGER = logging.getLogger(__name__)


class OpenAIWhisperSpeaker(BaseTool):
    """Use OpenAI as a speech to text engine
    Speech generation is with the OpenAI Whisper model."""
    name: str = "OpenAI_Whisper_Speech_to_Text"
    description: str = """Use this tool to speak a response. Pass the entire input unaltered to this tool."""
    api_key: Optional[str]
    voice_id: Optional[Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"]] = "nova"
    voice_model: Optional[Union[str, Literal["tts-1", "tts-1-hd"]]]
    output_dir: str = get_recordings_dir()
    output_file_path: Optional[str] = ""
    verbose: bool = True

    def __init__(
            self, 
            api_key: Optional[str] = None, 
            voice_id: Optional[Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"]] = "nova",
            voice_model: Optional[Union[str, Literal["tts-1", "tts-1-hd"]]] = "tts-1",
            output_dir: str = get_recordings_dir(),
            *args,
            **kwargs
        ) -> None:
        super().__init__(*args, **kwargs)
        self.api_key = api_key
        self.voice_id = voice_id
        self.voice_model = voice_model
        self.output_dir = output_dir
        self.output_file_path = ""

    def is_installed(self, lib_name: str) -> bool:
        lib = shutil.which(lib_name)
        if lib is None:
            return False
        return True

    def set_recording_path(self):
        # TODO: Clean this up, set from config, etc
        output_format = self.output_dir + f"output.{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.mp3".replace(' ','_')
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
    
    # Ripped from elevenlabs
    def play(self, audio: bytes, notebook: bool = False, use_ffmpeg: bool = True) -> None:
        if notebook:
            from IPython.display import Audio, display

            display(Audio(audio, rate=44100, autoplay=True))
        elif use_ffmpeg:
            if not self.is_installed("ffplay"):
                message = (
                    "ffplay from ffmpeg not found, necessary to play audio. "
                    "On mac you can install it with 'brew install ffmpeg'. "
                    "On linux and windows you can install it from https://ffmpeg.org/"
                )
                raise ValueError(message)
            args = ["ffplay", "-autoexit", "-", "-nodisp"]
            proc = subprocess.Popen(
                args=args,
                stdout=subprocess.PIPE,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            out, err = proc.communicate(input=audio)
            proc.poll()
        else:
            try:
                import io

                import sounddevice as sd
                import soundfile as sf
            except ModuleNotFoundError:
                message = (
                    "`pip install sounddevice soundfile` required when `use_ffmpeg=False` "
                )
                raise ValueError(message)
            sd.play(*sf.read(io.BytesIO(audio)))
            sd.wait()

    def _run(self, query: str, *args: Any, **kwargs: Any) -> Any:
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
                model=self.voice_model,  # type: ignore
                voice=self.voice_id,  # type: ignore
                input=query
            )
            # Write the response to a file directly, not used, but left here for prosperity
            # response.stream_to_file(self.output_file_path)
            
            # Stream the response directly to mpv
            audio = self.stream_audio(audio_stream=response.iter_bytes())

            # Write the response to a file
            with open(self.output_file_path, 'wb') as f:  # type: ignore
                f.write(audio)
            
            LOGGER.debug(f"Wrote response to {self.output_file_path}")

            return query
        except Exception as e:
            LOGGER.error("Error received while doing OpenAI Whisper TTS", exc_info=e)

    async def _arun(self, stream, *args: Any, **kwargs: Any) -> Any:
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

        # TODO: Consume from stream
        # Define a function to process messages in chunks
        def chunk_messages(messages, chunk_size=100):
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
            audio_bytes, chunk_text = await queue.get()  # TODO: Save message flag here
            try:
                LOGGER.info("Playing audio...")
                if audio_bytes is None:
                    return chunk_text
                # Write the response to a file
                # LOGGER.debug(f"Wrote response to {self.output_file_path}")
                if chunk_text:
                    LOGGER.debug(chunk_text)
                self.stream_audio(audio_stream=iter([audio_bytes]))
                LOGGER.info("Ending audio stream")
                with open(self.output_file_path, 'wb') as f:  # type: ignore
                    f.write(audio_bytes)
                return chunk_text
            except TypeError:
                return chunk_text

        async def process_chunks(chunk_text):
            audio_bytes = []
            audio_chunk = openai.audio.speech.create(
                model=self.voice_model,  # type: ignore
                voice=self.voice_id,  # type: ignore
                input=chunk_text
            )
            audio_bytes.append(audio_chunk.content)
            audio = b''.join(audio_bytes)
            await queue.put((audio, chunk_text))

        queue = asyncio.Queue()
        audio_task = asyncio.create_task(process_audio(queue))

        tasks = []
        for combined_message in chunk_messages(stream):
            tasks.append(asyncio.create_task(process_chunks(combined_message)))

        await asyncio.gather(*tasks)
        await queue.put(None)
        return await audio_task

    async def astream_speech_from_stream(self, stream, *args: Any, **kwargs: Any) -> Any:
        return await self._arun(stream, *args, **kwargs)