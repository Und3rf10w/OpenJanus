import asyncio
import base64
import json
import logging
import os
from typing import Iterator, Optional, Union
import websockets
from websockets.sync.client import connect

from elevenlabs.api.tts import TTS, Voice, Model, API, api_base_url_v1, text_chunker
from elevenlabs import VoiceSettings, is_voice_id


LOGGER = logging.getLogger(__name__)


async def async_text_chunker(chunks):
    splitters = (".", ",", "?", "!", ";", ":", "—", "-", "(", ")", "[", "]", "}", " ")
    buffer = ""
    async for text in chunks:
        if buffer.endswith(splitters):
            yield buffer if buffer.endswith(" ") else buffer + " "
            buffer = text
        elif text.startswith(splitters):
            output = buffer + text[0]
            yield output if output.endswith(" ") else output + " "
            buffer = text[1:]
        else:
            buffer += text
    if buffer != "":
        yield buffer + " "


async def generate_stream_input_async(text, voice: Voice, model: Model, api_key: Optional[str] = None):  #-> AsyncIterator[bytes]:
    BOS = json.dumps(
        dict(
            text=" ",
            try_trigger_generation=True,
            voice_settings=voice.settings.model_dump() if voice.settings else None,
            generation_config=dict(
                chunk_length_schedule=[50],
            ),
        )
    )
    EOS = json.dumps(dict(text=""))

    async with connect(
            f"wss://api.elevenlabs.io/v1/text-to-speech/{voice.voice_id}/stream-input?model_id={model.model_id}",
            additional_headers={
                "xi-api-key": api_key or os.environ.get("ELEVEN_API_KEY")
            },
    ) as websocket:
        # Send beginning of stream
        await websocket.send(BOS)

        # Send beginning of stream
        await websocket.send(BOS)

        # Stream text chunks and receive audio
        async for text_chunk in async_text_chunker(text):
            data = dict(text=text_chunk, try_trigger_generation=True)
            await websocket.send(json.dumps(data))
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=0.0001)
                data = json.loads(response)
                if data["audio"]:
                    yield base64.b64decode(data["audio"])  # type: ignore
            except asyncio.TimeoutError:
                pass

        # Send end of stream
        await websocket.send(EOS)

        # Receive remaining audio
        while True:
            try:
                response = await websocket.recv()
                data = json.loads(response)
                if data["audio"]:
                    yield base64.b64decode(data["audio"])  # type: ignore
            except websockets.exceptions.ConnectionClosed:
                break


DEFAULT_VOICE = Voice(
    voice_id="EXAVITQu4vr4xnSDxMaL",
    name="Bella",
    settings=VoiceSettings(
        stability=0.71, similarity_boost=0.5, style=0.0, use_speaker_boost=True
    ),
)


async def agenerate(
    text: Union[str, Iterator[str]],
    api_key: Optional[str] = None,
    voice: Union[str, Voice] = DEFAULT_VOICE,
    model: Union[str, Model] = "eleven_monolingual_v1",
    stream: bool = False,
    latency: int = 1,
    stream_chunk_size: int = 2048,
) -> Union[bytes, Iterator[bytes]]:
    TTS.generate_stream_input_async = generate_stream_input_async
    if isinstance(voice, str):
        voice_str = voice
        # If voice is valid voice_id, use it
        if is_voice_id(voice):
            voice = Voice(voice_id=voice)
        else:
            voice = next((v for v in voices() if v.name == voice_str), None)  # type: ignore # noqa E501

        # Raise error if voice not found
        if not voice:
            raise ValueError(f"Voice '{voice_str}' not found.")

    if isinstance(model, str):
        model = Model(model_id=model)

    assert isinstance(voice, Voice)
    assert isinstance(model, Model)

    if stream:
        if isinstance(text, str):
            for audio in TTS.generate_stream(
                text, voice, model, stream_chunk_size, api_key=api_key, latency=latency
            ):  # Change this line to use the async version
                yield audio
        elif isinstance(text, Iterator):
            async for audio in TTS.generate_stream_input_async(text, voice, model, api_key=api_key):
                yield audio
    else:
        assert isinstance(text, str)
        audio = TTS.generate(text, voice, model, api_key=api_key)  # Change this line to use the async version
        yield audio
