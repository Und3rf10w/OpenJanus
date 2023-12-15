import asyncio
from datetime import datetime
import logging
import pathlib
import pyaudio
import pydub
import threading
import wave

from langchain.agents import AgentExecutor
from langchain.document_loaders.blob_loaders import Blob
from langchain.schema import Document

# This is to support oai package >=1.0.0
from openjanus.stt.whisper.parser import OpenAIWhisperParser


LOGGER = logging.getLogger(__name__)


class Recorder:
    def __init__(self):
        self.chunk = 2048
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.record_path = "recordings"
        self.recording_extension = "wav"
        self.output_naming_format = f"recording.{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}".replace(' ','_')
        self.is_recording = False
        self.frames = []
        self.record_event = threading.Event()
        self.finished_recording_path = ""

    def callback(self,in_data, frame_count, time_info, status):
        if self.is_recording == True:
            self.frames.append(in_data)
            return (in_data, pyaudio.paContinue)

        elif self.is_recording == False:
            self.frames.append(in_data)
            return (in_data, pyaudio.paComplete)

        else:
            return (in_data,pyaudio.paContinue)

    def start_recording(self):
        LOGGER.info("Started recording audio...")
        self.frames = []
        self.audio = pyaudio.PyAudio()

        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
            stream_callback=self.callback
            # input_device_index=13  # TODO: Get from config
        )
        self.is_recording = True
        LOGGER.debug(f"self.is_recording: {self.is_recording}")
        self.stream.start_stream()
        self.record_event.set()

    # def convert_to_mp3(self):
    #     sound = pydub.AudioSegment.from_wav(f"{self.record_path}{self.output_naming_format}.{self.recording_extension}")
    #     self.mp3_output_filepath = f"{self.record_path}{self.output_naming_format}.mp3"
    #     sound.export(self.mp3_output_filepath, format="mp3")
    #     LOGGER.debug(f"Converted {self.record_path}{self.output_naming_format}.{self.recording_extension} to {self.record_path}{self.output_naming_format}.mp3")
    
    def stop_recording(self) -> str:
        self.record_event.clear()
        self.is_recording = False
        LOGGER.info("Stopped Recording audio...")
        if self.recording_extension == "wav":
            if self.stream.is_active():
                self.stream.stop_stream()
                self.stream.close()
                self.audio.terminate()
                self.finished_recording_path = str(pathlib.PurePath(f"{self.record_path}/{self.output_naming_format}.{self.recording_extension}"))
                wav_file = wave.open(f=self.finished_recording_path, mode='wb')
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(self.audio.get_sample_size(self.format))
                wav_file.setframerate(self.rate)
                wav_file.writeframes(b''.join(self.frames))
                wav_file.close()
                LOGGER.debug(f"Recording saved to {self.finished_recording_path}")
                return self.finished_recording_path
            else:
                return ""
        else:
            return ""
    def transcribe_and_invoke(self, agent_chain: AgentExecutor, recording_path: str):
        LOGGER.info("hit transcribe_and_invoke")
        try:
            # Construct a Blob from the recording file
            blob = Blob.from_path(recording_path)
            whisper_parser = OpenAIWhisperParser()
            # Generate document objects
            documents = whisper_parser.lazy_parse(blob)
            
            combined_transcription = []
            # Combine the documents
            for document in documents:
                LOGGER.debug(f"Transcription: {document.page_content}")
                combined_transcription.append(document.page_content)
            
            # These are just for testing
            # output = asyncio.run(agent_chain.ainvoke({"input": "Seraphim Station, this is john smith, requesting permission to land, over.", "chat_history": []}))
            # output = asyncio.run(agent_chain.ainvoke({"input": "Turn the ship's lights on", "chat_history": []}))
            
            output = asyncio.run(agent_chain.ainvoke({"input": ''.join(combined_transcription), "chat_history": []}))
            return output['output'][0]['response']
        except Exception as e:
            LOGGER.error(f"Failed to transcribe: {str(e)}")
