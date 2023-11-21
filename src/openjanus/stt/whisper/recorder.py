import asyncio
from datetime import datetime
import logging
import pyaudio
import pydub
import threading
import wave

from langchain.agents import AgentExecutor
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders import Blob
from langchain.schema import Document


LOGGER = logging.getLogger(__name__)


class Recorder:
    def __init__(self):
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.record_path = "recordings/"
        self.recording_extension = "wav"
        self.output_naming_format = f"recording.{datetime.now()}"
        self.is_recording = False
        self.frames = []

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
            input_device_index=13  # TODO: Get from config
        )
        self.is_recording = True
        while self.is_recording:
            data = self.stream.read(self.chunk)
            self.frames.append(data)

    # def convert_to_mp3(self):
    #     sound = pydub.AudioSegment.from_wav(f"{self.record_path}{self.output_naming_format}.{self.recording_extension}")
    #     self.mp3_output_filepath = f"{self.record_path}{self.output_naming_format}.mp3"
    #     sound.export(self.mp3_output_filepath, format="mp3")
    #     LOGGER.debug(f"Converted {self.record_path}{self.output_naming_format}.{self.recording_extension} to {self.record_path}{self.output_naming_format}.mp3")
    
    def stop_recording(self):
        self.is_recording = False
        LOGGER.info("Stopped Recording audio...")
        if self.recording_extension == "wav":
            if self.stream.is_active():
                self.stream.stop_stream()
                self.stream.close()
                self.audio.terminate()

                wav_file = wave.open(f"{self.record_path}{self.output_naming_format}.{self.recording_extension}", 'wb')
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(self.audio.get_sample_size(self.format))
                wav_file.setframerate(self.rate)
                wav_file.writeframes(b''.join(self.frames))
                wav_file.close()
                LOGGER.debug(f"Recording saved to {self.record_path}{self.output_naming_format}.{self.recording_extension}")

                # self.convert_to_mp3()

    def transcribe_and_invoke(self, agent_chain: AgentExecutor):
        try:
            # Construct a Blob from the recording file
            blob = Blob(path=f"{self.record_path}{self.output_naming_format}.{self.recording_extension}")
            whisper_parser = OpenAIWhisperParser()
            # Generate document objects
            documents = whisper_parser.lazy_parse(blob)
            
            combined_transcription = []
            # Combine the documents
            for document in documents:
                LOGGER.debug(f"Transcription: {document.page_content}")
                combined_transcription.append(document.page_content)
            asyncio.run(agent_chain.ainvoke({"input": ''.join(combined_transcription), "chat_history": []}))
        except Exception as e:
            LOGGER.error(f"Failed to transcribe: {str(e)}")
