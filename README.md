# OpenJanus - A voice controlled IA for Star Citizen
OpenJanus is a voice controlled intelligent assistant for Star Citizen. It makes use of OpenAI's GPT for text generation, OpenAI whisper for speech to text conversion, and can make use of either OpenAI Whisper or elevenlabs for text to speech generation.

OpenJanus works like so:

- Streaming audio from the microphone directly into a stt engine like Whisper.
- Streaming generations from a LLM directly into a tts engine like elevenlabs or whisper
- By using elevenlabs, one can clone any voice they want to use with OpenJanus

# Features
- ATC
- Onboard Ship IA capable of executing commands on the ship (e.g. "turn on my ship's lights)
- Integration with sc-trade.tools
- Integration with cstone.space
- ???

# Why?
I see [videos like this](https://www.youtube.com/watch?v=hHy7OZQX_nQ) and they weren't released as open source at the time.

# What/where is the license?
At this time I am not providing a license, but am providing the codebase. Depending on how things go, I'll decide to provide a license.