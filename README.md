# OpenJanus - A voice controlled IA for Star Citizen
I'll finish this when I get bored.

OpenJanus is a voice controlled intelligent assistant for Star Citizen. It makes use of OpenAI's GPT for text generation, OpenAI whisper for speech to text conversion, and can make use of either piper or elevenlabs for text to speech generation.

OpenJanus works like so:

- Streaming audio from the microphone directly into a stt engine like Whisper.
- Streaming generations from a LLM directly into a tts engine like piper or elevenlabs
- By using either elevenlabs or piper, one can clone any voice they want to use with OpenJanus

# Features
- ATC
- Onboard Ship IA capable of executing commands on the ship (e.g. "turn on my ship's lights)
- Integration with sc-trade.tools
- ???

# Why?
I get pissed when I see [videos like this](https://www.youtube.com/watch?v=hHy7OZQX_nQ) and they aren't released as open source; so I went full Bender and decided to make my own, only there's no blackjack or hookers.

# What/where is the license?
At this time I am not providing a license, but am providing the codebase. Depending on how things go, I'll decide to provide a license.