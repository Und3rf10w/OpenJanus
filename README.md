# OpenJanus - A voice controlled IA for Star Citizen
OpenJanus is a voice controlled intelligent assistant for Star Citizen. It makes use of OpenAI's GPT for text generation, OpenAI whisper for speech to text conversion, and can make use of either OpenAI Whisper or elevenlabs for text to speech generation.

OpenJanus works like so:

- Streaming audio from the microphone directly into a stt engine like Whisper.
- Streaming generations from a LLM directly into a tts engine like elevenlabs or whisper
- By using elevenlabs, one can clone any voice they want to use with OpenJanus

# Features
- ATC
- Onboard Ship IA capable of executing commands on the ship (e.g. "turn on my ship's lights")
- Use whatever models you want (technically this is easy with Langchain, but prompting is on you)
- ???

# Roadmap
- [x] Easier configuration and installation
- [ ] Documentation
- [ ] Integration with sc-trade.tools
- [ ] Integration with cstone.space
- [ ] Tests (lol)

# How?
First, set your `OPENAI_API_KEY` and `ELEVEN_API_KEY` in your environment. Next, install the project and its dependences. Run `__main__.py`, then press and hold F12, and speak into your microphone.

First, copy `config.example.toml` to `config.toml`.

First, set the `listen_key` in `config.toml`. For a `fx` key, e.g. `f12`, you can set it to `"fx"` (e.g. `"f12"`). For other keys, you can set it to the key. You must wrap this in double quotation marks (`"`). Multi key input is not currently supported.

Set the `tts_engine` in `config.toml`. This must be either `whisper`, or `elevenlabs`.

Next, set your API key for whatever service(s) you're using in `config.toml`. This should be self-explanatory

> [!WARNING]
> If you have `x_API_KEY` set in your environment, we will default to that first, else, we will pull from the config. E.g. if `OPENAI_API_KEY` is set in your environment, OpenJanus will try that value, and ignore the one in `config.toml`. OpenJanus will forcibly set the environment variable.

# Why?
I see [videos like this](https://www.youtube.com/watch?v=hHy7OZQX_nQ) and they weren't released as open source at the time.

# What/where is the license?
At this time I am not providing a license, but am providing the codebase. Depending on how things go, I'll decide to provide a license.