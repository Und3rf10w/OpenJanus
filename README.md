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
- Intergration with Cornerstone's [Item Finder](https://finder.cstone.space/) and [Planetary Survey](https://survey.cstone.space)
- ???

# Roadmap
- [x] Easier configuration and installation
- [ ] Documentation
- [ ] Integration with sc-trade.tools
- [x] Integration with cstone.space
    - [x] [Item Finder](https://finder.cstone.space/)
    - [x] [Planetary Survey](https://survey.cstone.space) 
- [ ] Tests (lol)

# How?
First, set your `OPENAI_API_KEY` and `ELEVEN_API_KEY` in your environment. Next, install the project and its dependences. Run `__main__.py`, then press and hold F12, and speak into your microphone.

First, copy `config.example.toml` to `config.toml`.

First, set the `listen_key` in `config.toml`. For a `fx` key, e.g. `f10`, you can set it to `"fx"` (e.g. `"f10"`). For other keys, you can set it to the key. You must wrap this in double quotation marks (`"`). Multi key input is not currently supported.

Set the `tts_engine` in `config.toml`. This must be either `whisper`, or `elevenlabs`.

Next, set your API key for whatever service(s) you're using in `config.toml`. This should be self-explanatory

> [!WARNING]
> If you have `x_API_KEY` set in your environment, we will default to that first, else, we will pull from the config. E.g. if `OPENAI_API_KEY` is set in your environment, OpenJanus will try that value, and ignore the one in `config.toml`. OpenJanus will forcibly set the environment variable.

# Why?
I prefer agentic architectures when it comes to AI tooling, and having a solution for this backed by [LangChain](https://www.langchain.com/) seemed like a reasonable approach

# What/where is the license?
At this time I am not providing a license, but am providing the codebase. Depending on how things go, I'll decide to provide a license.

# I want to play Star Citizen, but don't have an account
Haven't tried Star Citizen yet, but want to check it out and giving OpenJanus a try?
I would sincerely appreciate if you [used my referral code to create your account](https://robertsspaceindustries.com/enlist?referral=STAR-PVSB-Z7GR)!