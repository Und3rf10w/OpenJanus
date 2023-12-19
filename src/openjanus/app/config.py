from os import getenv, environ, path, makedirs
import logging
import toml
from typing import Dict, Any

from openjanus.utils.exceptions import (
    ApiKeyNotSetException, 
    ConfigFileNotFound,
    ConfigKeyNotFound,
    DirectoryCreationException,
    TtsMpvNotFoundException,
    TtsNotImplementedException
)


LOGGER = logging.getLogger(__name__)


def load_config() -> Dict[str, Any]:
    """Load the config file"""
    try:
        if getenv("OPENJANUS_CONFIG_PATH"):
            LOGGER.debug(f"Loading config from {getenv('OPENJANUS_CONFIG_PATH')}")
            config = toml.load(getenv("OPENJANUS_CONFIG_PATH"))  # type: ignore
        LOGGER.debug("Loading config from config.toml")
        config = toml.load("config.toml")
        return config
    except FileNotFoundError:
        if getenv("OPENJANUS_CONFIG_PATH"):
            LOGGER.error(f"Config file not found at {getenv('OPENJANUS_CONFIG_PATH')}")
            raise ConfigFileNotFound(getenv("OPENJANUS_CONFIG_PATH")) # type: ignore
        else:
            LOGGER.error("Config file not found at config.toml")
            raise ConfigFileNotFound("config.toml")
        

def set_openai_api_key() -> str:
    """Set the openai API Key, first by checking the environment variable, then by checking the config file"""
    try:
        if getenv("OPENAI_API_KEY"):
            LOGGER.debug("Setting openai API key from environment variable")
            return getenv("OPENAI_API_KEY")  # type: ignore
        else:
            LOGGER.debug("Setting openai API key from config file")
            config = load_config()
            environ["OPENAI_API_KEY"] = config["openai"]["api_key"]
            return config["openai"]["api_key"]
    except KeyError:
        LOGGER.error("The openai API key was not found in the environment variable or the config file")
        raise ApiKeyNotSetException("OpenAI")
    

def set_eleven_api_key() -> str:
    """Set the eleven API Key, first by checking the environment variable, then by checking the config file"""
    try:
        if getenv("ELEVEN_API_KEY"):
            LOGGER.debug("Setting elevenlabs API key from environment variable")
            return getenv("ELEVEN_API_KEY")  # type: ignore
        else:
            LOGGER.debug("Setting elevenlabs API key from config file")
            config = load_config()
            environ["ELEVEN_API_KEY"] = config["elevenlabs"]["api_key"]
            return config["elevenlabs"]["api_key"]
    except KeyError:
        LOGGER.error("The elevenlabs API key was not found in the environment variable or the config file")
        raise ApiKeyNotSetException("Elevenlabs")
    

def check_mpv_path() -> str:
    """Check the mpv path, first by checking the environment variable, then by checking the config file"""
    import shutil
    lib = shutil.which("mpv")
    if lib is None:
        try:
            LOGGER.debug("Setting mpv path from config file")
            config = load_config()
            if not path.isfile(config["openai"]["whisper"]["mpv_path"]):
                LOGGER.error("mpv.exe was not found")
                raise TtsMpvNotFoundException()
            else:
                return config["openai"]["whisper"]["mpv_path"]
        except KeyError:
            LOGGER.error("The mpv path was not found in the environment variable or the config file")
            raise ConfigKeyNotFound("openai/whisper/mpv_path")
        except FileNotFoundError:
            LOGGER.error("mpv.exe was not found")
            raise TtsMpvNotFoundException()
    else:
        return lib
    

def get_tts_engine() -> str:
    """Get the TTS engine, first by checking the environment variable, then by checking the config file"""
    try:
        if getenv("TTS_ENGINE"):
            LOGGER.debug("Setting TTS engine from environment variable")
            if getenv("TTS_ENGINE") not in ["elevenlabs", "whisper"]:
                LOGGER.error("The TTS engine is not valid")
                raise TtsNotImplementedException(getenv("TTS_ENGINE", "NOT_SET"))
            return getenv("TTS_ENGINE")  # type: ignore
        else:
            LOGGER.debug("Setting TTS engine from config file")
            config = load_config()
            if config["openjanus"]["tts_engine"] not in ["elevenlabs", "whisper"]:
                LOGGER.error("The TTS engine is not valid")
                raise TtsNotImplementedException(config["openjanus"]["tts_engine"])
            if config["openjanus"]["tts_engine"] == "whisper":
                _ = set_openai_api_key()  # This is actually a safe way to check if the API key is set for us to use
                check_mpv_path()
            return config["openjanus"]["tts_engine"]
    except KeyError:
        LOGGER.error("The TTS engine was not found in the environment variable or the config file")
        raise ConfigKeyNotFound("openjanus/tts_engine")
    

def get_recordings_dir() -> str:
    """Get the recordings directory by checking the config file"""
    try:
        LOGGER.debug("Getting recordings directory from config file")
        config = load_config()
        recordings_dir = config["openjanus"]["recordings_directory"]
        return path.relpath(recordings_dir)
    except KeyError:
        LOGGER.error("The recordings directory was not found in the environment variable or the config file")
        raise ConfigKeyNotFound("openjanus/recordings_directory")


def ensure_recordings_dir_exists():
    """Ensure that the recordings directory exists"""
    recordings_dir = get_recordings_dir()
    if not path.exists(recordings_dir):
        try:
            makedirs(recordings_dir)
        except Exception as e:
            LOGGER.error(f"Failed to create the {recordings_dir} directory", exc_info=e)
            raise DirectoryCreationException(f"Failed to create the {recordings_dir} directory") from e
        
def get_elevenlabs_config() -> Dict[str, Any]:
    """Get the elevenlabs config"""
    try:
        LOGGER.debug("Getting elevenlabs config from config file")
        config = load_config()
        set_eleven_api_key()
        if not config["elevenlabs"]["elevenlabs_voice_id"]:
            LOGGER.warning("The elevenlabs voice was not set, using the default voice")
            from openjanus.tts.elevenlabs.async_patch import DEFAULT_VOICE
            config["elevenlabs"]["elevenlabs_voice_id"] = DEFAULT_VOICE
        if not config["elevenlabs"]['elevenlabs_stability']:
            LOGGER.warning("The elevenlabs stability was not set, using the default stability")
            config["elevenlabs"]["elevenlabs_stability"] = 0.5
        if not config["elevenlabs"]['elevenlabs_similarity_boost']:
            LOGGER.warning("The elevenlabs similarity boost was not set, using the default similarity boost")
            config["elevenlabs"]["elevenlabs_similarity_boost"] = 0.75
        if not config["elevenlabs"]['elevenlabs_style']:
            LOGGER.warning("The elevenlabs style was not set, using the default style")
            config["elevenlabs"]["elevenlabs_style"] = 0
        if not config["elevenlabs"]['elevenlabs_use_speaker_boost'] or config["elevenlabs"]['use_speaker_boost'].lower() != "true":
            config["elevenlabs"]["elevenlabs_use_speaker_boost"] = False
        elif config["elevenlabs"]['elevenlabs_use_speaker_boost'].lower() == "true":
            config["elevenlabs"]["elevenlabs_use_speaker_boost"] = True
        else:
            LOGGER.warning("The elevenlabs use speaker boost was misconfigured, using the default use speaker boost")
            config["elevenlabs"]["use_speaker_boost"] = False
        return config["elevenlabs"]
            
    except KeyError:
        LOGGER.error("The elevenlabs config was not found in the environment variable or the config file")
        raise ConfigKeyNotFound("elevenlabs")
    
def get_openai_whisper_config() -> Dict[str, Any]:
    """Get the openai whisper config"""
    try:
        LOGGER.debug("Getting openai whisper config from config file")
        config = load_config()
        if not config["openai"]["whisper"]["whisper_voice_id"]:
            LOGGER.warning("The openai whisper voice id was not set, using the default voice id")
            config["openai"]["whisper"]["whisper_voice_id"] = "nova"
        if not config["openai"]["whisper"]["whisper_voice_model"]:
            LOGGER.warning("The openai whisper voice model was not set, using the default voice model")
            config["openai"]["whisper"]["whisper_voice_model"] = "tts-1"
        if not config["openai"]["whisper"]["whisper_engine"]:
            LOGGER.warning("The openai whisper engine was not set, using the default engine")
            config["openai"]["whisper"]["whisper_engine"] = "whisper-1"
        return config["openai"]["whisper"]
    except KeyError:
        LOGGER.error("The openai whisper config was not found in the environment variable or the config file")
        raise ConfigKeyNotFound("openai/whisper")


def startup_checks() -> bool:
    """Perform startup checks
    
    :returns: True if all checks pass"""
    _ = get_tts_engine()
    _ = ensure_recordings_dir_exists()
    return True  # Otherwise it'll error anyways