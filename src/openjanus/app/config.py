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
            if not path.isfile(config["openjanus"]["mpv_path"]):
                LOGGER.error("mpv.exe was not found")
                raise TtsMpvNotFoundException()
            else:
                return config["openjanus"]["mpv_path"]
        except KeyError:
            LOGGER.error("The mpv path was not found in the environment variable or the config file")
            raise ConfigKeyNotFound("openjanus/mpv_path")
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


def startup_checks() -> bool:
    """Perform startup checks
    
    :returns: True if all checks pass"""
    _ = get_tts_engine()
    _ = ensure_recordings_dir_exists()
    return True  # Otherwise it'll error anyways