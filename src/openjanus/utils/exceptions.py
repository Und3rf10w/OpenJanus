class EnvNotSetException(Exception):
    def __init__(self, env_variable: str):
        message = f"The environment variable {env_variable} must be set"
        super().__init__(message)


class ConfigFileNotFound(Exception):
    def __init__(self, config_file_path: str):
        message = f"The config file at {config_file_path} was not found"
        super().__init__(message)


class ConfigKeyNotFound(Exception):
    def __init__(self, config_key: str):
        message = f"The config key {config_key} was not found"
        super().__init__(message)


class ApiKeyNotSetException(Exception):
    def __init__(self, service: str):
        message = f"The {service} API key was not found in the environment variable or the config file"
        super().__init__(message)


class TtsNotSetException(Exception):
    def __init__(self):
        message = f"The TTS engine was not found in the environment variable or the config file"
        super().__init__(message)


class TtsNotImplementedException(Exception):
    def __init__(self, tts_engine: str):
        message = f"The TTS engine {tts_engine} is not implemented"
        super().__init__(message)


class DirectoryCreationException(Exception):
    def __init__(self, directory: str):
        message = f"Failed to create the {directory} directory"
        super().__init__(message)
