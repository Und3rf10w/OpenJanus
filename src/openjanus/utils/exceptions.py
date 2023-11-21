class EnvNotSetException(Exception):
    def __init__(self, env_variable):
        message = f"The environment variable {env_variable} must be set"
        super().__init__(message)