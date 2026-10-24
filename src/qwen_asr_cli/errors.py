class CliError(Exception):
    exit_code = 1


class InputError(CliError):
    exit_code = 2


class AudioError(CliError):
    exit_code = 3


class ModelError(CliError):
    exit_code = 4


class TranscriptionError(CliError):
    exit_code = 5
