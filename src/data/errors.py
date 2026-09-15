"""Exceptions raised by the EVRPTW-GR data layer."""


class EVRPTWGRError(Exception):
    """Base class for every error raised by this package."""


class DatasetNotFoundError(EVRPTWGRError):
    """The raw dataset root does not exist or holds no instance files."""


class MalformedInstanceError(EVRPTWGRError):
    """An instance file does not follow the EVRPTW-GR layout.

    Raised for structural damage: a bad header, a node row with the wrong
    number of columns, a non-numeric field, or an unknown node type.
    """

    def __init__(self, path, message, line_number=None):
        self.path = path
        self.line_number = line_number
        location = f"{path}" if line_number is None else f"{path}:{line_number}"
        super().__init__(f"{location}: {message}")


class MissingFieldError(EVRPTWGRError):
    """A required field is absent from an instance file.

    Nothing is substituted for the missing value; callers must decide what to
    do. Vehicle curb weight is exempt because it is genuinely absent from the
    Small_Network subset rather than missing by accident.
    """

    def __init__(self, path, field, message=None):
        self.path = path
        self.field = field
        detail = message or f"required field {field!r} is missing"
        super().__init__(f"{path}: {detail}")
