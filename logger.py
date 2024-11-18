# Logging configuration

# Imports
import sys
import logging

# Logger format
FMT = "[%(levelname)s][%(asctime)s] %(message)s"
DATEFMT = "%d-%b-%y %H:%M:%S"

# Custom formatter for the logger
class Formatter(logging.Formatter):

    LEVEL_REMAP = {
        'DEBUG': 'DEBUG',
        'INFO': ' INFO',
        'WARNING': ' WARN',
        'ERROR': 'ERROR',
        'CRITICAL': 'FATAL',
    }

    def format(self, record):
        record.levelname = self.LEVEL_REMAP.get(record.levelname, record.levelname)
        return super().format(record)

# Custom color formatter for the logger
class ColorFormatter(logging.Formatter):

    LEVEL_REMAP = {
        'DEBUG': '\x1b[38;21mDEBUG\x1b[0m',
        'INFO': '\x1b[38;5;39m INFO\x1b[0m',
        'WARNING': '\x1b[38;5;226m WARN\x1b[0m',
        'ERROR': '\x1b[38;5;196mERROR\x1b[0m',
        'CRITICAL': '\x1b[31;1mFATAL\x1b[0m',
    }

    def format(self, record):
        record.levelname = self.LEVEL_REMAP.get(record.levelname, record.levelname)
        return super().format(record)

# Configure the logger
stream_handler = logging.StreamHandler(stream=sys.stdout)
formatter = ColorFormatter(fmt=FMT, datefmt=DATEFMT)
stream_handler.setFormatter(formatter)
logging.basicConfig(level=logging.INFO, handlers=(stream_handler,))

# Get the logger
log = logging.getLogger(__name__)
# EOF
