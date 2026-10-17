import logging

# Keep test runs out of the real log file; configure_logging() skips configured loggers.
logging.getLogger("transcriber").addHandler(logging.NullHandler())
