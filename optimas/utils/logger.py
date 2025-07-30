import logging
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from queue import Queue
import os


def setup_logger(
    name: str = "Logger",
    log_level: int = logging.INFO,
    log_file: str = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up and return a thread-safe logger using a QueueHandler and QueueListener.
    If `log_file` is provided, log output will be written to a rotating file as well as stdout.

    Args:
        name (str): Name of the logger.
        log_level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
        log_file (str, optional): File path to write logs to. Enables file logging if set.
        max_bytes (int): Max file size in bytes before rotation (default: 10MB).
        backup_count (int): Number of rotated backup files to keep.

    Returns:
        logging.Logger: Configured logger instance.
    """
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%m-%d %H:%M:%S"
    )

    # Set up root logger once if using file logging
    if log_file and not logging.getLogger().handlers:
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)

        queue = Queue()
        queue_handler = QueueHandler(queue)
        root_logger.addHandler(queue_handler)

        handlers = []

        # Console handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        handlers.append(stream_handler)

        # File handler (only on local rank 0 for multi-GPU setups)
        if os.environ.get("LOCAL_RANK", "0") == "0":
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count
            )
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)

        listener = QueueListener(queue, *handlers)
        listener.start()
        root_logger.listener = listener  # Store for later cleanup

    # Set up named logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if not logger.handlers and not log_file:
        # Standalone stream-based logger setup (for no log_file use)
        queue = Queue()
        queue_handler = QueueHandler(queue)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        listener = QueueListener(queue, stream_handler)
        listener.start()

        logger.addHandler(queue_handler)
        logger.listener = listener  # Attach for optional later cleanup
        logger.propagate = False
    return logger
