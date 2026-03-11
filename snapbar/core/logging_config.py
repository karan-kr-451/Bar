import os
import logging
from logging.handlers import RotatingFileHandler

def setup_logging():
    """Configures global logging to console and a rotating file in the /logs directory."""
    
    # Project root is one level up from snapbar/core
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log_dir = os.path.join(project_root, "logs")
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log_file = os.path.join(log_dir, "snapbar.log")
    
    # Formatter: Timestamp [Level] Logger: Message
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 1. File Handler (Rotating: 5MB per file, keeps 5)
    file_handler = RotatingFileHandler(
        log_file, maxBytes=5*1024*1024, backupCount=5, encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # 2. Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # Root Logger Config: Default to INFO to avoid third-party noise
    # (Especially groq/openai/httpx which log full Base64 payloads at DEBUG level)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # 3. Specific Log Levels
    # Keep our own code at DEBUG
    logging.getLogger("snapbar").setLevel(logging.DEBUG)
    
    # Explicitly silence chatty libraries that log full request payloads
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("groq").setLevel(logging.INFO)
    logging.getLogger("openai").setLevel(logging.INFO)
    logging.getLogger("anthropic").setLevel(logging.INFO)
    
    # Avoid duplicate handlers if setup is called multiple times
    if not root_logger.handlers:
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
    logging.info("Logging initialized. Saving logs to: %s", log_file)
    logging.info("Third-party DEBUG logs suppressed to prevent Base64 bloat.")

def get_logger(name: str):
    """Utility to get a logger with a specific name."""
    return logging.getLogger(name)
