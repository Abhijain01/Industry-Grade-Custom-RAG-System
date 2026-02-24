import sys
from pathlib import Path
from loguru import logger
from .config import CONFIG

def setup_logger():
    logger.remove()
    
    # Configure console
    logger.add(
        sys.stdout, 
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=CONFIG.system.log_level if CONFIG else "INFO"
    )
    
    # Configure file (JSON structured logging)
    if CONFIG:
        log_path = Path(CONFIG.system.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            str(log_path),
            serialize=True, # Structured JSON logging
            rotation="10 MB",
            level=CONFIG.system.log_level
        )

setup_logger()
# Export the customized logger
get_logger = lambda name: logger.bind(module=name)
