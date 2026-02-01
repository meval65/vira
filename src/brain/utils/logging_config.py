import logging
import sys

def setup_logging(level=logging.INFO, format_style: str = "standard"):
    if format_style == "minimal":
        fmt = "%(levelname)s: %(message)s"
    elif format_style == "detailed":
        fmt = "%(asctime)s [%(name)s:%(lineno)d] %(levelname)s: %(message)s"
    else:
        fmt = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    
    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("motor").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    
    return logging.getLogger("vira")

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


