# Logging Architecture Guide

## Overview

This project uses a **centralized, production-grade logging system** following SOLID and clean architecture principles.

**Key principles:**
- ✅ Single Responsibility — One module controls all logging configuration
- ✅ Framework Agnostic — Works with CLI, Streamlit, APIs, or any interface
- ✅ Configurable — Change log levels, formats, handlers globally
- ✅ Swappable — Easy to add file logging, cloud logging, or other backends

---

## Quick Start

### Use logging in any module:

```python
from src.vision_ml.logging import get_logger

# Get a logger instance (one per module)
logger = get_logger(__name__)

# Use it
logger.debug("Detailed diagnostic info")
logger.info("Important event occurred")
logger.warning("Something unexpected but recoverable")
logger.error("Operation failed")
logger.critical("System failure")
```

### Configure globally:

```python
from src.vision_ml.logging import LoggerConfig
import logging

# Change log level for entire application
LoggerConfig.set_level(logging.DEBUG)

# Enable file logging
LoggerConfig.enable_file_logging("logs/app.log")

# Disable file logging
LoggerConfig.disable_file_logging()
```

---

## Architecture

### File Structure

```
src/vision_ml/logging/
├── __init__.py          # Public exports (get_logger, LoggerConfig)
├── logger.py            # Core configuration
└── LOGGING_GUIDE.md     # This file
```

### How It Works

1. **Logger Creation**: Call `get_logger(__name__)` once per module
2. **Configuration**: All loggers share `LoggerConfig` settings
3. **Output**: Logs go to console (and optionally file)
4. **Handlers**: Add new handlers to `LoggerConfig` as needed

---

## Configuration Options

All settings in `src/vision_ml/logging/logger.py`:

```python
class LoggerConfig:
    # Global log level
    LEVEL = logging.INFO

    # Console output
    CONSOLE_ENABLED = True
    CONSOLE_LEVEL = logging.INFO

    # File output (optional)
    FILE_ENABLED = False
    FILE_PATH = "logs/vision_ml.log"
    FILE_LEVEL = logging.DEBUG

    # Format (applies to all handlers)
    FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
```

---

## Log Levels & When to Use

| Level | When to Use | Example |
|-------|---|---|
| **DEBUG** | Fine-grained diagnostic info | `Frame loaded, size 640x480` |
| **INFO** | Significant application events | `Model initialized`, `Inference complete` |
| **WARNING** | Something unexpected but recoverable | `API key not set, falling back to local` |
| **ERROR** | Operation failed | `Roboflow upload failed: 404 Not Found` |
| **CRITICAL** | System failure | `Out of memory, shutting down` |

---

## Examples

### Basic Usage

```python
from src.vision_ml.logging import get_logger

logger = get_logger(__name__)

def process_video(video_path):
    logger.info(f"Processing video: {video_path}")
    try:
        # ... processing code ...
        logger.info(f"Processed {1000} frames successfully")
    except Exception as e:
        logger.error(f"Video processing failed: {e}")
        raise
```

### With Configuration

```python
from src.vision_ml.logging import get_logger, LoggerConfig
import logging

# Enable debug logging globally
LoggerConfig.set_level(logging.DEBUG)
LoggerConfig.enable_file_logging("logs/debug.log")

logger = get_logger(__name__)
logger.debug("Debug logging is now active")
```

### In a Class

```python
from src.vision_ml.logging import get_logger

class ModelTrainer:
    def __init__(self, config):
        self.logger = get_logger(__name__)
        self.logger.info("Trainer initialized")

    def train(self):
        self.logger.info("Training started")
        try:
            # ... training logic ...
            self.logger.info("Training completed")
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
```

---

## Integration Points

### Logging is used in these modules:

- ✅ `src/vision_ml/labeling/auto_labeler.py` — Roboflow upload, frame loading
- ✅ `src/vision_ml/inference/pipeline.py` — Pipeline initialization, processing
- ⏳ Other modules can be integrated as needed

### To add logging to a new module:

1. **Import**:
   ```python
   from src.vision_ml.logging import get_logger
   logger = get_logger(__name__)
   ```

2. **Use**:
   ```python
   logger.info("Event occurred")
   ```

That's it! No other configuration needed.

---

## Enabling File Logging

### At startup (main.py or app entry point):

```python
from src.vision_ml.logging import LoggerConfig
import logging

# Enable file logging for production
LoggerConfig.set_level(logging.INFO)
LoggerConfig.enable_file_logging("logs/vision_ml.log")

# Now all loggers will write to both console and file
```

### File output example:

```
2026-03-11 14:22:01 | INFO     | vision_ml.inference.pipeline | Initializing InferencePipeline
2026-03-11 14:22:01 | INFO     | vision_ml.inference.pipeline | Using primary detector: yolo11n
2026-03-11 14:22:01 | INFO     | vision_ml.labeling.auto_labeler | Loaded 20 pseudo-labels
2026-03-11 14:22:02 | ERROR    | vision_ml.labeling.auto_labeler | Roboflow API key not set
```

---

## Best Practices

### ✅ DO:
- Use `get_logger(__name__)` once per module at module level
- Use appropriate log level (info for important events, debug for details)
- Include context in error messages: `logger.error(f"Failed to load {path}: {e}")`
- Enable file logging in production

### ❌ DON'T:
- Create multiple loggers in same module
- Use `print()` for debugging (use `logger.debug()` instead)
- Log sensitive data (passwords, API keys)
- Catch and ignore exceptions without logging

---

## Extending the Logger

### Add custom handler (email alerts, Slack, etc.):

Edit `src/vision_ml/logging/logger.py` and add to `get_logger()`:

```python
# Example: Add Slack handler
slack_handler = logging.handlers.HTTPHandler(
    'hooks.slack.com',
    '/services/YOUR/WEBHOOK/URL',
    method='POST'
)
slack_handler.setLevel(logging.ERROR)
logger.addHandler(slack_handler)
```

### Change format globally:

```python
LoggerConfig.FORMAT = "%(asctime)s | %(name)s | %(message)s"  # Simpler
```

---

## Troubleshooting

### Logs not showing?
```python
from src.vision_ml.logging import LoggerConfig
import logging

LoggerConfig.set_level(logging.DEBUG)
```

### File logging not working?
```python
from src.vision_ml.logging import LoggerConfig

LoggerConfig.enable_file_logging("logs/app.log")
# Check that logs/ directory is writable
```

### Want to use different format per handler?
Edit `src/vision_ml/logging/logger.py` and customize formatter per handler:

```python
console_formatter = logging.Formatter(CONSOLE_FORMAT)
file_formatter = logging.Formatter(FILE_FORMAT)
```

---

## References

- [Python logging module docs](https://docs.python.org/3/library/logging.html)
- [SOLID principles](https://en.wikipedia.org/wiki/SOLID)
- [Clean Code principles](https://refactoring.guru/refactoring)
