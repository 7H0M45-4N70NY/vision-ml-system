# --- Vision ML System ---
"""
Vision ML System: Production-grade Retail Analytics.

Import Conventions:
------------------
1. Internal Modules (src/vision_ml/**/*):
   ALWAYS use relative imports to ensure portability.
   Example: `from ..logging import get_logger`

2. Streamlit Pages (pages/*.py):
   ALWAYS use absolute imports with 'src.' prefix.
   Example: `from src.vision_ml.inference.pipeline import InferencePipeline`

3. CLI Scripts (scripts/*.py):
   ALWAYS use sys.path.insert(0, ...) to add 'src' and use absolute imports
   WITHOUT 'src.' prefix.
   Example:
     sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
     from vision_ml.inference.pipeline import InferencePipeline
"""

__version__ = "1.0.0"
