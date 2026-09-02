"""Test-suite-wide import guarantees.

torch must be imported before faiss on macOS (OpenMP runtime conflict
causes a hard segfault otherwise). Importing it here makes every test
module safe regardless of collection order.
"""

import torch  # noqa: F401
