"""Test configuration.

``--doctest-modules`` collects every module under ``src``, which means pytest
*imports* every module under ``src``. ``rcc.viz`` imports matplotlib, and
matplotlib is an optional extra — so without this the suite would fail at
collection for anyone who installed the package without ``[viz]``, reporting an
ImportError rather than a test result. It is tested by importing rather than by
asking whether the module is installed: a matplotlib that is present but broken —
a mismatched build, a missing system library — fails the same way and should be
skipped the same way.

``examples`` goes on the path because ``tests/test_learning.py`` trains against a
real environment, and ``examples/_common.py`` already knows how to build one and
hand it over as tensors. One definition serves both, and breaking it for the
examples breaks it for the tests too.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "examples"))

try:
    import matplotlib  # noqa: F401
except ImportError:
    collect_ignore = ["src/rcc/viz.py"]
else:
    collect_ignore = []
