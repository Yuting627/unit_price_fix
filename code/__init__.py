"""Local package. Re-export stdlib ``code`` so pytest/pdb still work."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_stdlib_code():
    candidates = [
        Path(sys.base_prefix) / "Lib" / "code.py",
        Path(sys.base_prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "code.py",
    ]
    for path in candidates:
        if path.exists():
            spec = importlib.util.spec_from_file_location("_py_stdlib_code", path)
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(mod)
            return mod
    raise ImportError("cannot locate stdlib code.py")


_stdlib = _load_stdlib_code()
InteractiveConsole = _stdlib.InteractiveConsole
InteractiveInterpreter = _stdlib.InteractiveInterpreter
compile_command = _stdlib.compile_command
interact = _stdlib.interact
