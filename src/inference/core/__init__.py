"""C++ modules for the file inference pipeline."""


def load_native():
    from importlib import import_module

    try:
        return import_module("._native", __name__)
    except ModuleNotFoundError as exc:
        if exc.name != __name__ + "._native":
            raise
        raise ImportError(
            "Build the C++ extension from the project root: uv sync --group native && "
            "cmake -S src/inference/core -B build/core -DISMS_PYTHON=ON "
            "-DPython_EXECUTABLE=$PWD/.venv/bin/python && cmake --build build/core"
        ) from exc
