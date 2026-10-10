"""Assemble the notebooks from their plain-text sources and run them.

Every notebook's real source is a script in ``tools/nb_src``. Cells are
introduced by a banner::

    # ==[ markdown ]==
    # ==[ code ]==

Markdown lines are commented out in the script and un-commented here. Keeping
the source as text means a reviewer reads Python instead of notebook JSON,
while the committed ``.ipynb`` still carries its outputs for the marker.

    python tools/assemble_notebooks.py
    python tools/assemble_notebooks.py project1_population
    python tools/assemble_notebooks.py --check     # parse only, never execute
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import nbformat
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "tools" / "nb_src"
NOTEBOOK_DIR = ROOT / "notebooks"

BANNER = re.compile(r"^#\s*==\[\s*(markdown|code)\s*\]==\s*$")
TIMEOUT = 900


class NotebookBuilder:
    """A small state machine that turns banner-delimited text into cells."""

    def __init__(self) -> None:
        self.cells: list[nbformat.NotebookNode] = []
        self._kind: str | None = None
        self._lines: list[str] = []

    def feed(self, line: str) -> None:
        """Consume one line of the source script."""
        banner = BANNER.match(line)
        if banner:
            self.flush()
            self._kind = banner.group(1)
        elif self._kind is not None:
            self._lines.append(line)

    def flush(self) -> None:
        """Close the cell currently being collected, if it has any content."""
        if self._kind is None:
            return
        body = self._lines
        if self._kind == "markdown":
            body = [ln[2:] if ln.startswith("# ") else ln.lstrip("#") for ln in body]
        while body and not body[0].strip():
            body.pop(0)
        while body and not body[-1].strip():
            body.pop()
        if body:
            make = (nbformat.v4.new_markdown_cell if self._kind == "markdown"
                    else nbformat.v4.new_code_cell)
            self.cells.append(make("\n".join(body)))
        self._kind, self._lines = None, []

    def build(self) -> nbformat.NotebookNode:
        """Finish and return the notebook node."""
        self.flush()
        if not self.cells:
            raise ValueError("no '# ==[ code ]==' or '# ==[ markdown ]==' banners found")
        notebook = nbformat.v4.new_notebook()
        notebook.metadata.update({
            "kernelspec": {"name": "python3", "display_name": "Python 3",
                           "language": "python"},
            "language_info": {"name": "python", "version": sys.version.split()[0]},
        })
        notebook.cells = self.cells
        return notebook


def read(script: Path) -> nbformat.NotebookNode:
    """Parse one script into a notebook node."""
    builder = NotebookBuilder()
    for line in script.read_text(encoding="utf-8").splitlines():
        builder.feed(line)
    return builder.build()


def write(script: Path, execute: bool = True) -> Path:
    """Assemble a notebook, optionally execute it, and save it."""
    notebook = read(script)
    if execute:
        try:
            ExecutePreprocessor(timeout=TIMEOUT, kernel_name="python3").preprocess(
                notebook, {"metadata": {"path": str(NOTEBOOK_DIR)}})
        except CellExecutionError as failure:
            raise SystemExit(f"{script.name}: a cell raised; nothing written\n{failure}")
    NOTEBOOK_DIR.mkdir(exist_ok=True)
    destination = NOTEBOOK_DIR / f"{script.stem}.ipynb"
    nbformat.write(notebook, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble the project notebooks.")
    parser.add_argument("names", nargs="*", help="script stems to build; default all")
    parser.add_argument("--check", action="store_true",
                        help="parse the sources without executing them")
    args = parser.parse_args()

    scripts = ([SOURCE_DIR / f"{n}.py" for n in args.names] if args.names
               else sorted(SOURCE_DIR.glob("*.py")))
    for script in scripts:
        if not script.exists():
            raise SystemExit(f"no such script: {script}")
    for script in scripts:
        built = write(script, execute=not args.check)
        print(f"built {built.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
