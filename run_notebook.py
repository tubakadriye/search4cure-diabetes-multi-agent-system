"""Execute the walkthrough with live output and save only this run's cell outputs."""

import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


class LiveNotebookClient(NotebookClient):
    def process_message(self, msg, cell, cell_index):
        result = super().process_message(msg, cell, cell_index)
        if msg["msg_type"] == "stream":
            print(msg["content"]["text"], end="", flush=True)
        return result


def main():
    root = Path(__file__).resolve().parent
    runtime = root / "data" / "notebook_runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("IPYTHONDIR", str(runtime / "ipython"))
    os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(runtime / "jupyter"))
    notebook_path = root / "notebooks" / "walkthrough.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None

    def report_cell(cell, cell_index, **kwargs):
        if cell.cell_type == "code":
            print(f"Running cell {cell_index + 1}: {cell.id}", flush=True)

    client = LiveNotebookClient(notebook, timeout=180, kernel_name="search4cure",
                               resources={"metadata": {"path": str(root)}},
                               on_cell_start=report_cell)
    try:
        client.execute()
    finally:
        # Preserve a failed run's diagnostics without retaining stale successful outputs.
        nbformat.write(notebook, notebook_path)
    print("Notebook completed successfully; outputs saved in notebooks/walkthrough.ipynb", flush=True)


if __name__ == "__main__":
    main()
