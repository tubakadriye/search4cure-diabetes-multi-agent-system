"""Check notebook setup paths from both supported starting directories offline."""

import ast
import json
from pathlib import Path
import subprocess
import sys
import unittest


class WalkthroughTests(unittest.TestCase):
    def test_comparison_imports_from_both_directories_without_loading_models(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "notebooks" / "walkthrough.ipynb").read_text())
        source = "".join(next(c["source"] for c in notebook["cells"] if c["id"] == "embedding-comparison"))
        assignment = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "COMPARISON_CHUNKS" for t in n.targets))
        imports = "\n".join(source.splitlines()[:assignment.lineno - 1])
        for directory in (root, root / "notebooks"):
            with self.subTest(starting_directory=directory.name):
                result = subprocess.run([sys.executable, "-c", imports + '\nassert "torch" not in sys.modules\n'],
                                        cwd=directory, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_setup_keeps_outputs_in_project_data_from_both_directories(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "notebooks" / "walkthrough.ipynb").read_text())
        source = next(cell["source"] for cell in notebook["cells"] if cell["id"] == "setup")
        setup = "".join(source) if isinstance(source, list) else source
        check = "\nassert OUTPUT_DIR.resolve() == (PROJECT_ROOT / 'data/arxiv').resolve(), 'Notebook output must stay in the project data directory'\n"
        for directory in (root, root / "notebooks"):
            with self.subTest(starting_directory=directory.name):
                result = subprocess.run([sys.executable, "-c", setup + check], cwd=directory,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
