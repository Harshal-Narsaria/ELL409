from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def package_project(output):
    root = Path(__file__).resolve().parents[1]
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    include = ["ell409_tree", "configs", "tests", "notebooks", "scripts", "docs"]
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for name in include:
            for path in sorted((root / name).rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                    archive.write(path, "ell409-neural-tree/" + path.relative_to(root).as_posix())
        for name in ["README.md", "pyproject.toml", ".gitignore", "requirements-lock.txt", "implementation-status.json"]:
            if (root / name).exists():
                archive.write(root / name, "ell409-neural-tree/" + name)
    return output
