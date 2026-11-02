"""Build the root-level project report with an installed pdfLaTeX."""
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile


def build(root, directory):
    directory.mkdir(parents=True, exist_ok=True)
    command = ["pdflatex", "-no-shell-escape", "-interaction=nonstopmode",
               "-halt-on-error", f"-output-directory={directory}", "docs/report.tex"]
    for number in (1, 2):
        result = subprocess.run(command, cwd=root, capture_output=True, text=True)
        output = result.stdout + result.stderr
        (directory / f"build-{number}.log").write_text(output)
        if result.returncode:
            raise SystemExit(output[-12000:])
    destination = root / "report.pdf"
    shutil.copy2(directory / "report.pdf", destination)
    print(f"Wrote {destination} ({destination.stat().st_size:,} bytes)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, help="Keep build logs in this directory")
    args = parser.parse_args()
    if not shutil.which("pdflatex"):
        parser.error("pdflatex is required to build the report")
    root = Path(__file__).resolve().parents[1]
    if args.build_dir:
        build(root, args.build_dir.resolve())
    else:
        with tempfile.TemporaryDirectory(prefix="failure-recovery-report-") as path:
            build(root, Path(path))


if __name__ == "__main__":
    main()
