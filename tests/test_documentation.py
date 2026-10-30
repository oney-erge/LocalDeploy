"""Small checks for public documentation hygiene."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
IGNORED_DIRS = {".git", ".setup", ".venv", "build", "dist", "htmlcov", "node_modules"}


def _markdown_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*.md")
        if not any(part in IGNORED_DIRS for part in path.relative_to(ROOT).parts)
    )


def _tracked_text_files() -> list[Path]:
    names = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    binary_suffixes = {".gif", ".icns", ".ico", ".png"}
    return [
        ROOT / name
        for name in names
        if Path(name).suffix.lower() not in binary_suffixes and (ROOT / name).is_file()
    ]


def test_tracked_text_does_not_use_em_dashes() -> None:
    bad = [path.relative_to(ROOT).as_posix() for path in _tracked_text_files() if "\N{EM DASH}" in path.read_text(encoding="utf-8")]
    assert not bad, f"replace em dashes in: {bad}"


def test_local_markdown_links_exist_with_exact_case() -> None:
    repo_paths = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if not any(part in IGNORED_DIRS for part in path.relative_to(ROOT).parts)
    }
    missing: list[str] = []

    for document in _markdown_files():
        text = document.read_text(encoding="utf-8")
        targets = [match.group(1).strip() for match in re.finditer(r"!?\[[^\]]*\]\(([^)]+)\)", text)]
        targets += [
            match.group(1).strip()
            for match in re.finditer(r"<(?:a|img)\b[^>]+(?:href|src)=[\"']([^\"']+)[\"']", text, re.IGNORECASE)
        ]

        for raw_target in targets:
            target = raw_target.split()[0].strip("<>")
            if target.startswith(("#", "data:", "http://", "https://", "mailto:")):
                continue
            relative_target = unquote(target.split("#", 1)[0]).replace("\\", "/")
            if not relative_target:
                continue
            resolved = (document.parent / relative_target).resolve()
            try:
                repo_relative = resolved.relative_to(ROOT).as_posix()
            except ValueError:
                missing.append(f"{document.relative_to(ROOT)}: {target} leaves the repository")
                continue
            if repo_relative not in repo_paths:
                missing.append(f"{document.relative_to(ROOT)}: {target}")

    assert not missing, "missing or case-mismatched documentation links:\n" + "\n".join(missing)
