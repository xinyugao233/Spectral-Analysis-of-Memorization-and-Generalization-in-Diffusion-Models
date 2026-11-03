"""Check local Markdown document/image targets without freezing prose style."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit


def check(root: Path) -> list[tuple[str, str]]:
    broken = []
    checked = 0
    for document in root.rglob("*.md"):
        if any(
            part in {".git", ".venv", "outputs", "__pycache__"}
            for part in document.relative_to(root).parts
        ):
            continue
        source = re.sub(
            r"```.*?```", "", document.read_text(errors="replace"), flags=re.S
        )
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", source):
            target = target.split(' "')[0].strip("<>")
            parsed = urlsplit(target)
            if parsed.scheme or target.startswith(("#", "/")) or not parsed.path:
                continue
            checked += 1
            if not (document.parent / unquote(parsed.path)).exists():
                broken.append((str(document.relative_to(root)), target))
    print(f"{root}: checked {checked} local targets; {len(broken)} missing")
    for document, target in broken:
        print(f"{document}: {target}")
    return broken


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root", nargs="?", type=Path, default=Path(__file__).resolve().parents[1]
    )
    return int(bool(check(parser.parse_args().root.resolve())))


if __name__ == "__main__":
    raise SystemExit(main())
