"""Generate reviewable mkdocstrings pages for every public top-level class."""

import argparse
import ast
import json
from pathlib import Path


def render(root: Path) -> dict[Path, str]:
    """Inspect source syntax without importing models or executing user code."""
    pages: dict[Path, str] = {}
    inventory: list[dict[str, str]] = []
    index = [
        "# API reference",
        "",
        "Every public top-level class is indexed from source. Mathematical context,",
        "assumptions, complexity and failure modes appear in the [family cards](../model-cards.md).",
        "",
        "The common execution boundary remains `Experiment.run(RunContext) -> RunResult`.",
        "",
    ]
    for source in sorted((root / "src/learning_atlas").rglob("*.py")):
        classes = [
            node.name
            for node in ast.parse(source.read_text()).body
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_")
        ]
        if not classes:
            continue
        module = ".".join(source.relative_to(root / "src").with_suffix("").parts)
        name = module.replace(".", "-") + ".md"
        content = [
            f"# {module}",
            "",
            "[Mathematical context and family contracts](../model-cards.md)",
            "",
            "[Architecture and execution invariants](../architecture.md)",
            "",
        ]
        for cls in classes:
            inventory.append({"module": module, "class": cls, "page": name})
            content.extend(
                [
                    f"::: {module}.{cls}",
                    "    options:",
                    "      show_root_heading: true",
                    "      show_source: false",
                    "",
                ]
            )
        pages[root / "docs/api" / name] = "\n".join(content)
        index.append(f"- [{module}]({name}): " + ", ".join(f"`{cls}`" for cls in classes))
    pages[root / "docs/api/index.md"] = "\n".join(index) + "\n"
    pages[root / "docs/api/inventory.json"] = json.dumps(inventory, indent=2) + "\n"
    return pages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    for path, content in render(root).items():
        if args.check:
            if not path.is_file() or path.read_text() != content:
                parser.error(f"API reference drift: {path.name}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)


if __name__ == "__main__":
    main()
