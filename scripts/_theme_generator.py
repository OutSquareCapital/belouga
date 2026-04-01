from pathlib import Path

import pyochain as pc


def generate_themes(caller: Path, dest: Path) -> int:
    from pygments.styles._mapping import (  # pyright: ignore[reportMissingTypeStubs]  # noqa: PLC2701
        STYLES,
    )

    styles = (
        pc.Iter(STYLES.values()).map_star(lambda _, style, __: f'"{style}"').join(" ,")
    )
    file_content = dest.read_text(encoding="utf-8")
    start_marker = "# theme marker START"
    end_marker = "# theme marker END"
    lit = f"Themes = Literal[{styles}]"

    start_idx = file_content.find(start_marker)
    end_idx = file_content.find(end_marker)
    content = f"{file_content[: start_idx + len(start_marker)]}\n{lit}\n'''{_doc(caller)}'''\n{file_content[end_idx:]}"

    return dest.write_text(content, encoding="utf-8")


def _doc(caller: Path) -> str:
    return f"""Themes available for SQL syntax highlighting in the `sql_query` method.

Dynamically generated from the available styles in the `pygments` library by `{caller.as_posix()}`.

Do NOT edit manually."""
