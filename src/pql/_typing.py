from typing import TYPE_CHECKING, Literal

import pyochain as pc

if TYPE_CHECKING:
    from ._joins import JoinKeys
JoinStrategy = Literal["inner", "left", "right", "outer", "semi", "anti"]
AsofJoinStrategy = Literal["backward", "forward"]
UniqueKeepStrategy = Literal["any", "none", "first", "last"]
type TransferEncoding = Literal["hex", "base64"]
RankMethod = Literal["average", "min", "max", "dense", "ordinal"]
PivotAgg = Literal[
    "min", "max", "first", "last", "sum", "mean", "median", "len", "count"
]
type JoinKeysRes[T: pc.Seq[str] | str] = pc.Result[JoinKeys[T], ValueError]
type GroupByClause = Literal["ROLLUP", "CUBE"]

# theme marker START
Themes = Literal[
    "abap",
    "algol",
    "algol_nu",
    "arduino",
    "autumn",
    "bw",
    "borland",
    "coffee",
    "colorful",
    "default",
    "dracula",
    "emacs",
    "friendly_grayscale",
    "friendly",
    "fruity",
    "github-dark",
    "gruvbox-dark",
    "gruvbox-light",
    "igor",
    "inkpot",
    "lightbulb",
    "lilypond",
    "lovelace",
    "manni",
    "material",
    "monokai",
    "murphy",
    "native",
    "nord-darker",
    "nord",
    "one-dark",
    "paraiso-dark",
    "paraiso-light",
    "pastie",
    "perldoc",
    "rainbow_dash",
    "rrt",
    "sas",
    "solarized-dark",
    "solarized-light",
    "staroffice",
    "stata-dark",
    "stata-light",
    "tango",
    "trac",
    "vim",
    "vs",
    "xcode",
    "zenburn",
]
"""Themes available for SQL syntax highlighting in the `sql_query` method.

Dynamically generated from the available styles in the `pygments` library by `scripts/__main__.py`.

Do NOT edit manually."""
# theme marker END
