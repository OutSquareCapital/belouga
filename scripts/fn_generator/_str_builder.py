from string import Formatter

import polars as pl
import pyochain as pc

EMPTY_STR = pl.lit("")


def format_kwords(
    txt: str, *, ignore_nulls: bool = False, **kwargs: pl.Expr
) -> pl.Expr:
    kword_map = pc.Dict.from_ref(kwargs)
    if ignore_nulls:
        kword_map = _ignore_nulls(kword_map)

    return (
        pc
        .Iter(Formatter().parse(txt))
        .map_star(lambda lit, field, _fmt, _conv: (lit, pc.Option(field)))
        .collect()
        .into(
            lambda parts: pl.format(
                parts
                .iter()
                .map_star(
                    lambda lit, field: field.map(lambda _: f"{lit}{{}}").unwrap_or(lit)
                )
                .join(""),
                *parts
                .iter()
                .filter_map_star(lambda _lit, field: field)
                .filter_map(kword_map.get_item),
            )
        )
    )


def _ignore_nulls(kwargs: pc.Dict[str, pl.Expr]) -> pc.Dict[str, pl.Expr]:
    return (
        kwargs
        .items()
        .iter()
        .map_star(lambda k, v: (k, v.fill_null(EMPTY_STR)))
        .collect(pc.Dict)
    )
