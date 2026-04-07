from functools import partial

import polars as pl

from ._schemas import DuckCols, ParamLists, PyCols
from ._str_builder import EMPTY_STR, format_kwords

_INDENT = "\n            "
_INDENT2 = "\n        "
_LINE_END = ".\n\n        "


def to_func(
    has_params: pl.Expr, py: PyCols, p_lists: ParamLists, dk: DuckCols
) -> pl.Expr:
    return format_kwords(
        _txt(),
        func_name=py.name.alias("func_name"),
        args=_args(has_params, p_lists),
        varargs=_varargs(dk, py),
        self_type=py.self_type,
        description=_description(dk, py),
        see_also_section=_see_also(py),
        args_section=_args_section(has_params, p_lists, dk, py),
        examples_section=_examples_section(dk),
        sql_name=py.sql_name,
        expr_call=_expr_call(has_params, py, p_lists, dk),
        ignore_nulls=True,
    )


def _args(has_params: pl.Expr, p_lists: ParamLists) -> pl.Expr:
    return pl.when(has_params.gt(1)).then(
        pl.format(", {}", p_lists.signatures.list.slice(1).list.join(", "))
    )


def _varargs(dk: DuckCols, py: PyCols) -> pl.Expr:
    return pl.when(dk.varargs.is_not_null()).then(
        pl.format(", *args: {}", py.varargs_type)
    )


def _dk_args(has_params: pl.Expr, p_lists: ParamLists) -> pl.Expr:
    return pl.when(has_params.gt(1)).then(
        format_kwords(", {args}", args=p_lists.names.list.slice(1).list.join(", "))
    )


def _dk_varargs(dk: DuckCols) -> pl.Expr:
    return pl.when(dk.varargs.is_not_null()).then(pl.lit(", *args"))


def _expr_call(
    has_params: pl.Expr, py: PyCols, p_lists: ParamLists, dk: DuckCols
) -> pl.Expr:
    formatter = partial(
        format_kwords,
        dk_args=_dk_args(has_params, p_lists),
        dk_varargs=_dk_varargs(dk),
        ignore_nulls=True,
    )
    return (
        pl
        .when(py.glot_name.is_not_null())
        .then(
            formatter(
                'func("{glot_name}", self.inner(){dk_args}{dk_varargs})',
                glot_name=py.glot_name,
            )
        )
        .otherwise(
            formatter(
                'anon("{sql_name}", self.inner(){dk_args}{dk_varargs})',
                sql_name=py.sql_name,
            )
        )
    )


def _args_section(
    has_params: pl.Expr, p_lists: ParamLists, dk: DuckCols, py: PyCols
) -> pl.Expr:
    return pl.when(has_params.gt(1).or_(dk.varargs.is_not_null())).then(
        format_kwords(
            "\n\n        Args:\n{posargs}{varargs}",
            posargs=p_lists.docs.list.slice(1).list.join("\n"),
            varargs=pl.when(dk.varargs.is_not_null()).then(
                format_kwords(
                    "\n            *args ({pytypes}): `{dk_types}` expression",
                    pytypes=py.varargs_type,
                    dk_types=dk.varargs,
                )
            ),
            ignore_nulls=True,
        )
    )


def _see_also(py: PyCols) -> pl.Expr:
    return pl.when(py.aliases.list.len().gt(0)).then(
        format_kwords(
            "\n\n        See Also:\n            {aliases}",
            aliases=py.aliases.list.sort().list.join(", "),
            ignore_nulls=True,
        )
    )


def _description(dk: DuckCols, py: PyCols) -> pl.Expr:
    return (
        pl
        .when(dk.description.is_not_null())
        .then(
            dk.description.str
            .strip_chars()
            .str.replace_all("\u2019", "'")
            .str.replace_all('"', EMPTY_STR)
            .str.replace_all(r"\n[ \t]*", _INDENT2)
            .str.replace_all(r"\. ", _LINE_END)
            .str.replace_all(".\n        ", _LINE_END)
            .str.strip_chars_end(".")
        )
        .otherwise(pl.format("SQL {} function", py.sql_name))
    )


def _examples_section(dk: DuckCols) -> pl.Expr:
    non_empty = dk.examples.list.eval(
        pl.element().filter(pl.element().str.strip_chars().ne(EMPTY_STR))
    )
    txt = """

        Examples:
            ```sql
            {examples}
            ```"""
    return pl.when(non_empty.list.len().gt(0)).then(
        format_kwords(
            txt,
            examples=non_empty.list.eval(
                pl.element().str.replace_all(r"\n[ \t]*", _INDENT)
            ).list.join(_INDENT),
            ignore_nulls=True,
        )
    )


def _txt() -> str:
    return '''
    def {func_name}(self{args}{varargs}) -> {self_type}:
        """{description}.

        **SQL name**: *{sql_name}*{see_also_section}{args_section}{examples_section}

        Returns:
            {self_type}
        """
        return self._cls({expr_call})
        '''
