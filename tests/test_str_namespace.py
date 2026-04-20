import polars as pl
import pyochain as pc
import pytest
from polars.testing import assert_frame_equal

import pql

type TestArgs = tuple[pql.Expr | str, pl.Expr | str]

pql_text = pql.col("text")
pql_text_short = pql.col("text_short")
pql_dt_str = pql.col("dt_str")
pl_text = pl.col("text")
pl_text_short = pl.col("text_short")
pl_dt_str = pl.col("dt_str")

_LF = pql.LazyFrame({
    "text": [
        "  Hello World suffix  ",
        "  foo bar baz suffix  ",
        "  Polars is great suffix  ",
        "  Testing string functions suffix  ",
    ],
    "text_nullable": [
        "  abc  ",
        "abc",
        "",
        "  ",
    ],
    "text_short": [
        "a",
        "ab",
        "",
        "abc",
    ],
    "date_str": [
        "2024-01-15",
        "2024-02-20",
        "2024-03-25",
        "2024-04-30",
    ],
    "dt_str": [
        "2024-01-15 10:30:00",
        "2024-02-20 15:45:30",
        "2024-03-25 20:00:00",
        "2024-04-30 23:59:59",
    ],
    "dt_mixed": [
        "2024-01-15",
        "2024-02-20 15:45:30",
        "2024-03-25",
        "2024-04-30 23:59:59",
    ],
    "time_str": [
        "10:30:00",
        "15:45:30",
        "20:00:00",
        "23:59:59",
    ],
    "normalize_input": [
        "ardèch",
        "Café",
        "résumé",
        "naive",
    ],
    "text_with_null": [
        "aa",
        None,
        "bb",
        "cc",
    ],
    "prefixed": [
        "prefix_text",
        "prefix_other",
        "prefix_sample",
        "prefix_data",
    ],
    "suffixed": [
        "text_suffix",
        "other_suffix",
        "sample_suffix",
        "data_suffix",
    ],
    "prefix_exact": [
        "foobar",
        "foofoobar",
        "baab",
        "barfoo",
    ],
    "suffix_exact": [
        "foobar",
        "foobarbar",
        "barfoo",
        "ababa",
    ],
    "prefix_col": [
        "prefix_",
        "prefix_",
        "pre",
        "data",
    ],
    "suffix_col": [
        "_suffix",
        "_suffix",
        "suffix",
        "data",
    ],
    "suffix_val": pc.Iter(range(4)).map(lambda _: "suffix").collect(list),
    "json": ['{"a": 1}', '{"a": 2}', '{"a": 3}', '{"a": 4}'],
    "json_path": ["$.a", "$.a", "$.a", "$.a"],
    "numbers": ["123.456", "456.789", "789.123", "1234.567"],
    "signed_numbers": ["-1", "+7", "-12345", None],
})


def sample_df() -> pql.LazyFrame:
    return _LF


def assert_eq(pql_expr: pql.Expr, polars_expr: pl.Expr) -> None:
    assert_frame_equal(
        sample_df().select(pql_expr).lazy(),
        sample_df().lazy().select(polars_expr),
        check_dtypes=False,
        check_row_order=False,
    )


def test_to_uppercase() -> None:
    assert_eq(pql_text.str.to_uppercase(), pl_text.str.to_uppercase())


def test_to_lowercase() -> None:
    assert_eq(pql_text.str.to_lowercase(), pl_text.str.to_lowercase())


def test_len_chars() -> None:
    assert_eq(pql_text.str.len_chars(), pl_text.str.len_chars())


def test_contains_literal() -> None:
    assert_eq(
        pql_text.str.contains("lo", literal=True),
        pl_text.str.contains("lo", literal=True),
    )


def test_contains_regex() -> None:
    assert_eq(
        pql_text.str.contains(r"\d+", literal=False),
        pl_text.str.contains(r"\d+", literal=False),
    )


def test_starts_with() -> None:
    assert_eq(pql_text.str.starts_with("Hello"), pl_text.str.starts_with("Hello"))


def test_ends_with() -> None:
    assert_eq(pql_text.str.ends_with("suffix"), pl_text.str.ends_with("suffix"))


def test_replace() -> None:
    pql_replace = pql_text.str.replace
    pl_replace = pl_text.str.replace
    hi = "Hi"
    assert_eq(pql_replace("Hello", hi), pl_replace("Hello", hi))
    sep = "_"
    assert_eq(pql_replace("a", sep, n=2), pl_replace("a", sep, n=2))
    assert_eq(pql_replace("a", sep, n=0), pl_replace("a", sep, n=0))
    assert_eq(pql_replace("a", sep, n=-1), pl_replace("a", sep, n=-1))


_SPACE = " "


@pytest.mark.parametrize("characters", [" ", None])
def test_strip_chars(characters: str | None) -> None:
    assert_eq(pql_text.str.strip_chars(characters), pl_text.str.strip_chars(characters))


@pytest.mark.parametrize("characters", [" ", None])
def test_strip_chars_start(characters: str | None) -> None:
    assert_eq(
        pql_text.str.strip_chars_start(characters),
        pl_text.str.strip_chars_start(characters),
    )


def test_strip_chars_end() -> None:
    assert_eq(pql_text.str.strip_chars_end(), pl_text.str.strip_chars_end())
    assert_eq(pql_text.str.strip_chars_end(_SPACE), pl_text.str.strip_chars_end(_SPACE))


@pytest.mark.parametrize("offset", [0, 2, 5])
@pytest.mark.parametrize("length", [None, 1, 3, 5])
def test_slice(offset: int, length: int) -> None:
    assert_eq(
        pql_text_short.str.slice(offset=offset, length=length),
        pl_text_short.str.slice(offset=offset, length=length),
    )


def test_len_bytes() -> None:
    assert_eq(pql_text.str.len_bytes(), pl_text.str.len_bytes())


@pytest.mark.parametrize("n", [1, 2, 3])
def test_head_tail(n: int) -> None:
    assert_eq(pql_text.str.head(n), pl_text.str.head(n))
    assert_eq(pql_text.str.tail(n), pl_text.str.tail(n))


def test_reverse_str() -> None:
    assert_eq(pql_text.str.reverse(), pl_text.str.reverse())


def test_to_titlecase() -> None:
    assert_eq(pql_text.str.to_titlecase(), pl_text.str.to_titlecase())


def test_split() -> None:
    sep = ","
    assert_eq(pql_text.str.split(sep), pl_text.str.split(sep))


def test_extract_all() -> None:
    ptrn = r"\d+"
    assert_eq(pql_text.str.extract_all(ptrn), pl_text.str.extract_all(ptrn))


def test_extract() -> None:
    ptrn = r"(\w+)"
    ptrn_2 = r"(\w+)\s+(\w+)"
    pql_extract = pql_text.str.extract
    pl_extract = pl_text.str.extract
    assert_eq(pql_extract(ptrn), pl_extract(ptrn))
    assert_eq(pql_extract(ptrn_2, group_index=2), pl_extract(ptrn_2, group_index=2))
    assert_eq(pql_extract(ptrn, group_index=0), pl_extract(ptrn, group_index=0))


def test_find() -> None:
    pattern = r"[A-Z][a-z]+"
    world = "World"
    missing = "missing"
    pql_find = pql_text.str.find
    pl_find = pl_text.str.find
    assert_eq(pql_find(world, literal=True), pl_find(world, literal=True))
    assert_eq(pql_find(missing, literal=True), pl_find(missing, literal=True))
    assert_eq(pql_find(pattern, literal=False), pl_find(pattern, literal=False))


def test_escape_regex() -> None:
    assert_eq(pql_text.str.escape_regex(), pl_text.str.escape_regex())


@pytest.mark.parametrize(
    "json_path", [("$.a", "$.a"), (pql.col("json_path"), pl.col("json_path"))]
)
def test_json_path_match(json_path: TestArgs) -> None:
    assert_eq(
        pql.col("json").str.json_path_match(json_path[0]),
        pl.col("json").str.json_path_match(json_path[1]),
    )


@pytest.mark.parametrize("delimiter", ["|", "-", ","])
@pytest.mark.parametrize("ignore_nulls", [True, False])
def test_join(delimiter: str, ignore_nulls: bool) -> None:
    assert_eq(
        pql_text_short.str.join(delimiter, ignore_nulls=ignore_nulls),
        pl_text_short.str.join(delimiter, ignore_nulls=ignore_nulls),
    )


def test_to_date() -> None:
    fmt = "%Y-%m-%d"
    assert_eq(
        pql.col("date_str").str.to_date(format=fmt),
        pl.col("date_str").str.to_date(format=fmt),
    )


def test_to_datetime() -> None:
    fmt = "%Y-%m-%d %H:%M:%S"
    assert_eq(
        pql_dt_str.str.to_datetime(format=fmt), pl_dt_str.str.to_datetime(format=fmt)
    )


def test_to_time() -> None:
    fmt = "%H:%M:%S"
    assert_eq(
        pql.col("time_str").str.to_time(format=fmt),
        pl.col("time_str").str.to_time(format=fmt),
    )


def test_strptime() -> None:
    fmt = "%Y-%m-%d %H:%M:%S"
    assert_eq(pql_dt_str.str.strptime(fmt), pl_dt_str.str.strptime(pl.Datetime, fmt))


def test_normalize() -> None:
    """Duckdb currently only supports NFC normalization."""
    assert_eq(
        pql.col("normalize_input").str.normalize(),
        pl.col("normalize_input").str.normalize("NFC"),
    )


@pytest.mark.parametrize("scale", [0, 2, 3])
def test_to_decimal(scale: int) -> None:
    assert_eq(
        pql.col("numbers").str.to_decimal(scale=scale),
        pl.col("numbers").str.to_decimal(scale=scale),
    )


@pytest.mark.parametrize(
    "prefixes",
    [
        ("prefix_", "prefix_"),
        (pql.col("prefix_col"), pl.col("prefix_col")),
        ("foo", "foo"),
    ],
)
def test_strip_prefix(prefixes: TestArgs) -> None:
    assert_eq(
        pql.col("prefixed").str.strip_prefix(prefixes[0]),
        pl.col("prefixed").str.strip_prefix(prefixes[1]),
    )


@pytest.mark.parametrize(
    "suffixes",
    [
        ("_suffix", "_suffix"),
        (pql.col("suffix_col"), pl.col("suffix_col")),
        ("bar", "bar"),
    ],
)
def test_strip_suffix(suffixes: TestArgs) -> None:
    assert_eq(
        pql.col("suffixed").str.strip_suffix(suffixes[0]),
        pl.col("suffixed").str.strip_suffix(suffixes[1]),
    )


def test_replace_all() -> None:
    pql_replace_all = pql_text.str.replace_all
    pl_replace_all = pl_text.str.replace_all
    assert_eq(
        pql_replace_all("o", "0", literal=True), pl_replace_all("o", "0", literal=True)
    )

    assert_eq(
        pql_replace_all("l", "L", literal=True), pl_replace_all("l", "L", literal=True)
    )

    assert_eq(
        pql_replace_all(r"\d+", "X", literal=False),
        pl_replace_all(r"\d+", "X", literal=False),
    )
    assert_eq(
        pql_replace_all("suffix", pql.col("suffix_val"), literal=True),
        pl_replace_all("suffix", pl.col("suffix_val"), literal=True),
    )


@pytest.mark.parametrize("pattern", ["a", r"\d+"])
@pytest.mark.parametrize("literal", [True, False])
def test_count_matches(pattern: str, literal: bool) -> None:
    assert_eq(
        (pql_text.str.count_matches(pattern, literal=literal)),
        (pl_text.str.count_matches(pattern, literal=literal)),
    )


@pytest.mark.parametrize("length", [5, 10])
@pytest.mark.parametrize("fill_char", ["*", "-", " "])
def test_pad_start(length: int, fill_char: str) -> None:
    assert_eq(
        pql_text_short.str.pad_start(length, fill_char=fill_char),
        pl_text_short.str.pad_start(length, fill_char=fill_char),
    )


@pytest.mark.parametrize("length", [5, 10])
@pytest.mark.parametrize("fill_char", ["*", "-", " "])
def test_pad_end(length: int, fill_char: str) -> None:
    assert_eq(
        pql_text_short.str.pad_end(length, fill_char=fill_char),
        pl_text_short.str.pad_end(length, fill_char=fill_char),
    )


@pytest.mark.parametrize("length", [4, 5, 10])
def test_zfill(length: int) -> None:
    assert_eq(pql.col("numbers").str.zfill(length), pl.col("numbers").str.zfill(length))
    assert_eq(
        pql.col("signed_numbers").str.zfill(length),
        pl.col("signed_numbers").str.zfill(length),
    )


@pytest.mark.parametrize("encoding", ["base64", "hex"])
def test_encode(encoding: pql.sql.typing.TransferEncoding) -> None:
    assert_eq(pql_text.str.encode(encoding), pl_text.str.encode(encoding))
