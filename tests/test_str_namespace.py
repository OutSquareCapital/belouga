import polars as pl
import pytest
from polars.testing import assert_frame_equal
from pyochain import Iter

import belugas as bl
from belugas.typing import TransferEncoding

type TestArgs = tuple[bl.Expr | str, pl.Expr | str]

bl_text = bl.col("text")
bl_text_short = bl.col("text_short")
bl_dt_str = bl.col("dt_str")
pl_text = pl.col("text")
pl_text_short = pl.col("text_short")
pl_dt_str = pl.col("dt_str")

_LF = pl.DataFrame({
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
    "suffix_val": Iter(range(4)).map(lambda _: "suffix").collect(list),
    "json": ['{"a": 1}', '{"a": 2}', '{"a": 3}', '{"a": 4}'],
    "json_path": ["$.a", "$.a", "$.a", "$.a"],
    "numbers": ["123.456", "456.789", "789.123", "1234.567"],
    "signed_numbers": ["-1", "+7", "-12345", None],
}).pipe(bl.from_arrow)


def test_to_uppercase() -> None:
    assert_eq(bl_text.str.to_uppercase(), pl_text.str.to_uppercase())


def test_to_lowercase() -> None:
    assert_eq(bl_text.str.to_lowercase(), pl_text.str.to_lowercase())


def test_len_chars() -> None:
    assert_eq(bl_text.str.len_chars(), pl_text.str.len_chars())


def test_contains_literal() -> None:
    assert_eq(
        bl_text.str.contains("lo"),
        pl_text.str.contains("lo", literal=True),
    )


def test_contains_regex() -> None:
    assert_eq(
        bl_text.re.matches(r"\d+"),
        pl_text.str.contains(r"\d+", literal=False),
    )


def test_starts_with() -> None:
    assert_eq(bl_text.str.starts_with("Hello"), pl_text.str.starts_with("Hello"))


def test_ends_with() -> None:
    assert_eq(bl_text.str.ends_with("suffix"), pl_text.str.ends_with("suffix"))


def test_replace() -> None:
    bl_replace = bl_text.str.replace
    pl_replace = pl_text.str.replace_all
    hi = "Hi"
    assert_eq(bl_replace("Hello", hi), pl_replace("Hello", hi))
    sep = "_"
    assert_eq(bl_replace("a", sep), pl_replace("a", sep))


_SPACE = " "


@pytest.mark.parametrize("characters", [" ", None])
def test_strip_chars(characters: str | None) -> None:
    assert_eq(bl_text.str.strip_chars(characters), pl_text.str.strip_chars(characters))


@pytest.mark.parametrize("characters", [" ", None])
def test_strip_chars_start(characters: str | None) -> None:
    assert_eq(
        bl_text.str.strip_chars_start(characters),
        pl_text.str.strip_chars_start(characters),
    )


def test_strip_chars_end() -> None:
    assert_eq(bl_text.str.strip_chars_end(), pl_text.str.strip_chars_end())
    assert_eq(bl_text.str.strip_chars_end(_SPACE), pl_text.str.strip_chars_end(_SPACE))


@pytest.mark.parametrize("offset", [0, 2, 5])
@pytest.mark.parametrize("length", [None, 1, 3, 5])
def test_slice(offset: int, length: int) -> None:
    assert_eq(
        bl_text_short.str.slice(offset=offset, length=length),
        pl_text_short.str.slice(offset=offset, length=length),
    )


def test_len_bytes() -> None:
    assert_eq(bl_text.str.len_bytes(), pl_text.str.len_bytes())


@pytest.mark.parametrize("n", [1, 2, 3])
def test_head_tail(n: int) -> None:
    assert_eq(bl_text.str.head(n), pl_text.str.head(n))
    assert_eq(bl_text.str.tail(n), pl_text.str.tail(n))


def test_reverse_str() -> None:
    assert_eq(bl_text.str.reverse(), pl_text.str.reverse())


def test_to_titlecase() -> None:
    assert_eq(bl_text.str.to_titlecase(), pl_text.str.to_titlecase())


def test_split() -> None:
    sep = ","
    assert_eq(bl_text.str.split(sep), pl_text.str.split(sep))


def test_extract_all() -> None:
    ptrn = r"\d+"
    assert_eq(bl_text.str.extract_all(ptrn), pl_text.str.extract_all(ptrn))


def test_extract() -> None:
    ptrn = r"(\w+)"
    ptrn_2 = r"(\w+)\s+(\w+)"
    bl_extract = bl_text.str.extract
    pl_extract = pl_text.str.extract
    assert_eq(bl_extract(ptrn), pl_extract(ptrn))
    assert_eq(bl_extract(ptrn_2, group_index=2), pl_extract(ptrn_2, group_index=2))
    assert_eq(bl_extract(ptrn, group_index=0), pl_extract(ptrn, group_index=0))


def test_find() -> None:
    pattern = r"[A-Z][a-z]+"
    world = "World"
    missing = "missing"
    bl_find = bl_text.str.find
    pl_find = pl_text.str.find
    assert_eq(bl_find(world, literal=True), pl_find(world, literal=True))
    assert_eq(bl_find(missing, literal=True), pl_find(missing, literal=True))
    assert_eq(bl_find(pattern, literal=False), pl_find(pattern, literal=False))


def test_escape_regex() -> None:
    assert_eq(bl_text.str.escape_regex(), pl_text.str.escape_regex())


@pytest.mark.parametrize(
    "json_path", [("$.a", "$.a"), (bl.col("json_path"), pl.col("json_path"))]
)
def test_json_path_match(json_path: TestArgs) -> None:
    assert_eq(
        bl.col("json").str.json_path_match(json_path[0]),
        pl.col("json").str.json_path_match(json_path[1]),
    )


@pytest.mark.parametrize("delimiter", ["|", "-", ","])
@pytest.mark.parametrize("ignore_nulls", [True, False])
def test_join(delimiter: str, ignore_nulls: bool) -> None:
    assert_eq(
        bl_text_short.str.join(delimiter, ignore_nulls=ignore_nulls),
        pl_text_short.str.join(delimiter, ignore_nulls=ignore_nulls),
    )


def test_to_date() -> None:
    fmt = "%Y-%m-%d"
    assert_eq(
        bl.col("date_str").str.to_date(format=fmt),
        pl.col("date_str").str.to_date(format=fmt),
    )


def test_to_datetime() -> None:
    fmt = "%Y-%m-%d %H:%M:%S"
    assert_eq(
        bl_dt_str.str.to_datetime(format=fmt), pl_dt_str.str.to_datetime(format=fmt)
    )


def test_to_time() -> None:
    fmt = "%H:%M:%S"
    assert_eq(
        bl.col("time_str").str.to_time(format=fmt),
        pl.col("time_str").str.to_time(format=fmt),
    )


def test_strptime() -> None:
    fmt = "%Y-%m-%d %H:%M:%S"
    assert_eq(bl_dt_str.str.strptime(fmt), pl_dt_str.str.strptime(pl.Datetime, fmt))


def test_normalize() -> None:
    """Duckdb currently only supports NFC normalization."""
    assert_eq(
        bl.col("normalize_input").str.normalize(),
        pl.col("normalize_input").str.normalize("NFC"),
    )


@pytest.mark.parametrize("scale", [0, 2, 3])
def test_to_decimal(scale: int) -> None:
    assert_eq(
        bl.col("numbers").str.to_decimal(scale=scale),
        pl.col("numbers").str.to_decimal(scale=scale),
    )


@pytest.mark.parametrize(
    "prefixes",
    [
        ("prefix_", "prefix_"),
        (bl.col("prefix_col"), pl.col("prefix_col")),
        ("foo", "foo"),
    ],
)
def test_strip_prefix(prefixes: TestArgs) -> None:
    assert_eq(
        bl.col("prefixed").str.strip_prefix(prefixes[0]),
        pl.col("prefixed").str.strip_prefix(prefixes[1]),
    )


@pytest.mark.parametrize(
    "suffixes",
    [
        ("_suffix", "_suffix"),
        (bl.col("suffix_col"), pl.col("suffix_col")),
        ("bar", "bar"),
    ],
)
def test_strip_suffix(suffixes: TestArgs) -> None:
    assert_eq(
        bl.col("suffixed").str.strip_suffix(suffixes[0]),
        pl.col("suffixed").str.strip_suffix(suffixes[1]),
    )


def test_replace_all() -> None:
    bl_replace_all = bl_text.str.replace_all
    pl_replace_all = pl_text.str.replace_all
    assert_eq(
        bl_replace_all("o", "0", literal=True), pl_replace_all("o", "0", literal=True)
    )

    assert_eq(
        bl_replace_all("l", "L", literal=True), pl_replace_all("l", "L", literal=True)
    )

    assert_eq(
        bl_replace_all(r"\d+", "X", literal=False),
        pl_replace_all(r"\d+", "X", literal=False),
    )
    assert_eq(
        bl_replace_all("suffix", bl.col("suffix_val"), literal=True),
        pl_replace_all("suffix", pl.col("suffix_val"), literal=True),
    )


@pytest.mark.parametrize("pattern", ["a", r"\d+"])
@pytest.mark.parametrize("literal", [True, False])
def test_count_matches(pattern: str, literal: bool) -> None:
    assert_eq(
        (bl_text.str.count_matches(pattern, literal=literal)),
        (pl_text.str.count_matches(pattern, literal=literal)),
    )


@pytest.mark.parametrize("length", [5, 10])
@pytest.mark.parametrize("fill_char", ["*", "-", " "])
def test_pad_start(length: int, fill_char: str) -> None:
    assert_eq(
        bl_text_short.str.pad_start(length, fill_char=fill_char),
        pl_text_short.str.pad_start(length, fill_char=fill_char),
    )


@pytest.mark.parametrize("length", [5, 10])
@pytest.mark.parametrize("fill_char", ["*", "-", " "])
def test_pad_end(length: int, fill_char: str) -> None:
    assert_eq(
        bl_text_short.str.pad_end(length, fill_char=fill_char),
        pl_text_short.str.pad_end(length, fill_char=fill_char),
    )


@pytest.mark.parametrize("length", [4, 5, 10])
def test_zfill(length: int) -> None:
    assert_eq(bl.col("numbers").str.zfill(length), pl.col("numbers").str.zfill(length))
    assert_eq(
        bl.col("signed_numbers").str.zfill(length),
        pl.col("signed_numbers").str.zfill(length),
    )


@pytest.mark.parametrize("encoding", ["base64", "hex"])
def test_encode(encoding: TransferEncoding) -> None:
    assert_eq(bl_text.str.encode(encoding), pl_text.str.encode(encoding))


def assert_eq(bl_expr: bl.Expr, polars_expr: pl.Expr) -> None:
    assert_frame_equal(
        sample_df().select(bl_expr).lazy(),
        sample_df().lazy().select(polars_expr),
        check_dtypes=False,
        check_row_order=False,
    )


def sample_df() -> bl.LazyFrame:
    return _LF
