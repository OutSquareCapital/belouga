# Roadmap

Here are notes of ideas and to-dos.

## TBD (to check, or verify)

- Views, etc... that are duckdb AND relation/expression specific, have to check more.
- Check how to integrate with narwhals (for framelib)
- Analyze geometric datatypes.
- Full interop with data formats and network(iceberg, credentials, etc...), for polars parity and full duckdb support.

## Refactors

- Metadata handling is messy AF

## Features

- Support geometrics datatypes (after check mentionned above)
- Check the interop between polars SQL interface and duckdb SQL interface, and see if we can make them work together in a way that is more seamless than the current state of affairs.
- Same but for expressions. potentially improve the current pushdowns of polars lazy exprs with duckdb exprs.
- See if refactoring scans in a Rust module to have native Python -> Pyarrow -> DuckDB conversions is a real performance boost, and if so, do it.
