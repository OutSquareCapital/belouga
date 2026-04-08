
# pql vs Polars API Comparison Report

This report shows the API coverage of pql compared to other libraries.

## Summary

Each summary cell is relative to Polars.

| Class               | Coverage                                                                                     | Implemented | Matched | Missing | Mismatched | Extra |
| ------------------- | -------------------------------------------------------------------------------------------- | ----------- | ------- | ------- | ---------- | ----- |
| LazyFrame           | <span style="color: #f39c12;">███</span><span style="color: #bdc3c7;">░░░░░░░</span> (38.8%) | 80          | 31      | 24      | 25         | 7     |
| Expr                | <span style="color: #f39c12;">████</span><span style="color: #bdc3c7;">░░░░░░</span> (43.0%) | 214         | 92      | 90      | 32         | 1     |
| LazyGroupBy         | <span style="color: #27ae60;">██████</span><span style="color: #bdc3c7;">░░░░</span> (62.5%) | 16          | 10      | 4       | 2          | 0     |
| ExprStrNameSpace    | <span style="color: #e74c3c;">██</span><span style="color: #bdc3c7;">░░░░░░░░</span> (29.8%) | 47          | 14      | 10      | 23         | 2     |
| ExprListNameSpace   | <span style="color: #f39c12;">████</span><span style="color: #bdc3c7;">░░░░░░</span> (46.5%) | 43          | 20      | 19      | 4          | 2     |
| ExprStructNameSpace | <span style="color: #f39c12;">████</span><span style="color: #bdc3c7;">░░░░░░</span> (40.0%) | 5           | 2       | 2       | 1          | 2     |
| ExprNameNameSpace   | <span style="color: #27ae60;">███████</span><span style="color: #bdc3c7;">░░░</span> (70.0%) | 10          | 7       | 3       | 0          | 2     |
| ExprArrNameSpace    | <span style="color: #f39c12;">█████</span><span style="color: #bdc3c7;">░░░░░</span> (54.8%) | 31          | 17      | 9       | 5          | 4     |
| ExprDtNameSpace     | <span style="color: #f39c12;">████</span><span style="color: #bdc3c7;">░░░░░░</span> (48.9%) | 45          | 22      | 18      | 5          | 2     |
| ModuleFunctions     | <span style="color: #e74c3c;">█</span><span style="color: #bdc3c7;">░░░░░░░░░</span> (14.3%) | 168         | 24      | 120     | 24         | 13    |
| selectors           | <span style="color: #f39c12;">█████</span><span style="color: #bdc3c7;">░░░░░</span> (54.3%) | 35          | 19      | 12      | 4          | 0     |
| DataType            | <span style="color: #f39c12;">█████</span><span style="color: #bdc3c7;">░░░░░</span> (57.1%) | 14          | 8       | 5       | 1          | 2     |
| Schema              | <span style="color: #e74c3c;"></span><span style="color: #bdc3c7;">░░░░░░░░░░</span> (0.0%)  | 17          | 0       | 17      | 0          | 0     |

## LazyFrame

### [x] Missing Methods (24)

- `collect_async`
  - **Polars**: (gevent: bool, engine: EngineType, optimizations: QueryOptFlags) -> Awaitable[DataFrame] | _GeventDataFrameResult[DataFrame]
- `collect_batches`
  - **Polars**: (chunk_size: int | None, maintain_order: bool, lazy: bool, engine: EngineType, optimizations: QueryOptFlags) -> Iterator[DataFrame]
- `deserialize`
  - **Polars**: (source: str | bytes | Path | IOBase, format: SerializationFormat) -> LazyFrame
- `group_by_dynamic`
  - **Polars**: (index_column: IntoExpr, every: str | timedelta, period: str | timedelta | None, offset: str | timedelta | None, include_boundaries: bool, closed: ClosedInterval, label: Label, group_by: IntoExpr | Iterable[IntoExpr] | None, start_by: StartBy) -> LazyGroupBy
- `inspect`
  - **Polars**: (fmt: str) -> LazyFrame
- `interpolate`
  - **Polars**: () -> LazyFrame
- `join_where`
  - **Polars**: (other: LazyFrame, *predicates: Expr | Iterable[Expr], suffix: str) -> LazyFrame
- `map_batches`
  - **Polars**: (function: Callable[[DataFrame], DataFrame], predicate_pushdown: bool, projection_pushdown: bool, slice_pushdown: bool, no_optimizations: bool, schema: None | SchemaDict, validate_output_schema: bool, streamable: bool) -> LazyFrame
- `match_to_schema`
  - **Polars**: (schema: SchemaDict | Schema, missing_columns: Literal['insert', 'raise'] | Mapping[str, Literal['insert', 'raise'] | Expr], missing_struct_fields: Literal['insert', 'raise'] | Mapping[str, Literal['insert', 'raise']], extra_columns: Literal['ignore', 'raise'], extra_struct_fields: Literal['ignore', 'raise'] | Mapping[str, Literal['ignore', 'raise']], integer_cast: Literal['upcast', 'forbid'] | Mapping[str, Literal['upcast', 'forbid']], float_cast: Literal['upcast', 'forbid'] | Mapping[str, Literal['upcast', 'forbid']]) -> LazyFrame
- `merge_sorted`
  - **Polars**: (other: LazyFrame, key: str) -> LazyFrame
- `pipe_with_schema`
  - **Polars**: (function: Callable[[LazyFrame, Schema], LazyFrame]) -> LazyFrame
- `profile`
  - **Polars**: (type_coercion: bool, predicate_pushdown: bool, projection_pushdown: bool, simplify_expression: bool, no_optimization: bool, slice_pushdown: bool, comm_subplan_elim: bool, comm_subexpr_elim: bool, cluster_with_columns: bool, collapse_joins: bool, show_plot: bool, truncate_nodes: int, figsize: tuple[int, int], engine: EngineType, optimizations: QueryOptFlags, **_kwargs: Any) -> tuple[DataFrame, DataFrame]
- `remote`
  - **Polars**: (context: ComputeContext | None, plan_type: PlanTypePreference, n_retries: int, engine: Engine, scaling_mode: ScalingMode) -> LazyFrameRemote
- `remove`
  - **Polars**: (*predicates: IntoExprColumn | Iterable[IntoExprColumn] | bool | list[bool] | np.ndarray[Any, Any], **constraints: Any) -> LazyFrame
- `rolling`
  - **Polars**: (index_column: IntoExpr, period: str | timedelta, offset: str | timedelta | None, closed: ClosedInterval, group_by: IntoExpr | Iterable[IntoExpr] | None) -> LazyGroupBy
- `serialize`
  - **Polars**: (file: IOBase | str | Path | None, format: SerializationFormat) -> bytes | str | None
- `set_sorted`
  - **Polars**: (column: str | list[str], *more_columns: str, descending: bool | list[bool], nulls_last: bool | list[bool]) -> LazyFrame
- `show_graph`
  - **Polars**: (optimized: bool, show: bool, output_path: str | Path | None, raw_output: bool, figsize: tuple[float, float], type_coercion: bool,_type_check: bool, predicate_pushdown: bool, projection_pushdown: bool, simplify_expression: bool, slice_pushdown: bool, comm_subplan_elim: bool, comm_subexpr_elim: bool, cluster_with_columns: bool, collapse_joins: bool, engine: EngineType, plan_stage: PlanStage,_check_order: bool, optimizations: QueryOptFlags) -> str | None
- `sink_batches`
  - **Polars**: (function: Callable[[DataFrame], bool | None], chunk_size: int | None, maintain_order: bool, lazy: bool, engine: EngineType, optimizations: QueryOptFlags) -> LazyFrame | None
- `sink_delta`
  - **Polars**: (target: DeltaTable, mode: Literal['error', 'append', 'overwrite', 'ignore', 'merge'], storage_options: StorageOptionsDict | None, credential_provider: CredentialProviderFunction | Literal['auto'] | None, delta_write_options: dict[str, Any] | None, delta_merge_options: dict[str, Any] | None, optimizations: QueryOptFlags) -> TableMerger | None
- `sink_iceberg`
  - **Polars**: (target: Table, mode: Literal['append', 'overwrite'], catalog: IcebergCatalogConfig | None, storage_options: StorageOptionsDict | None) -> DataFrame
- `sink_ipc`
  - **Polars**: (path: str | Path | IO[bytes] | PartitionBy, compression: IpcCompression | None, compat_level: CompatLevel | None, record_batch_size: int | None, maintain_order: bool, storage_options: StorageOptionsDict | None, credential_provider: CredentialProviderFunction | Literal['auto'] | None, retries: int | None, sync_on_close: SyncOnCloseMethod | None, mkdir: bool, lazy: bool, engine: EngineType, optimizations: QueryOptFlags, _record_batch_statistics: bool) -> LazyFrame | None
- `sql`
  - **Polars**: (query: str, table_name: str) -> LazyFrame
- `update`
  - **Polars**: (other: LazyFrame, on: str | Sequence[str] | None, how: Literal['left', 'inner', 'full'], left_on: str | Sequence[str] | None, right_on: str | Sequence[str] | None, include_nulls: bool, maintain_order: MaintainOrderJoin | None) -> LazyFrame

### [!] Signature Mismatches (25)

- `cast`
  - **Polars**: (`dtypes: Mapping[ColumnNameOrSelector | PolarsDataType, PolarsDataType | PythonDataType] | PolarsDataType | pl.DataTypeExpr | Schema`, `strict: bool`) -> LazyFrame
  - **pql**: (`dtypes: Mapping[str, DataType] | DataType`) -> Self
- `collect`
  - **Polars**: (`type_coercion: bool`, `predicate_pushdown: bool`, `projection_pushdown: bool`, `simplify_expression: bool`, `slice_pushdown: bool`, `comm_subplan_elim: bool`, `comm_subexpr_elim: bool`, `cluster_with_columns: bool`, `collapse_joins: bool`, `no_optimization: bool`, `engine: EngineType`, `background: bool`, `optimizations: QueryOptFlags`, `**_kwargs: Any`) -> DataFrame | InProcessQuery
  - **pql**: () -> DataFrame
- `describe`
  - **Polars**: (`percentiles: Sequence[float] | float | None`, `interpolation: QuantileMethod`) -> DataFrame
  - **pql**: () -> Self
- `drop`
  - **Polars**: (*columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector], `strict: bool`) -> LazyFrame
  - **pql**: (columns: TryIter[IntoExprColumn], *more_columns: IntoExprColumn) -> Self
- `drop_nans`
  - **Polars**: (`subset: ColumnNameOrSelector | Collection[ColumnNameOrSelector] | None`) -> LazyFrame
  - **pql**: (`subset: TryIter[str]`) -> Self
- `drop_nulls`
  - **Polars**: (`subset: ColumnNameOrSelector | Collection[ColumnNameOrSelector] | None`) -> LazyFrame
  - **pql**: (`subset: TryIter[str]`) -> Self
- `explain`
  - **Polars**: (`format: ExplainFormat`, `optimized: bool`, `type_coercion: bool`, `predicate_pushdown: bool`, `projection_pushdown: bool`, `simplify_expression: bool`, `slice_pushdown: bool`, `comm_subplan_elim: bool`, `comm_subexpr_elim: bool`, `cluster_with_columns: bool`, `collapse_joins: bool`, `streaming: bool`, `engine: EngineType`, `tree_format: bool | None`, `optimizations: QueryOptFlags`) -> str
  - **pql**: (`kind: ExplainType | ExplainTypeLiteral`) -> str
- `explode`
  - **Polars**: (columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector], *more_columns: ColumnNameOrSelector, `empty_as_null: bool`, `keep_nulls: bool`) -> LazyFrame
  - **pql**: (columns: TryIter[IntoExprColumn], *more_columns: IntoExprColumn) -> Self
- `fill_null`
  - **Polars**: (`value: Any | Expr | None`, strategy: FillNullStrategy | None, limit: int | None, `matches_supertype: bool`) -> LazyFrame
  - **pql**: (value: IntoExpr, strategy: FillNullStrategy | None, limit: int | None) -> Self
- `filter`
  - **Polars**: (`*predicates: IntoExprColumn | Iterable[IntoExprColumn] | bool | list[bool] | np.ndarray[Any, Any]`, **constraints: Any) -> LazyFrame
  - **pql**: (predicates: TryIter[IntoExprColumn], *more_predicates: IntoExprColumn, **constraints: IntoExpr) -> Self
- `group_by`
  - **Polars**: (`*by: IntoExpr | Iterable[IntoExpr]`, `maintain_order: bool`, `**named_by: IntoExpr`) -> LazyGroupBy
  - **pql**: (`keys: TryIter[IntoExpr]`, *more_keys: IntoExpr, `drop_null_keys: bool`, `strategy: GroupByClause | None`) -> LazyGroupBy
- `join`
  - **Polars**: (other: LazyFrame, `on: str | Expr | Sequence[str | Expr] | None`, how: JoinStrategy, `left_on: str | Expr | Sequence[str | Expr] | None`, `right_on: str | Expr | Sequence[str | Expr] | None`, suffix: str, `validate: JoinValidation`, `nulls_equal: bool`, `coalesce: bool | None`, `maintain_order: MaintainOrderJoin | None`, allow_parallel: bool, force_parallel: bool) -> LazyFrame
  - **pql**: (other: Self, `on: TryIter[str]`, how: JoinStrategy, `left_on: TryIter[str]`, `right_on: TryIter[str]`, suffix: str) -> Self
- `join_asof`
  - **Polars**: (other: LazyFrame, `left_on: str | None | Expr`, `right_on: str | None | Expr`, `on: str | None | Expr`, `by_left: str | Sequence[str] | None`, `by_right: str | Sequence[str] | None`, `by: str | Sequence[str] | None`, strategy: AsofJoinStrategy, suffix: str, `tolerance: str | int | float | timedelta | None`, allow_parallel: bool, force_parallel: bool, `coalesce: bool`, `allow_exact_matches: bool`, `check_sortedness: bool`) -> LazyFrame
  - **pql**: (other: Self, left_on: str | None, right_on: str | None, on: str | None, `by_left: TryIter[str]`, `by_right: TryIter[str]`, `by: TryIter[str]`, strategy: AsofJoinStrategy, suffix: str) -> Self
- `pivot`
  - **Polars**: (`on: ColumnNameOrSelector | Sequence[ColumnNameOrSelector]`, `on_columns: Sequence[Any] | pl.Series | pl.DataFrame`, `index: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None`, `values: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None`, `aggregate_function: PivotAgg | Expr | None`, maintain_order: bool, separator: str, `column_naming: Literal['auto', 'combine']`) -> LazyFrame
  - **pql**: (`on: TryIter[str]`, on_columns: Sequence[PythonLiteral], `index: TryIter[str]`, `values: TryIter[str]`, aggregate_function: PivotAgg, maintain_order: bool, separator: str) -> Self
- `quantile`
  - **Polars**: (`quantile: float | Expr`, interpolation: QuantileMethod) -> LazyFrame
  - **pql**: (quantile: float) -> Self
- `rename`
  - **Polars**: (`mapping: Mapping[str, str] | Callable[[str], str]`, `strict: bool`) -> LazyFrame
  - **pql**: (mapping: Mapping[str, str]) -> Self
- `shift`
  - **Polars**: (`n: int | IntoExprColumn`, `fill_value: IntoExpr | None`) -> LazyFrame
  - **pql**: (n: int, fill_value: IntoExpr) -> Self
- `show`
  - **Polars**: (`limit: int | None`, `ascii_tables: bool | None`, `decimal_separator: str | None`, `thousands_separator: str | bool | None`, `float_precision: int | None`, `fmt_float: FloatFmt | None`, `fmt_str_lengths: int | None`, `fmt_table_cell_list_len: int | None`, `tbl_cell_alignment: Alignment | None`, `tbl_cell_numeric_alignment: Alignment | None`, `tbl_cols: int | None`, `tbl_column_data_type_inline: bool | None`, `tbl_dataframe_shape_below: bool | None`, `tbl_formatting: TableFormatNames | None`, `tbl_hide_column_data_types: bool | None`, `tbl_hide_column_names: bool | None`, `tbl_hide_dtype_separator: bool | None`, `tbl_hide_dataframe_shape: bool | None`, `tbl_width_chars: int | None`, `trim_decimal_zeros: bool | None`) -> None
  - **pql**: (`max_width: SupportsInt | None`, `max_rows: SupportsInt | None`, `max_col_width: SupportsInt | None`, `null_value: str | None`, `render_mode: RenderModeLiteral | None`) -> None
- `sink_csv`
  - **Polars**: (`path: str | Path | IO[bytes] | IO[str] | PartitionBy`, `include_bom: bool`, `compression: Literal['uncompressed', 'gzip', 'zstd']`, `compression_level: int | None`, `check_extension: bool`, include_header: bool, separator: str, `line_terminator: str`, `quote_char: str`, `batch_size: int`, `datetime_format: str | None`, `date_format: str | None`, `time_format: str | None`, `float_scientific: bool | None`, `float_precision: int | None`, `decimal_comma: bool`, `null_value: str | None`, `quote_style: CsvQuoteStyle | None`, `maintain_order: bool`, `storage_options: StorageOptionsDict | None`, `credential_provider: CredentialProviderFunction | Literal['auto'] | None`, `retries: int | None`, `sync_on_close: SyncOnCloseMethod | None`, `mkdir: bool`, `lazy: bool`, `engine: EngineType`, `optimizations: QueryOptFlags`) -> LazyFrame | None
  - **pql**: (path: str | Path, separator: str, include_header: bool) -> None
- `sink_ndjson`
  - **Polars**: (`path: str | Path | IO[bytes] | IO[str] | PartitionBy`, `compression: Literal['uncompressed', 'gzip', 'zstd']`, `compression_level: int | None`, `check_extension: bool`, `maintain_order: bool`, `storage_options: StorageOptionsDict | None`, `credential_provider: CredentialProviderFunction | Literal['auto'] | None`, `retries: int | None`, `sync_on_close: SyncOnCloseMethod | None`, `mkdir: bool`, `lazy: bool`, `engine: EngineType`, `optimizations: QueryOptFlags`) -> LazyFrame | None
  - **pql**: (path: str | Path) -> None
- `sink_parquet`
  - **Polars**: (`path: str | Path | IO[bytes] | PartitionBy`, `compression: str`, `compression_level: int | None`, `statistics: bool | str | dict[str, bool]`, `row_group_size: int | None`, `data_page_size: int | None`, `maintain_order: bool`, `storage_options: StorageOptionsDict | None`, `credential_provider: CredentialProviderFunction | Literal['auto'] | None`, `retries: int | None`, `sync_on_close: SyncOnCloseMethod | None`, `metadata: ParquetMetadata | None`, `arrow_schema: ArrowSchemaExportable | None`, `mkdir: bool`, `lazy: bool`, `engine: EngineType`, `optimizations: QueryOptFlags`) -> LazyFrame | None
  - **pql**: (path: str | Path, `compression: ParquetCompression`) -> None
- `unique`
  - **Polars**: (`subset: IntoExpr | Collection[IntoExpr] | None`, keep: UniqueKeepStrategy, `maintain_order: bool`) -> LazyFrame
  - **pql**: (`subset: TryIter[str] | None`, keep: UniqueKeepStrategy, `order_by: TrySeq[str]`) -> Self
- `unnest`
  - **Polars**: (columns: ColumnNameOrSelector | Collection[ColumnNameOrSelector], *more_columns: ColumnNameOrSelector, `separator: str | None`) -> LazyFrame
  - **pql**: (`columns: TryIter[IntoExprColumn]`, *more_columns: IntoExprColumn) -> Self
- `unpivot`
  - **Polars**: (`on: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None`, `index: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None`, `variable_name: str | None`, `value_name: str | None`, `streamable: bool`) -> LazyFrame
  - **pql**: (`on: TryIter[str]`, `index: TryIter[str]`, variable_name: str, value_name: str, `order_by: TryIter[str]`) -> Self
- `with_row_index`
  - **Polars**: (name: str, `offset: int`) -> LazyFrame
  - **pql**: (name: str, `order_by: TryIter[str]`) -> Self

### [+] Extra Methods (pql-only) (7)

- `fetch_all`
- `group_by_all`
- `inner`
- `join_cross`
- `shape`
- `sql_query`
- `union`

## Expr

### [x] Missing Methods (90)

- `agg_groups`
  - **Polars**: () -> Expr
- `append`
  - **Polars**: (other: IntoExpr, upcast: bool) -> Expr
- `arg_max`
  - **Polars**: () -> Expr
- `arg_min`
  - **Polars**: () -> Expr
- `arg_true`
  - **Polars**: () -> Expr
- `arg_unique`
  - **Polars**: () -> Expr
- `bin`
  - **Polars**: () -> ExprBinaryNameSpace
- `bitwise_count_ones`
  - **Polars**: () -> Expr
- `bitwise_count_zeros`
  - **Polars**: () -> Expr
- `bitwise_leading_ones`
  - **Polars**: () -> Expr
- `bitwise_leading_zeros`
  - **Polars**: () -> Expr
- `bitwise_trailing_ones`
  - **Polars**: () -> Expr
- `bitwise_trailing_zeros`
  - **Polars**: () -> Expr
- `bottom_k`
  - **Polars**: (k: int | IntoExprColumn) -> Expr
- `bottom_k_by`
  - **Polars**: (by: IntoExpr | Iterable[IntoExpr], k: int | IntoExprColumn, reverse: bool | Sequence[bool]) -> Expr
- `cat`
  - **Polars**: () -> ExprCatNameSpace
- `cumulative_eval`
  - **Polars**: (expr: Expr, min_samples: int) -> Expr
- `cut`
  - **Polars**: (breaks: Sequence[float], labels: Sequence[str_] | None, left_closed: bool, include_breaks: bool) -> Expr
- `deserialize`
  - **Polars**: (source: str_ | Path | IOBase | bytes, format: SerializationFormat) -> Expr
- `drop_nans`
  - **Polars**: () -> Expr
- `drop_nulls`
  - **Polars**: () -> Expr
- `eq_missing`
  - **Polars**: (other: Any) -> Expr
- `ewm_mean`
  - **Polars**: (com: float | None, span: float | None, half_life: float | None, alpha: float | None, adjust: bool, min_samples: int, ignore_nulls: bool) -> Expr
- `ewm_mean_by`
  - **Polars**: (by: str_| IntoExpr, half_life: str_ | timedelta) -> Expr
- `ewm_std`
  - **Polars**: (com: float | None, span: float | None, half_life: float | None, alpha: float | None, adjust: bool, bias: bool, min_samples: int, ignore_nulls: bool) -> Expr
- `ewm_var`
  - **Polars**: (com: float | None, span: float | None, half_life: float | None, alpha: float | None, adjust: bool, bias: bool, min_samples: int, ignore_nulls: bool) -> Expr
- `exclude`
  - **Polars**: (columns: str_| PolarsDataType | Collection[str_ | PolarsDataType], *more_columns: str_ | PolarsDataType) -> Expr
- `explode`
  - **Polars**: (empty_as_null: bool, keep_nulls: bool) -> Expr
- `ext`
  - **Polars**: () -> ExprExtensionNameSpace
- `extend_constant`
  - **Polars**: (value: IntoExpr, n: int | IntoExprColumn) -> Expr
- `filter`
  - **Polars**: (*predicates: IntoExprColumn | Iterable[IntoExprColumn], **constraints: Any) -> Expr
- `from_json`
  - **Polars**: (value: str_) -> Expr
- `gather`
  - **Polars**: (indices: int | Sequence[int] | IntoExpr | Series | np.ndarray[Any, Any]) -> Expr
- `gather_every`
  - **Polars**: (n: int, offset: int) -> Expr
- `get`
  - **Polars**: (index: int | Expr, null_on_oob: bool) -> Expr
- `head`
  - **Polars**: (n: int | Expr) -> Expr
- `hist`
  - **Polars**: (bins: IntoExpr | None, bin_count: int | None, include_category: bool, include_breakpoint: bool) -> Expr
- `index_of`
  - **Polars**: (element: IntoExpr) -> Expr
- `inspect`
  - **Polars**: (fmt: str_) -> Expr
- `interpolate`
  - **Polars**: (method: InterpolationMethod) -> Expr
- `interpolate_by`
  - **Polars**: (by: IntoExpr) -> Expr
- `item`
  - **Polars**: (allow_empty: bool) -> Expr
- `limit`
  - **Polars**: (n: int | Expr) -> Expr
- `lower_bound`
  - **Polars**: () -> Expr
- `map_batches`
  - **Polars**: (function: Callable[[Series], Series | Any], return_dtype: DataTypeExpr | None, agg_list: bool, is_elementwise: bool, returns_scalar: bool) -> Expr
- `map_elements`
  - **Polars**: (function: Callable[[Any], Any], return_dtype: DataTypeExpr | None, skip_nulls: bool, pass_name: bool, strategy: MapElementsStrategy, returns_scalar: bool) -> Expr
- `meta`
  - **Polars**: () -> ExprMetaNameSpace
- `nan_max`
  - **Polars**: () -> Expr
- `nan_min`
  - **Polars**: () -> Expr
- `ne_missing`
  - **Polars**: (other: Any) -> Expr
- `peak_max`
  - **Polars**: () -> Expr
- `peak_min`
  - **Polars**: () -> Expr
- `qcut`
  - **Polars**: (quantiles: Sequence[float] | int, labels: Sequence[str_] | None, left_closed: bool, allow_duplicates: bool, include_breaks: bool) -> Expr
- `rechunk`
  - **Polars**: () -> Expr
- `reinterpret`
  - **Polars**: (signed: bool | None, dtype: PolarsDataType | None) -> Expr
- `replace_strict`
  - **Polars**: (old: IntoExpr | Sequence[Any] | Mapping[Any, Any], new: IntoExpr | Sequence[Any] | NoDefault, default: IntoExpr | NoDefault, return_dtype: DataTypeExpr | None) -> Expr
- `reshape`
  - **Polars**: (dimensions: tuple[int, ...]) -> Expr
- `reverse`
  - **Polars**: () -> Expr
- `rle`
  - **Polars**: () -> Expr
- `rle_id`
  - **Polars**: () -> Expr
- `rolling`
  - **Polars**: (index_column: IntoExprColumn, period: str_ | timedelta, offset: str_ | timedelta | None, closed: ClosedInterval) -> Expr
- `rolling_kurtosis`
  - **Polars**: (window_size: int, fisher: bool, bias: bool, min_samples: int | None, center: bool) -> Expr
- `rolling_map`
  - **Polars**: (function: Callable[[Series], Any], window_size: int, weights: list_[float] | None, min_samples: int | None, center: bool) -> Expr
- `rolling_max_by`
  - **Polars**: (by: IntoExpr, window_size: timedelta | str_, min_samples: int, closed: ClosedInterval) -> Expr
- `rolling_mean_by`
  - **Polars**: (by: IntoExpr, window_size: timedelta | str_, min_samples: int, closed: ClosedInterval) -> Expr
- `rolling_median_by`
  - **Polars**: (by: IntoExpr, window_size: timedelta | str_, min_samples: int, closed: ClosedInterval) -> Expr
- `rolling_min_by`
  - **Polars**: (by: IntoExpr, window_size: timedelta | str_, min_samples: int, closed: ClosedInterval) -> Expr
- `rolling_quantile`
  - **Polars**: (quantile: float, interpolation: QuantileMethod, window_size: int, weights: list_[float] | None, min_samples: int | None, center: bool) -> Expr
- `rolling_quantile_by`
  - **Polars**: (by: IntoExpr, window_size: timedelta | str_, quantile: float, interpolation: QuantileMethod, min_samples: int, closed: ClosedInterval) -> Expr
- `rolling_rank`
  - **Polars**: (window_size: int, method: RankMethod, seed: int | None, min_samples: int | None, center: bool) -> Expr
- `rolling_rank_by`
  - **Polars**: (by: IntoExpr, window_size: timedelta | str_, method: RankMethod, seed: int | None, min_samples: int, closed: ClosedInterval) -> Expr
- `rolling_skew`
  - **Polars**: (window_size: int, bias: bool, min_samples: int | None, center: bool) -> Expr
- `rolling_std_by`
  - **Polars**: (by: IntoExpr, window_size: timedelta | str_, min_samples: int, closed: ClosedInterval, ddof: int) -> Expr
- `rolling_sum_by`
  - **Polars**: (by: IntoExpr, window_size: timedelta | str_, min_samples: int, closed: ClosedInterval) -> Expr
- `rolling_var_by`
  - **Polars**: (by: IntoExpr, window_size: timedelta | str_, min_samples: int, closed: ClosedInterval, ddof: int) -> Expr
- `round_sig_figs`
  - **Polars**: (digits: int) -> Expr
- `sample`
  - **Polars**: (n: int | IntoExprColumn | None, fraction: float | IntoExprColumn | None, with_replacement: bool, shuffle: bool, seed: int | None) -> Expr
- `search_sorted`
  - **Polars**: (element: ndarray[Any, Any], side: SearchSortedSide, descending: bool) -> Expr
- `set_sorted`
  - **Polars**: (descending: bool, nulls_last: bool) -> Expr
- `shuffle`
  - **Polars**: (seed: int | None) -> Expr
- `slice`
  - **Polars**: (offset: int | Expr, length: int | Expr | None) -> Expr
- `sort`
  - **Polars**: (descending: bool, nulls_last: bool) -> Expr
- `sort_by`
  - **Polars**: (by: IntoExpr | Iterable[IntoExpr], *more_by: IntoExpr, descending: bool | Sequence[bool], nulls_last: bool | Sequence[bool], multithreaded: bool, maintain_order: bool) -> Expr
- `tail`
  - **Polars**: (n: int | Expr) -> Expr
- `to_physical`
  - **Polars**: () -> Expr
- `top_k`
  - **Polars**: (k: int | IntoExprColumn) -> Expr
- `top_k_by`
  - **Polars**: (by: IntoExpr | Iterable[IntoExpr], k: int | IntoExprColumn, reverse: bool | Sequence[bool]) -> Expr
- `unique_counts`
  - **Polars**: () -> Expr
- `upper_bound`
  - **Polars**: () -> Expr
- `value_counts`
  - **Polars**: (sort: bool, parallel: bool, name: str_ | None, normalize: bool) -> Expr

### [!] Signature Mismatches (32)

- `alias`
  - **Polars**: (`name: str_`) -> Expr
  - **pql**: (`name: str`) -> Self
- `all`
  - **Polars**: (`ignore_nulls: bool`) -> Expr
  - **pql**: () -> Self
- `any`
  - **Polars**: (`ignore_nulls: bool`) -> Expr
  - **pql**: () -> Self
- `cast`
  - **Polars**: (`dtype: DataTypeExpr | type[Any]`, `strict: bool`, `wrap_numerical: bool`) -> Expr
  - **pql**: (`dtype: DataType`) -> Self
- `clip`
  - **Polars**: (`lower_bound: NumericLiteral | TemporalLiteral | IntoExprColumn | None`, `upper_bound: NumericLiteral | TemporalLiteral | IntoExprColumn | None`) -> Expr
  - **pql**: (`lower_bound: IntoExpr`, `upper_bound: IntoExpr`) -> Self
- `diff`
  - **Polars**: (`n: int | IntoExpr`, `null_behavior: NullBehavior`) -> Expr
  - **pql**: () -> Self
- `dot`
  - **Polars**: (`other: Expr | str_`) -> Expr
  - **pql**: (`other: IntoExpr`) -> Self
- `fill_nan`
  - **Polars**: (`value: int | float | Expr | None`) -> Expr
  - **pql**: (`value: float | IntoExprColumn | None`) -> Self
- `fill_null`
  - **Polars**: (`value: Any | Expr | None`, strategy: FillNullStrategy | None, limit: int | None) -> Expr
  - **pql**: (value: IntoExpr, strategy: FillNullStrategy | None, limit: int | None) -> Self
- `forward_fill`
  - **Polars**: (`limit: int | None`) -> Expr
  - **pql**: () -> Self
- `hash`
  - **Polars**: (seed: int, `seed_1: int | None`, `seed_2: int | None`, `seed_3: int | None`) -> Expr
  - **pql**: (seed: int) -> Self
- `implode`
  - **Polars**: (`maintain_order: bool`) -> Expr
  - **pql**: () -> Self
- `is_in`
  - **Polars**: (`other: Expr | Collection[Any] | Series`, `nulls_equal: bool`) -> Expr
  - **pql**: (`other: TryIter[IntoExpr]`) -> Self
- `last`
  - **Polars**: (`ignore_nulls: bool`) -> Expr
  - **pql**: () -> Self
- `log`
  - **Polars**: (`base: float | IntoExpr`) -> Expr
  - **pql**: (base: float) -> Self
- `mode`
  - **Polars**: (`maintain_order: bool`) -> Expr
  - **pql**: () -> Self
- `over`
  - **Polars**: (`partition_by: IntoExpr | Iterable[IntoExpr] | None`, *more_exprs: IntoExpr, `order_by: IntoExpr | Iterable[IntoExpr] | None`, descending: bool, nulls_last: bool, `mapping_strategy: WindowMappingStrategy`) -> Expr
  - **pql**: (partition_by: TryIter[IntoExpr], *more_exprs: IntoExpr, order_by: TryIter[IntoExpr], descending: bool, nulls_last: bool) -> Self
- `pct_change`
  - **Polars**: (`n: int | IntoExprColumn`) -> Expr
  - **pql**: (n: int) -> Self
- `pow`
  - **Polars**: (`exponent: IntoExprColumn | int | float`) -> Expr
  - **pql**: (`other: IntoExprColumn | float`) -> Self
- `quantile`
  - **Polars**: (`quantile: float | list_[float] | Expr`, interpolation: QuantileMethod) -> Expr
  - **pql**: (quantile: float, interpolation: bool) -> Self
- `rank`
  - **Polars**: (method: RankMethod, descending: bool, `seed: int | None`) -> Expr
  - **pql**: (method: RankMethod, descending: bool) -> Self
- `repeat_by`
  - **Polars**: (`by: Series | Expr | str_ | int`) -> Expr
  - **pql**: (`by: IntoExprColumn | int`) -> Self
- `replace`
  - **Polars**: (`old: IntoExpr | Sequence[Any] | Mapping[Any, Any]`, `new: IntoExpr | Sequence[Any] | NoDefault`, `default: IntoExpr | NoDefault`, `return_dtype: PolarsDataType | None`) -> Expr
  - **pql**: (old: IntoExpr, new: IntoExpr) -> Self
- `rolling_max`
  - **Polars**: (window_size: int, `weights: list_[float] | None`, min_samples: int | None, center: bool) -> Expr
  - **pql**: (window_size: int, min_samples: int | None, center: bool) -> Self
- `rolling_mean`
  - **Polars**: (window_size: int, `weights: list_[float] | None`, min_samples: int | None, center: bool) -> Expr
  - **pql**: (window_size: int, min_samples: int | None, center: bool) -> Self
- `rolling_median`
  - **Polars**: (window_size: int, `weights: list_[float] | None`, min_samples: int | None, center: bool) -> Expr
  - **pql**: (window_size: int, min_samples: int | None, center: bool) -> Self
- `rolling_min`
  - **Polars**: (window_size: int, `weights: list_[float] | None`, min_samples: int | None, center: bool) -> Expr
  - **pql**: (window_size: int, min_samples: int | None, center: bool) -> Self
- `rolling_std`
  - **Polars**: (window_size: int, `weights: list_[float] | None`, min_samples: int | None, center: bool, ddof: int) -> Expr
  - **pql**: (window_size: int, min_samples: int | None, center: bool, ddof: int) -> Self
- `rolling_sum`
  - **Polars**: (window_size: int, `weights: list_[float] | None`, min_samples: int | None, center: bool) -> Expr
  - **pql**: (window_size: int, min_samples: int | None, center: bool) -> Self
- `rolling_var`
  - **Polars**: (window_size: int, `weights: list_[float] | None`, min_samples: int | None, center: bool, ddof: int) -> Expr
  - **pql**: (window_size: int, min_samples: int | None, center: bool, ddof: int) -> Self
- `shift`
  - **Polars**: (`n: int | IntoExprColumn`, `fill_value: IntoExpr | None`) -> Expr
  - **pql**: (n: int) -> Self
- `unique`
  - **Polars**: (`maintain_order: bool`) -> Expr
  - **pql**: () -> Self

### [+] Extra Methods (pql-only) (1)

- `inner`

## LazyGroupBy

### [x] Missing Methods (4)

- `having`
  - **Polars**: (*predicates: IntoExpr | Iterable[IntoExpr]) -> LazyGroupBy
- `head`
  - **Polars**: (n: int) -> LazyFrame
- `map_groups`
  - **Polars**: (function: Callable[[DataFrame], DataFrame], schema: SchemaDict | None) -> LazyFrame
- `tail`
  - **Polars**: (n: int) -> LazyFrame

### [!] Signature Mismatches (2)

- `first`
  - **Polars**: (`ignore_nulls: bool`) -> LazyFrame
  - **pql**: () -> LazyFrame
- `last`
  - **Polars**: (`ignore_nulls: bool`) -> LazyFrame
  - **pql**: () -> LazyFrame

## ExprStrNameSpace

### [x] Missing Methods (10)

- `contains_any`
  - **Polars**: (patterns: IntoExpr, ascii_case_insensitive: bool) -> Expr
- `decode`
  - **Polars**: (encoding: TransferEncoding, strict: bool) -> Expr
- `extract_groups`
  - **Polars**: (pattern: str) -> Expr
- `extract_many`
  - **Polars**: (patterns: IntoExpr, ascii_case_insensitive: bool, overlapping: bool, leftmost: bool) -> Expr
- `find_many`
  - **Polars**: (patterns: IntoExpr, ascii_case_insensitive: bool, overlapping: bool, leftmost: bool) -> Expr
- `json_decode`
  - **Polars**: (dtype: DataTypeExpr, infer_schema_length: int | None) -> Expr
- `replace_many`
  - **Polars**: (patterns: IntoExpr | Mapping[str, str], replace_with: IntoExpr | NoDefault, ascii_case_insensitive: bool, leftmost: bool) -> Expr
- `split_exact`
  - **Polars**: (by: IntoExpr, n: int, inclusive: bool) -> Expr
- `splitn`
  - **Polars**: (by: IntoExpr, n: int) -> Expr
- `to_integer`
  - **Polars**: (base: int | IntoExprColumn, dtype: PolarsIntegerType, strict: bool) -> Expr

### [!] Signature Mismatches (23)

- `contains`
  - **Polars**: (`pattern: str | Expr`, literal: bool, `strict: bool`) -> Expr
  - **pql**: (`pattern: IntoExprColumn`, literal: bool) -> Expr
- `count_matches`
  - **Polars**: (`pattern: str | Expr`, literal: bool) -> Expr
  - **pql**: (`pattern: IntoExprColumn`, literal: bool) -> Expr
- `ends_with`
  - **Polars**: (`suffix: str | Expr`) -> Expr
  - **pql**: (`suffix: IntoExprColumn`) -> Expr
- `extract_all`
  - **Polars**: (`pattern: str | Expr`) -> Expr
  - **pql**: (`pattern: IntoExprColumn`) -> Expr
- `find`
  - **Polars**: (`pattern: str | Expr`, literal: bool, `strict: bool`) -> Expr
  - **pql**: (`pattern: IntoExprColumn`, literal: bool) -> Expr
- `head`
  - **Polars**: (`n: int | IntoExprColumn`) -> Expr
  - **pql**: (n: int) -> Expr
- `normalize`
  - **Polars**: (`form: UnicodeForm`) -> Expr
  - **pql**: () -> Expr
- `pad_end`
  - **Polars**: (`length: int | IntoExprColumn`, fill_char: str) -> Expr
  - **pql**: (length: int, `fill_char: IntoExprColumn`) -> Expr
- `pad_start`
  - **Polars**: (`length: int | IntoExprColumn`, fill_char: str) -> Expr
  - **pql**: (length: int, `fill_char: IntoExprColumn`) -> Expr
- `replace`
  - **Polars**: (`pattern: str | Expr`, `value: str | Expr`, literal: bool, n: int) -> Expr
  - **pql**: (pattern: str, `value: IntoExprColumn`, literal: bool, n: int) -> Expr
- `replace_all`
  - **Polars**: (`pattern: str | Expr`, `value: str | Expr`, literal: bool) -> Expr
  - **pql**: (`pattern: IntoExprColumn`, `value: IntoExprColumn`, literal: bool) -> Expr
- `slice`
  - **Polars**: (`offset: int | IntoExprColumn`, `length: int | IntoExprColumn | None`) -> Expr
  - **pql**: (offset: int, length: int | None) -> Expr
- `split`
  - **Polars**: (`by: IntoExpr`, `inclusive: bool`, `literal: bool`, `strict: bool`) -> Expr
  - **pql**: (by: IntoExprColumn) -> Expr
- `starts_with`
  - **Polars**: (`prefix: str | Expr`) -> Expr
  - **pql**: (`prefix: IntoExprColumn`) -> Expr
- `strip_chars`
  - **Polars**: (`characters: IntoExpr`) -> Expr
  - **pql**: (`characters: IntoExprColumn | None`) -> Expr
- `strip_chars_end`
  - **Polars**: (`characters: IntoExpr`) -> Expr
  - **pql**: (`characters: IntoExprColumn | None`) -> Expr
- `strip_chars_start`
  - **Polars**: (`characters: IntoExpr`) -> Expr
  - **pql**: (`characters: IntoExprColumn | None`) -> Expr
- `strptime`
  - **Polars**: (`dtype: PolarsTemporalType`, `format: str | None`, `strict: bool`, `exact: bool`, `cache: bool`, `ambiguous: Ambiguous | Expr`) -> Expr
  - **pql**: (`format: IntoExprColumn`) -> Expr
- `tail`
  - **Polars**: (`n: int | IntoExprColumn`) -> Expr
  - **pql**: (n: int) -> Expr
- `to_date`
  - **Polars**: (format: str | None, `strict: bool`, `exact: bool`, `cache: bool`) -> Expr
  - **pql**: (`format: IntoExprColumn | None`) -> Expr
- `to_datetime`
  - **Polars**: (format: str | None, `time_unit: TimeUnit | None`, `time_zone: str | None`, `strict: bool`, `exact: bool`, `cache: bool`, `ambiguous: Ambiguous | Expr`) -> Expr
  - **pql**: (`format: IntoExprColumn | None`) -> Expr
- `to_time`
  - **Polars**: (format: str | None, `strict: bool`, `cache: bool`) -> Expr
  - **pql**: (`format: IntoExprColumn | None`) -> Expr
- `zfill`
  - **Polars**: (`length: int | IntoExprColumn`) -> Expr
  - **pql**: (`width: int`) -> Expr

### [+] Extra Methods (pql-only) (2)

- `inner`
- `pipe`

## ExprListNameSpace

### [x] Missing Methods (19)

- `agg`
  - **Polars**: (expr: Expr) -> Expr
- `arg_max`
  - **Polars**: () -> Expr
- `arg_min`
  - **Polars**: () -> Expr
- `concat`
  - **Polars**: (other: list[Expr | str] | Expr | str | Series | list[Any]) -> Expr
- `diff`
  - **Polars**: (n: int, null_behavior: NullBehavior) -> Expr
- `gather`
  - **Polars**: (indices: Expr | Series | list[int] | list[list[int]], null_on_oob: bool) -> Expr
- `gather_every`
  - **Polars**: (n: int | IntoExprColumn, offset: int | IntoExprColumn) -> Expr
- `head`
  - **Polars**: (n: int | str | Expr) -> Expr
- `item`
  - **Polars**: (allow_empty: bool) -> Expr
- `sample`
  - **Polars**: (n: int | IntoExprColumn | None, fraction: float | IntoExprColumn | None, with_replacement: bool, shuffle: bool, seed: int | None) -> Expr
- `set_difference`
  - **Polars**: (other: IntoExpr | Collection[Any]) -> Expr
- `set_intersection`
  - **Polars**: (other: IntoExpr | Collection[Any]) -> Expr
- `set_symmetric_difference`
  - **Polars**: (other: IntoExpr | Collection[Any]) -> Expr
- `set_union`
  - **Polars**: (other: IntoExpr | Collection[Any]) -> Expr
- `shift`
  - **Polars**: (n: int | IntoExprColumn) -> Expr
- `slice`
  - **Polars**: (offset: int | str | Expr, length: int | str | Expr | None) -> Expr
- `tail`
  - **Polars**: (n: int | str | Expr) -> Expr
- `to_array`
  - **Polars**: (width: int) -> Expr
- `to_struct`
  - **Polars**: (n_field_strategy: ListToStructWidthStrategy | None, fields: Sequence[str] | Callable[[int], str] | None, upper_bound: int | None) -> Expr

### [!] Signature Mismatches (4)

- `contains`
  - **Polars**: (item: IntoExpr, `nulls_equal: bool`) -> Expr
  - **pql**: (item: IntoExpr) -> Expr
- `explode`
  - **Polars**: (`empty_as_null: bool`, `keep_nulls: bool`) -> Expr
  - **pql**: () -> Expr
- `get`
  - **Polars**: (`index: int | Expr | str`, `null_on_oob: bool`) -> Expr
  - **pql**: (index: int) -> Expr
- `unique`
  - **Polars**: (`maintain_order: bool`) -> Expr
  - **pql**: () -> Expr

### [+] Extra Methods (pql-only) (2)

- `inner`
- `pipe`

## ExprStructNameSpace

### [x] Missing Methods (2)

- `rename_fields`
  - **Polars**: (names: Sequence[str]) -> Expr
- `unnest`
  - **Polars**: () -> Expr

### [!] Signature Mismatches (1)

- `field`
  - **Polars**: (`name: str | list[str]`, *more_names: str) -> Expr
  - **pql**: (name: str) -> Expr

### [+] Extra Methods (pql-only) (2)

- `inner`
- `pipe`

## ExprNameNameSpace

### [x] Missing Methods (3)

- `map_fields`
  - **Polars**: (function: Callable[[str], str]) -> Expr
- `prefix_fields`
  - **Polars**: (prefix: str) -> Expr
- `suffix_fields`
  - **Polars**: (suffix: str) -> Expr

### [+] Extra Methods (pql-only) (2)

- `inner`
- `pipe`

## ExprArrNameSpace

### [x] Missing Methods (9)

- `agg`
  - **Polars**: (expr: Expr) -> Expr
- `arg_max`
  - **Polars**: () -> Expr
- `arg_min`
  - **Polars**: () -> Expr
- `head`
  - **Polars**: (n: int | str | Expr, as_array: bool) -> Expr
- `shift`
  - **Polars**: (n: int | IntoExprColumn) -> Expr
- `slice`
  - **Polars**: (offset: int | str | Expr, length: int | str | Expr | None, as_array: bool) -> Expr
- `tail`
  - **Polars**: (n: int | str | Expr, as_array: bool) -> Expr
- `to_list`
  - **Polars**: () -> Expr
- `to_struct`
  - **Polars**: (fields: Sequence[str] | Callable[[int], str] | None) -> Expr

### [!] Signature Mismatches (5)

- `contains`
  - **Polars**: (item: IntoExpr, `nulls_equal: bool`) -> Expr
  - **pql**: (item: IntoExpr) -> Expr
- `eval`
  - **Polars**: (expr: Expr, `as_list: bool`) -> Expr
  - **pql**: (expr: Expr) -> Expr
- `explode`
  - **Polars**: (`empty_as_null: bool`, `keep_nulls: bool`) -> Expr
  - **pql**: () -> Expr
- `get`
  - **Polars**: (`index: int | IntoExprColumn`, `null_on_oob: bool`) -> Expr
  - **pql**: (index: int) -> Expr
- `unique`
  - **Polars**: (`maintain_order: bool`) -> Expr
  - **pql**: () -> Expr

### [+] Extra Methods (pql-only) (4)

- `drop_nulls`
- `filter`
- `inner`
- `pipe`

## ExprDtNameSpace

### [x] Missing Methods (18)

- `add_business_days`
  - **Polars**: (n: int | IntoExpr, week_mask: Iterable[bool], holidays: Iterable[dt.date] | Expr | pl.Series, roll: Roll) -> Expr
- `base_utc_offset`
  - **Polars**: () -> Expr
- `cast_time_unit`
  - **Polars**: (time_unit: TimeUnit) -> Expr
- `combine`
  - **Polars**: (time: time | Expr, time_unit: TimeUnit) -> Expr
- `convert_time_zone`
  - **Polars**: (time_zone: str) -> Expr
- `days_in_month`
  - **Polars**: () -> Expr
- `dst_offset`
  - **Polars**: () -> Expr
- `is_business_day`
  - **Polars**: (week_mask: Iterable[bool], holidays: Iterable[dt.date] | Expr | pl.Series) -> Expr
- `is_leap_year`
  - **Polars**: () -> Expr
- `replace`
  - **Polars**: (year: int | IntoExpr | None, month: int | IntoExpr | None, day: int | IntoExpr | None, hour: int | IntoExpr | None, minute: int | IntoExpr | None, second: int | IntoExpr | None, microsecond: int | IntoExpr | None, ambiguous: Ambiguous | Expr) -> Expr
- `replace_time_zone`
  - **Polars**: (time_zone: str | None, ambiguous: Ambiguous | Expr, non_existent: NonExistent) -> Expr
- `total_days`
  - **Polars**: (fractional: bool) -> Expr
- `total_hours`
  - **Polars**: (fractional: bool) -> Expr
- `total_microseconds`
  - **Polars**: (fractional: bool) -> Expr
- `total_milliseconds`
  - **Polars**: (fractional: bool) -> Expr
- `total_minutes`
  - **Polars**: (fractional: bool) -> Expr
- `total_nanoseconds`
  - **Polars**: (fractional: bool) -> Expr
- `total_seconds`
  - **Polars**: (fractional: bool) -> Expr

### [!] Signature Mismatches (5)

- `offset_by`
  - **Polars**: (`by: str | Expr`) -> Expr
  - **pql**: (`by: IntoExpr`) -> Expr
- `round`
  - **Polars**: (`every: timedelta | IntoExprColumn`) -> Expr
  - **pql**: (every: str) -> Expr
- `second`
  - **Polars**: (`fractional: bool`) -> Expr
  - **pql**: () -> Expr
- `to_string`
  - **Polars**: (`format: str | None`) -> Expr
  - **pql**: (`format: IntoExprColumn`) -> Expr
- `truncate`
  - **Polars**: (`every: timedelta | Expr`) -> Expr
  - **pql**: (`every: str`) -> Expr

### [+] Extra Methods (pql-only) (2)

- `inner`
- `pipe`

## ModuleFunctions

### [x] Missing Methods (120)

- `BaseExtension`
  - **Polars**: (name: str, storage: PolarsDataType, metadata: str | None) -> None
- `Catalog`
  - **Polars**: (workspace_url: str, bearer_token: str | None, require_https: bool) -> None
- `CompatLevel`
  - **Polars**: () -> None
- `DataTypeExpr`
  - **Polars**: ()
- `Extension`
  - **Polars**: (name: str, storage: PolarsDataType, metadata: str | None) -> None
- `Field`
  - **Polars**: (name: str, dtype: PolarsDataType) -> None
- `Float16`
  - **Polars**: ()
- `Null`
  - **Polars**: ()
- `Object`
  - **Polars**: ()
- `ScanCastOptions`
  - **Polars**: (integer_cast: Literal['forbid'] | IntegerCastOption | Collection[IntegerCastOption], float_cast: Literal['forbid'] | FloatCastOption | Collection[FloatCastOption], datetime_cast: Literal['forbid'] | DatetimeCastOption | Collection[DatetimeCastOption], missing_struct_fields: Literal['insert', 'raise'], extra_struct_fields: Literal['ignore', 'raise'], categorical_to_string: Literal['allow', 'forbid'],_internal_call: bool) -> None
- `Schema`
  - **Polars**: (schema: Mapping[str, SchemaInitDataType] | Iterable[tuple[str, SchemaInitDataType] | ArrowSchemaExportable] | ArrowSchemaExportable | None, check_dtypes: bool) -> None
- `Unknown`
  - **Polars**: ()
- `align_frames`
  - **Polars**: (*frames: FrameType | Iterable[FrameType], on: str | Expr | Sequence[str] | Sequence[Expr] | Sequence[str | Expr], how: JoinStrategy, select: str | Expr | Sequence[str | Expr] | None, descending: bool | Sequence[bool]) -> list[FrameType]
- `any`
  - **Polars**: (*names: str, ignore_nulls: bool) -> Expr | bool | None
- `approx_n_unique`
  - **Polars**: (*columns: str) -> Expr
- `arange`
  - **Polars**: (start: int | IntoExprColumn, end: int | IntoExprColumn | None, step: int, dtype: PolarsIntegerType | DataTypeExpr, eager: bool) -> Expr | Series
- `arctan2`
  - **Polars**: (y: str | Expr, x: str | Expr) -> Expr
- `arg_sort_by`
  - **Polars**: (exprs: IntoExpr | Iterable[IntoExpr], *more_exprs: IntoExpr, descending: bool | Sequence[bool], nulls_last: bool | Sequence[bool], multithreaded: bool, maintain_order: bool) -> Expr
- `arg_where`
  - **Polars**: (condition: Expr | Series, eager: bool) -> Expr | Series
- `build_info`
  - **Polars**: () -> dict[str, Any]
- `business_day_count`
  - **Polars**: (start: date | IntoExprColumn, end: date | IntoExprColumn, week_mask: Iterable[bool], holidays: Iterable[date] | Expr | Series) -> Expr
- `collect_all`
  - **Polars**: (lazy_frames: Iterable[LazyFrame], type_coercion: bool, predicate_pushdown: bool, projection_pushdown: bool, simplify_expression: bool, no_optimization: bool, slice_pushdown: bool, comm_subplan_elim: bool, comm_subexpr_elim: bool, cluster_with_columns: bool, collapse_joins: bool, optimizations: QueryOptFlags, engine: EngineType, lazy: bool) -> list[DataFrame] | LazyFrame
- `collect_all_async`
  - **Polars**: (lazy_frames: Iterable[LazyFrame], gevent: bool, engine: EngineType, optimizations: QueryOptFlags) -> Awaitable[list[DataFrame]] |_GeventDataFrameResult[list[DataFrame]]
- `concat`
  - **Polars**: (items: Iterable[PolarsType], how: ConcatMethod, rechunk: bool, parallel: bool, strict: bool) -> PolarsType
- `concat_arr`
  - **Polars**: (exprs: IntoExpr | Iterable[IntoExpr], *more_exprs: IntoExpr) -> Expr
- `concat_list`
  - **Polars**: (exprs: IntoExpr | Iterable[IntoExpr], *more_exprs: IntoExpr) -> Expr
- `concat_str`
  - **Polars**: (exprs: IntoExpr | Iterable[IntoExpr], *more_exprs: IntoExpr, separator: str, ignore_nulls: bool) -> Expr
- `corr`
  - **Polars**: (a: IntoExpr, b: IntoExpr, method: CorrelationMethod, ddof: int | None, propagate_nans: bool, eager: bool) -> Expr | Series
- `count`
  - **Polars**: (*columns: str) -> Expr
- `cov`
  - **Polars**: (a: IntoExpr, b: IntoExpr, ddof: int, eager: bool) -> Expr | Series
- `cum_count`
  - **Polars**: (*columns: str, reverse: bool) -> Expr
- `cum_fold`
  - **Polars**: (acc: IntoExpr, function: Callable[[Series, Series], Series], exprs: Sequence[Expr | str] | Expr, returns_scalar: bool, return_dtype: DataTypeExpr | PolarsDataType | None, include_init: bool) -> Expr
- `cum_reduce`
  - **Polars**: (function: Callable[[Series, Series], Series], exprs: Sequence[Expr | str] | Expr, returns_scalar: bool, return_dtype: DataTypeExpr | PolarsDataType | None) -> Expr
- `cum_sum`
  - **Polars**: (*names: str) -> Expr
- `cum_sum_horizontal`
  - **Polars**: (*exprs: IntoExpr | Iterable[IntoExpr]) -> Expr
- `date`
  - **Polars**: (year: Expr | str | int, month: Expr | str | int, day: Expr | str | int) -> Expr
- `date_range`
  - **Polars**: (start: date | datetime | IntoExprColumn, end: date | datetime | IntoExprColumn, interval: str | timedelta, closed: ClosedInterval, eager: bool) -> Series | Expr
- `date_ranges`
  - **Polars**: (start: date | datetime | IntoExprColumn, end: date | datetime | IntoExprColumn, interval: str | timedelta, closed: ClosedInterval, eager: bool) -> Series | Expr
- `datetime`
  - **Polars**: (year: int | IntoExpr, month: int | IntoExpr, day: int | IntoExpr, hour: int | IntoExpr | None, minute: int | IntoExpr | None, second: int | IntoExpr | None, microsecond: int | IntoExpr | None, time_unit: TimeUnit, time_zone: str | None, ambiguous: Ambiguous | Expr) -> Expr
- `datetime_range`
  - **Polars**: (start: datetime | date | IntoExprColumn, end: datetime | date | IntoExprColumn, interval: str | timedelta, closed: ClosedInterval, time_unit: TimeUnit | None, time_zone: str | None, eager: bool) -> Series | Expr
- `datetime_ranges`
  - **Polars**: (start: datetime | date | IntoExprColumn, end: datetime | date | IntoExprColumn, interval: str | timedelta, closed: ClosedInterval, time_unit: TimeUnit | None, time_zone: str | None, eager: bool) -> Series | Expr
- `defer`
  - **Polars**: (function: Callable[[], DataFrame], schema: SchemaDict | Callable[[], SchemaDict], validate_schema: bool) -> LazyFrame
- `dtype_of`
  - **Polars**: (col_or_expr: str | Expr) -> DataTypeExpr
- `duration`
  - **Polars**: (weeks: Expr | str | int | float | None, days: Expr | str | int | float | None, hours: Expr | str | int | float | None, minutes: Expr | str | int | float | None, seconds: Expr | str | int | float | None, milliseconds: Expr | str | int | float | None, microseconds: Expr | str | int | float | None, nanoseconds: Expr | str | int | float | None, time_unit: TimeUnit | None) -> Expr
- `escape_regex`
  - **Polars**: (s: str) -> str
- `exclude`
  - **Polars**: (columns: str | PolarsDataType | Collection[str] | Collection[PolarsDataType], *more_columns: str | PolarsDataType) -> Expr
- `explain_all`
  - **Polars**: (lazy_frames: Iterable[LazyFrame], optimizations: QueryOptFlags) -> str
- `field`
  - **Polars**: (name: str | list[str]) -> Expr
- `first`
  - **Polars**: (*columns: str) -> Expr
- `fold`
  - **Polars**: (acc: IntoExpr, function: Callable[[Series, Series], Series], exprs: Sequence[Expr | str] | Expr, returns_scalar: bool, return_dtype: DataTypeExpr | PolarsDataType | None) -> Expr
- `format`
  - **Polars**: (f_string: str, *args: IntoExpr) -> Expr
- `from_arrow`
  - **Polars**: (data: RecordBatch | Iterable[pa.RecordBatch | pa.Table] | ArrowArrayExportable | ArrowStreamExportable, schema: SchemaDefinition | None, schema_overrides: SchemaDict | None, rechunk: bool) -> DataFrame | Series
- `from_dataframe`
  - **Polars**: (df: SupportsInterchange | ArrowArrayExportable | ArrowStreamExportable, allow_copy: bool | None, rechunk: bool) -> DataFrame
- `from_epoch`
  - **Polars**: (column: str | Expr | Series | Sequence[int | float], time_unit: EpochTimeUnit) -> Expr | Series
- `from_pandas`
  - **Polars**: (data: Series[Any] | pd.Index[Any] | pd.DatetimeIndex, schema_overrides: SchemaDict | None, rechunk: bool, nan_to_null: bool, include_index: bool) -> DataFrame | Series
- `from_repr`
  - **Polars**: (data: str) -> DataFrame | Series
- `from_torch`
  - **Polars**: (tensor: Tensor, schema: SchemaDefinition | None, schema_overrides: SchemaDict | None, orient: Orientation | None, force: bool) -> DataFrame
- `get_index_type`
  - **Polars**: () -> PolarsIntegerType
- `groups`
  - **Polars**: (column: str) -> Expr
- `head`
  - **Polars**: (column: str, n: int) -> Expr
- `implode`
  - **Polars**: (*columns: str, maintain_order: bool) -> Expr
- `int_range`
  - **Polars**: (start: int | IntoExprColumn, end: int | IntoExprColumn | None, step: int, dtype: PolarsIntegerType | DataTypeExpr, eager: bool) -> Expr | Series
- `int_ranges`
  - **Polars**: (start: int | IntoExprColumn, end: int | IntoExprColumn | None, step: int | IntoExprColumn, dtype: PolarsIntegerType | DataTypeExpr, eager: bool) -> Expr | Series
- `json_normalize`
  - **Polars**: (data: dict[Any, Any] | Sequence[dict[Any, Any] | Any], separator: str, max_level: int | None, schema: Schema | None, strict: bool, infer_schema_length: int | None, encoder: JSONEncoder | None) -> DataFrame
- `last`
  - **Polars**: (*columns: str) -> Expr
- `linear_space`
  - **Polars**: (start: NumericLiteral | TemporalLiteral | IntoExpr, end: NumericLiteral | TemporalLiteral | IntoExpr, num_samples: int | IntoExpr, closed: ClosedInterval, eager: bool) -> Expr | Series
- `linear_spaces`
  - **Polars**: (start: NumericLiteral | TemporalLiteral | IntoExprColumn, end: NumericLiteral | TemporalLiteral | IntoExprColumn, num_samples: int | IntoExprColumn, closed: ClosedInterval, as_array: bool, eager: bool) -> Expr | Series
- `map_batches`
  - **Polars**: (exprs: Sequence[str | Expr], function: Callable[[Sequence[Series]], Series | Any], return_dtype: DataTypeExpr | None, is_elementwise: bool, returns_scalar: bool) -> Expr
- `map_groups`
  - **Polars**: (exprs: Sequence[str | Expr], function: Callable[[Sequence[Series]], Series | Any], return_dtype: DataTypeExpr | None, is_elementwise: bool, returns_scalar: bool) -> Expr
- `n_unique`
  - **Polars**: (*columns: str) -> Expr
- `nth`
  - **Polars**: (*indices: int | Sequence[int], strict: bool) -> Expr
- `ones`
  - **Polars**: (n: int | Expr, dtype: PolarsDataType, eager: bool) -> Expr | Series
- `quantile`
  - **Polars**: (column: str, quantile: float | Expr, interpolation: QuantileMethod) -> Expr
- `read_avro`
  - **Polars**: (source: str | Path | IO[bytes] | bytes, columns: list[int] | list[str] | None, n_rows: int | None) -> DataFrame
- `read_clipboard`
  - **Polars**: (separator: str, **kwargs: Any) -> DataFrame
- `read_csv`
  - **Polars**: (source: str | Path | IO[str] | IO[bytes] | bytes, has_header: bool, columns: Sequence[int] | Sequence[str] | None, new_columns: Sequence[str] | None, separator: str, comment_prefix: str | None, quote_char: str | None, skip_rows: int, skip_lines: int, schema: SchemaDict | None, schema_overrides: Mapping[str, PolarsDataType] | Sequence[PolarsDataType] | None, null_values: str | Sequence[str] | dict[str, str] | None, missing_utf8_is_empty_string: bool, ignore_errors: bool, try_parse_dates: bool, n_threads: int | None, infer_schema: bool, infer_schema_length: int | None, batch_size: int, n_rows: int | None, encoding: CsvEncoding | str, low_memory: bool, rechunk: bool, use_pyarrow: bool, storage_options: StorageOptionsDict | None, skip_rows_after_header: int, row_index_name: str | None, row_index_offset: int, sample_size: int, eol_char: str, raise_if_empty: bool, truncate_ragged_lines: bool, decimal_comma: bool, glob: bool) -> DataFrame
- `read_database`
  - **Polars**: (query: str | TextClause | Selectable, connection: ConnectionOrCursor | str, iter_batches: bool, batch_size: int | None, schema_overrides: SchemaDict | None, infer_schema_length: int | None, execute_options: dict[str, Any] | None) -> DataFrame | Iterator[DataFrame]
- `read_database_uri`
  - **Polars**: (query: list[str] | str, uri: str, partition_on: str | None, partition_range: tuple[int, int] | None, partition_num: int | None, protocol: str | None, engine: DbReadEngine | None, schema_overrides: SchemaDict | None, execute_options: dict[str, Any] | None, pre_execution_query: str | list[str] | None) -> DataFrame
- `read_delta`
  - **Polars**: (source: str | Path | DeltaTable, version: int | str | datetime | None, columns: list[str] | None, rechunk: bool | None, storage_options: StorageOptionsDict | None, credential_provider: CredentialProviderFunction | Literal['auto'] | None, delta_table_options: dict[str, Any] | None, use_pyarrow: bool, pyarrow_options: dict[str, Any] | None) -> DataFrame
- `read_excel`
  - **Polars**: (source: FileSource, sheet_id: int | Sequence[int] | None, sheet_name: str | list[str] | tuple[str, ...] | None, table_name: str | None, engine: ExcelSpreadsheetEngine, engine_options: dict[str, Any] | None, read_options: dict[str, Any] | None, has_header: bool, columns: Sequence[int] | Sequence[str] | str | None, schema_overrides: SchemaDict | None, infer_schema_length: int | None, include_file_paths: str | None, drop_empty_rows: bool, drop_empty_cols: bool, raise_if_empty: bool) -> DataFrame | dict[str, pl.DataFrame]
- `read_ipc`
  - **Polars**: (source: str | Path | IO[bytes] | bytes, columns: list[int] | list[str] | None, n_rows: int | None, use_pyarrow: bool, memory_map: bool, storage_options: StorageOptionsDict | None, row_index_name: str | None, row_index_offset: int, rechunk: bool) -> DataFrame
- `read_ipc_schema`
  - **Polars**: (source: str | Path | IO[bytes] | bytes) -> dict[str, DataType]
- `read_ipc_stream`
  - **Polars**: (source: str | Path | IO[bytes] | bytes, columns: list[int] | list[str] | None, n_rows: int | None, use_pyarrow: bool, storage_options: StorageOptionsDict | None, row_index_name: str | None, row_index_offset: int, rechunk: bool) -> DataFrame
- `read_json`
  - **Polars**: (source: str | Path | IOBase | bytes, schema: SchemaDefinition | None, schema_overrides: SchemaDefinition | None, infer_schema_length: int | None) -> DataFrame
- `read_lines`
  - **Polars**: (source: str | Path | IO[str] | IO[bytes] | bytes | list[str] | list[Path] | list[IO[str]] | list[IO[bytes]], name: str, n_rows: int | None, row_index_name: str | None, row_index_offset: int, glob: bool, storage_options: StorageOptionsDict | None, credential_provider: CredentialProviderFunction | Literal['auto'] | None, include_file_paths: str | None) -> DataFrame
- `read_ndjson`
  - **Polars**: (source: str | Path | IO[str] | IO[bytes] | bytes | list[str] | list[Path] | list[IO[str]] | list[IO[bytes]], schema: SchemaDefinition | None, schema_overrides: SchemaDefinition | None, infer_schema_length: int | None, batch_size: int | None, n_rows: int | None, low_memory: bool, rechunk: bool, row_index_name: str | None, row_index_offset: int, ignore_errors: bool, storage_options: StorageOptionsDict | None, credential_provider: CredentialProviderFunction | Literal['auto'] | None, retries: int | None, file_cache_ttl: int | None, include_file_paths: str | None) -> DataFrame
- `read_ods`
  - **Polars**: (source: FileSource, sheet_id: int | Sequence[int] | None, sheet_name: str | list[str] | tuple[str, ...] | None, has_header: bool, columns: Sequence[int] | Sequence[str] | None, schema_overrides: SchemaDict | None, infer_schema_length: int | None, include_file_paths: str | None, drop_empty_rows: bool, drop_empty_cols: bool, raise_if_empty: bool) -> DataFrame | dict[str, pl.DataFrame]
- `read_parquet`
  - **Polars**: (source: FileSource, columns: list[int] | list[str] | None, n_rows: int | None, row_index_name: str | None, row_index_offset: int, parallel: ParallelStrategy, use_statistics: bool, hive_partitioning: bool | None, glob: bool, schema: SchemaDict | None, hive_schema: SchemaDict | None, try_parse_hive_dates: bool, rechunk: bool, low_memory: bool, storage_options: StorageOptionsDict | None, credential_provider: CredentialProviderFunction | Literal['auto'] | None, retries: int | None, use_pyarrow: bool, pyarrow_options: dict[str, Any] | None, memory_map: bool, include_file_paths: str | None, missing_columns: Literal['insert', 'raise'], allow_missing_columns: bool | None) -> DataFrame
- `read_parquet_metadata`
  - **Polars**: (source: str | Path | IO[bytes] | bytes, storage_options: StorageOptionsDict | None, credential_provider: CredentialProviderFunction | Literal['auto'] | None, retries: int | None) -> dict[str, str]
- `read_parquet_schema`
  - **Polars**: (source: str | Path | IO[bytes] | bytes) -> dict[str, DataType]
- `reduce`
  - **Polars**: (function: Callable[[Series, Series], Series], exprs: Sequence[Expr | str] | Expr, returns_scalar: bool, return_dtype: DataTypeExpr | PolarsDataType | None) -> Expr
- `register_extension_type`
  - **Polars**: (ext_name: str, ext_class: type[dt.BaseExtension] | None, as_storage: bool) -> None
- `repeat`
  - **Polars**: (value: IntoExpr | None, n: int | Expr, dtype: PolarsDataType | None, eager: bool) -> Expr | Series
- `rolling_corr`
  - **Polars**: (a: str | Expr, b: str | Expr, window_size: int, min_samples: int | None, ddof: int) -> Expr
- `rolling_cov`
  - **Polars**: (a: str | Expr, b: str | Expr, window_size: int, min_samples: int | None, ddof: int) -> Expr
- `row_index`
  - **Polars**: (name: str) -> Expr
- `scan_csv`
  - **Polars**: (source: str | Path | IO[str] | IO[bytes] | bytes | list[str] | list[Path] | list[IO[str]] | list[IO[bytes]] | list[bytes], has_header: bool, separator: str, comment_prefix: str | None, quote_char: str | None, skip_rows: int, skip_lines: int, schema: SchemaDict | None, schema_overrides: SchemaDict | Sequence[PolarsDataType] | None, null_values: str | Sequence[str] | dict[str, str] | None, missing_utf8_is_empty_string: bool, ignore_errors: bool, cache: bool | None, with_column_names: Callable[[list[str]], list[str]] | None, infer_schema: bool, infer_schema_length: int | None, n_rows: int | None, encoding: CsvEncoding, low_memory: bool, rechunk: bool, skip_rows_after_header: int, row_index_name: str | None, row_index_offset: int, try_parse_dates: bool, eol_char: str, new_columns: Sequence[str] | None, raise_if_empty: bool, truncate_ragged_lines: bool, decimal_comma: bool, glob: bool, storage_options: StorageOptionsDict | None, credential_provider: CredentialProviderFunction | Literal['auto'] | None, retries: int | None, file_cache_ttl: int | None, include_file_paths: str | None, missing_columns: Literal['insert', 'raise'] | None) -> LazyFrame
- `scan_delta`
  - **Polars**: (source: str | Path | DeltaTable, version: int | str | datetime | None, storage_options: StorageOptionsDict | None, credential_provider: CredentialProviderFunction | Literal['auto'] | None, delta_table_options: dict[str, Any] | None, use_pyarrow: bool, pyarrow_options: dict[str, Any] | None, rechunk: bool | None) -> LazyFrame
- `scan_iceberg`
  - **Polars**: (source: Table, snapshot_id: int | None, storage_options: StorageOptionsDict | None, catalog: IcebergCatalogConfig | None, reader_override: Literal['native', 'pyiceberg'] | None, use_metadata_statistics: bool, fast_deletion_count: bool | None, use_pyiceberg_filter: bool) -> LazyFrame
- `scan_ipc`
  - **Polars**: (source: str | Path | IO[bytes] | bytes | list[str] | list[Path] | list[IO[bytes]] | list[bytes], n_rows: int | None, cache: bool, rechunk: bool, row_index_name: str | None, row_index_offset: int, glob: bool, storage_options: StorageOptionsDict | None, credential_provider: CredentialProviderFunction | Literal['auto'] | None, memory_map: bool, retries: int | None, file_cache_ttl: int | None, hive_partitioning: bool | None, hive_schema: SchemaDict | None, try_parse_hive_dates: bool, include_file_paths: str | None,_record_batch_statistics: bool) -> LazyFrame
- `scan_lines`
  - **Polars**: (source: str | Path | IO[str] | IO[bytes] | bytes | list[str] | list[Path] | list[IO[str]] | list[IO[bytes]], name: str, n_rows: int | None, row_index_name: str | None, row_index_offset: int, glob: bool, storage_options: StorageOptionsDict | None, credential_provider: CredentialProviderFunction | Literal['auto'] | None, include_file_paths: str | None) -> LazyFrame
- `scan_ndjson`
  - **Polars**: (source: str | Path | IO[str] | IO[bytes] | bytes | list[str] | list[Path] | list[IO[str]] | list[IO[bytes]], schema: SchemaDefinition | None, schema_overrides: SchemaDefinition | None, infer_schema_length: int | None, batch_size: int | None, n_rows: int | None, low_memory: bool, rechunk: bool, row_index_name: str | None, row_index_offset: int, ignore_errors: bool, storage_options: StorageOptionsDict | None, credential_provider: CredentialProviderFunction | Literal['auto'] | None, retries: int | None, file_cache_ttl: int | None, include_file_paths: str | None) -> LazyFrame
- `scan_parquet`
  - **Polars**: (source: FileSource, n_rows: int | None, row_index_name: str | None, row_index_offset: int, parallel: ParallelStrategy, use_statistics: bool, hive_partitioning: bool | None, glob: bool, hidden_file_prefix: str | Sequence[str] | None, schema: SchemaDict | None, hive_schema: SchemaDict | None, try_parse_hive_dates: bool, rechunk: bool, low_memory: bool, cache: bool, storage_options: StorageOptionsDict | None, credential_provider: CredentialProviderFunction | Literal['auto'] | None, retries: int | None, include_file_paths: str | None, missing_columns: Literal['insert', 'raise'], allow_missing_columns: bool | None, extra_columns: Literal['ignore', 'raise'], cast_options: ScanCastOptions | None,_column_mapping: ColumnMapping | None,_default_values: DefaultFieldValues | None,_deletion_files: DeletionFiles | None,_table_statistics: DataFrame | None,_row_count: tuple[int, int] | None) -> LazyFrame
- `scan_pyarrow_dataset`
  - **Polars**: (source: Dataset, allow_pyarrow_filter: bool, batch_size: int | None) -> LazyFrame
- `select`
  - **Polars**: (*exprs: IntoExpr | Iterable[IntoExpr], eager: bool, **named_exprs: IntoExpr) -> DataFrame | LazyFrame
- `self_dtype`
  - **Polars**: () -> DataTypeExpr
- `set_random_seed`
  - **Polars**: (seed: int) -> None
- `sql`
  - **Polars**: (query: str, eager: bool) -> DataFrame | LazyFrame
- `sql_expr`
  - **Polars**: (sql: str | Sequence[str]) -> Expr | list[Expr]
- `std`
  - **Polars**: (column: str, ddof: int) -> Expr
- `struct`
  - **Polars**: (*exprs: IntoExpr | Iterable[IntoExpr], schema: SchemaDict | None, eager: bool, **named_exprs: IntoExpr) -> Expr | Series
- `struct_with_fields`
  - **Polars**: (mapping: Mapping[str, PolarsDataType | pl.DataTypeExpr]) -> DataTypeExpr
- `tail`
  - **Polars**: (column: str, n: int) -> Expr
- `thread_pool_size`
  - **Polars**: () -> int
- `time`
  - **Polars**: (hour: Expr | str | int | None, minute: Expr | str | int | None, second: Expr | str | int | None, microsecond: Expr | str | int | None) -> Expr
- `time_range`
  - **Polars**: (start: time | IntoExprColumn | None, end: time | IntoExprColumn | None, interval: str | timedelta, closed: ClosedInterval, eager: bool) -> Series | Expr
- `time_ranges`
  - **Polars**: (start: time | IntoExprColumn | None, end: time | IntoExprColumn | None, interval: str | timedelta, closed: ClosedInterval, eager: bool) -> Series | Expr
- `union`
  - **Polars**: (items: Iterable[PolarsType], how: ConcatMethod, strict: bool) -> PolarsType
- `var`
  - **Polars**: (column: str, ddof: int) -> Expr
- `zeros`
  - **Polars**: (n: int | Expr, dtype: PolarsDataType, eager: bool) -> Expr | Series

### [!] Signature Mismatches (24)

- `Array`
  - **Polars**: (`inner: PolarsDataType | PythonDataType`, `shape: int | tuple[int, ...] | None`, `width: int | None`) -> None
  - **pql**: (`inner: DataType`, `size: int`) -> None
- `Datetime`
  - **Polars**: (`time_unit: TimeUnit`, `time_zone: str | tzinfo | None`) -> None
  - **pql**: (`time_unit: EpochTimeUnit`) -> None
- `Decimal`
  - **Polars**: (`precision: int | None`, scale: int) -> None
  - **pql**: (precision: int, scale: int) -> None
- `Duration`
  - **Polars**: (`time_unit: TimeUnit`) -> None
  - **pql**: () -> None
- `Enum`
  - **Polars**: (`categories: Series | Iterable[str] | type[enum.Enum]`) -> None
  - **pql**: (`categories: Iterable[str] | type[PyEnum]`) -> None
- `Expr`
  - **Polars**: ()
  - **pql**: (`_inner: SqlExpr`, `meta: ExprMeta`) -> None
- `LazyFrame`
  - **Polars**: (`data: FrameInitTypes | None`, `schema: SchemaDefinition | None`, `schema_overrides: SchemaDict | None`, `strict: bool`, `orient: Orientation | None`, `infer_schema_length: int | None`, `nan_to_null: bool`, `height: int | None`) -> None
  - **pql**: (`data: IntoRel`, orient: Orientation) -> None
- `List`
  - **Polars**: (`inner: PolarsDataType | PythonDataType`) -> None
  - **pql**: (`inner: DataType`) -> None
- `Struct`
  - **Polars**: (`fields: Sequence[Field] | SchemaDict`) -> None
  - **pql**: (`fields: IntoDict[str, DataType]`) -> None
- `all`
  - **Polars**: (`*names: str`, `ignore_nulls: bool`) -> Expr
  - **pql**: (`exclude: TryIter[IntoExprColumn]`) -> Expr
- `coalesce`
  - **Polars**: (exprs: IntoExpr | Iterable[IntoExpr], *more_exprs: IntoExpr, `eager: bool`) -> Expr | Series
  - **pql**: (exprs: TryIter[IntoExpr], *more_exprs: IntoExpr) -> Expr
- `from_dict`
  - **Polars**: (`data: Mapping[str, Sequence[object] | Mapping[str, Sequence[object]] | Series]`, `schema: SchemaDefinition | None`, `schema_overrides: SchemaDict | None`, `strict: bool`) -> DataFrame
  - **pql**: (`mapping: IntoDict[str, PythonLiteral]`) -> LazyFrame
- `from_dicts`
  - **Polars**: (`data: Iterable[Mapping[str, Any]]`, `schema: SchemaDefinition | None`, `schema_overrides: SchemaDict | None`, `strict: bool`, `infer_schema_length: int | None`) -> DataFrame
  - **pql**: (data: Sequence[Mapping[str, PythonLiteral]]) -> LazyFrame
- `from_numpy`
  - **Polars**: (`data: ndarray[Any, Any]`, `schema: SchemaDefinition | None`, `schema_overrides: SchemaDict | None`, `orient: Orientation | None`) -> DataFrame
  - **pql**: (`arr: AnyArray`, orient: Orientation) -> LazyFrame
- `from_records`
  - **Polars**: (`data: Sequence[Any]`, `schema: SchemaDefinition | None`, `schema_overrides: SchemaDict | None`, `strict: bool`, `orient: Orientation | None`, `infer_schema_length: int | None`) -> DataFrame
  - **pql**: (`data: SeqIntoVals`, orient: Orientation) -> LazyFrame
- `lit`
  - **Polars**: (value: Any, `dtype: PolarsDataType | DataTypeExpr | None`, `allow_object: bool`) -> Expr
  - **pql**: (value: PythonLiteral) -> Expr
- `max`
  - **Polars**: (`*names: str`) -> Expr
  - **pql**: (`cols: TryIter[str]`, *more_cols: str) -> Expr
- `mean`
  - **Polars**: (`*columns: str`) -> Expr
  - **pql**: (`cols: TryIter[str]`, *more_cols: str) -> Expr
- `mean_horizontal`
  - **Polars**: (*exprs: IntoExpr | Iterable[IntoExpr], `ignore_nulls: bool`) -> Expr
  - **pql**: (exprs: TryIter[IntoExpr], *more_exprs: IntoExpr) -> Expr
- `median`
  - **Polars**: (`*columns: str`) -> Expr
  - **pql**: (`cols: TryIter[str]`, *more_cols: str) -> Expr
- `min`
  - **Polars**: (`*names: str`) -> Expr
  - **pql**: (`cols: TryIter[str]`, *more_cols: str) -> Expr
- `sum`
  - **Polars**: (`*names: str`) -> Expr
  - **pql**: (`cols: TryIter[str]`, *more_cols: str) -> Expr
- `sum_horizontal`
  - **Polars**: (*exprs: IntoExpr | Iterable[IntoExpr], `ignore_nulls: bool`) -> Expr
  - **pql**: (exprs: TryIter[IntoExpr], *more_exprs: IntoExpr) -> Expr
- `when`
  - **Polars**: (*predicates: IntoExprColumn | Iterable[IntoExprColumn] | bool, `**constraints: Any`) -> When
  - **pql**: (`predicates: TryIter[IntoExpr]`, *more_predicates: IntoExpr) -> When

### [+] Extra Methods (pql-only) (13)

- `BitString`
- `DatetimeTZ`
- `Geometry`
- `Json`
- `Map`
- `Number`
- `TimeTZ`
- `UUID`
- `Union`
- `from_df`
- `from_query`
- `from_table`
- `from_table_function`

## selectors

### [x] Missing Methods (12)

- `alpha`
  - **Polars**: (ascii_only: bool, ignore_spaces: bool) -> Selector
- `alphanumeric`
  - **Polars**: (ascii_only: bool, ignore_spaces: bool) -> Selector
- `array`
  - **Polars**: (inner: Selector | None, width: int | None) -> Selector
- `by_index`
  - **Polars**: (*indices: int | range | Sequence[int | range], require_all: bool) -> Selector
- `datetime`
  - **Polars**: (time_unit: TimeUnit | Collection[TimeUnit] | None, time_zone: timezone | Collection[str | pydatetime.timezone | None] | None) -> Selector
- `digit`
  - **Polars**: (ascii_only: bool) -> Selector
- `exclude`
  - **Polars**: (columns: str | PolarsDataType | Selector | Expr | Collection[str | PolarsDataType | Selector | Expr], *more_columns: str | PolarsDataType | Selector | Expr) -> Selector
- `expand_selector`
  - **Polars**: (target: DataFrame | LazyFrame | Mapping[str, PolarsDataType], selector: Selector | Expr, strict: bool) -> tuple[str, ...]
- `first`
  - **Polars**: (strict: bool) -> Selector
- `is_selector`
  - **Polars**: (obj: Any) -> bool
- `last`
  - **Polars**: (strict: bool) -> Selector
- `list`
  - **Polars**: (inner: None | Selector) -> Selector

### [!] Signature Mismatches (4)

- `by_dtype`
  - **Polars**: (`*dtypes: PolarsDataType | PythonDataType | Iterable[PolarsDataType] | Iterable[PythonDataType]`) -> Selector
  - **pql**: (`*dtypes: type[dt.DataType]`) -> Selector
- `by_name`
  - **Polars**: (`*names: str | Collection[str]`, `require_all: bool`) -> Selector
  - **pql**: (*names: str) -> Selector
- `duration`
  - **Polars**: (`time_unit: TimeUnit | Collection[TimeUnit] | None`) -> Selector
  - **pql**: () -> Selector
- `string`
  - **Polars**: (`include_categorical: bool`) -> Selector
  - **pql**: () -> Selector

## DataType

### [x] Missing Methods (5)

- `base_type`
  - **Polars**: () -> type[Self]
- `from_python`
  - **Polars**: (py_type: PythonDataType) -> PolarsDataType
- `is_object`
  - **Polars**: () -> bool
- `to_dtype_expr`
  - **Polars**: () -> DataTypeExpr
- `to_python`
  - **Polars**: () -> PythonDataType

### [!] Signature Mismatches (1)

- `is_`
  - **Polars**: (`other: PolarsDataType`) -> bool
  - **pql**: (`other: T`) -> TypeIs[T]

### [+] Extra Methods (pql-only) (2)

- `from_duckdb`
- `from_sql`

## Schema

### [x] Missing Methods (19)

- `clear`
  - **Polars**: ()
- `contains_dtype`
  - **Polars**: (dtype: DataType, recursive: bool) -> bool
- `copy`
  - **Polars**: ()
- `dtypes`
  - **Polars**: () -> list[DataType]
- `fromkeys`
  - **Polars**: (iterable, value=...)
- `get`
  - **Polars**: (key, default=...)
- `items`
  - **Polars**: ()
- `keys`
  - **Polars**: ()
- `len`
  - **Polars**: () -> int
- `move_to_end`
  - **Polars**: (key, last=...)
- `names`
  - **Polars**: () -> list[str]
- `pop`
- `popitem`
  - **Polars**: (last=...)
- `setdefault`
  - **Polars**: (key, default=...)
- `to_arrow`
  - **Polars**: (compat_level: CompatLevel | None) -> Schema
- `to_frame`
  - **Polars**: (eager: bool) -> DataFrame | LazyFrame
- `to_python`
  - **Polars**: () -> dict[str, type]
- `update`
- `values`
  - **Polars**: ()
