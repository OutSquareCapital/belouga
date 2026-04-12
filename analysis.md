# full brainstorming analysis

---

## Status quo

- **`LazyFrame`** wraps `ScanSource` = `(duckdb.DuckDBPyRelation, pc.Vec[str])`.
- Every transformation (`select`, `filter`, `sort`, `join`, ...) calls `self.inner().relation.<method>()` immediately → produces a new DuckDB relation → wraps it in a new `ScanSource`.
- **`SqlExpr`** already wraps `exp.Expr` (sqlglot AST) internally. Conversion to `duckdb.Expression` only happens at `.into_duckdb()`.
- **Pivot/unpivot/asof join** already build sqlglot ASTs and materialize via `_from_sql_expr()` → `ScanSource.from_query(ast.sql(dialect="duckdb"))`. The pattern already exists.
- **`ExprPlan`** resolves expression metadata (multi-col selectors, aliasing, distinct/agg detection) and produces `DuckDBPyRelation` via `select_context`, `with_columns_context`, `agg_context`, `group_by_all_context`.

**23 direct `self.inner().relation` access points** in _frame.py, plus_groupby.py's aggregator.

---

## Option A: Full sqlglot AST (SQLFrame-like CTE chains)

### Concept

`LazyFrame._inner` becomes a `QueryPlan` storing:

```python
@dataclass
class QueryPlan:
    base: ScanSource  # materialized base data (dict/numpy/scan)
    ast: exp.Select  # the full query tree built on top
    tracked_cols: pc.Vec[str]  # column names tracked through transforms
```

Every transformation mutates the AST:

- `select()` → `exp.select(*resolved_exprs).from_(cte_alias)`
- `filter()` → `ast.where(predicate)`
- `sort()` → `ast.order_by(*ordered_exprs)`
- `limit(n)` → `ast.limit(n)`
- `with_columns()` → SELECT existing + new FROM current
- `group_by().agg()` → SELECT aggs FROM current GROUP BY keys
- `join()` → `ast.join(other_alias, on=condition, join_type=...)`
- `union()` → `exp.Union(this=self.ast, expression=other.ast)`

Each chaining step wraps the previous query as a CTE (like SQLFrame), ensuring composability.

Terminal methods (`collect`, `lazy`, `columns`, `dtypes`, `shape`, `explain`, `show`, `sink_*`, `fetch_all`):

1. Register base relation(s) as DuckDB temp views
2. Generate SQL: `ast.sql(dialect="duckdb")`
3. `duckdb.from_query(sql)` or `duckdb.execute(sql)`
4. Return result

### Pros

- Full SQL introspectability at any point in the chain
- sqlglot optimizer can merge projections, push predicates, eliminate dead columns
- Portable foundation: switching backends (Postgres, Spark, SQLite) becomes possible
- Serializable: the AST can be stored/sent/replayed
- Naturally composable: CTEs are DuckDB's bread and butter
- Eliminates reliance on DuckDB's Python relation API quirks/bugs
- `SqlExpr` already builds sqlglot AST internally → no double conversion

### Cons

- **Column tracking is manual**: Currently `relation.columns` handles it. With AST we must propagate `tracked_cols` at every step. For `select()`, `with_columns()`, `join()`, `explode()` this is non-trivial — especially for multi-column selectors (`*`, `cs.numeric()`, etc.) that resolve against known columns.
- **ExprPlan rewrite**: The 4 context methods (`select_context`, `with_columns_context`, `agg_context`, `group_by_all_context`) must produce AST nodes instead of `DuckDBPyRelation`. This is the single hardest migration point: the distinct/aggregation/broadcast logic is sophisticated.
- **CTE overhead**: Each step = 1 CTE. Long chains generate verbose SQL. DuckDB's optimizer handles this well perf-wise, but explain output and `sql_query()` become noisy (sqlglot optimizer can help).
- **DuckDB-specific terminal features**: `describe()`, `shape` (needs row count) → require materialization anyway, but currently they're just `relation.describe()` / `relation.shape`. Post-refactor they need a `from_query` round-trip.
- **Registration side effects**: Base data must exist as a named DuckDB table/view. Need a mechanism to register temp views and manage their lifecycle.
- **Largest refactor scope**: Touches _frame.py,_meta.py, _groupby.py, `_joins.py`, `_scans.py`, and possibly `selectors.py`.

---

## Option B: Hybrid — AST layer on top of materialized base

### Concept

`ScanSource` stays as-is for base data. LazyFrame carries both:

```python
@dataclass
class QueryPlan:
    base: ScanSource  # materialized base (always available)
    transforms: pc.Vec[ASTTransform]  # pending AST operations
```

Where `ASTTransform` is a discriminated union of lightweight ops:

```python
type ASTTransform = SelectAST | FilterAST | SortAST | LimitAST | ...
```

On terminal methods, the transforms are **compiled** into a single sqlglot AST referencing the base relation, then executed via `from_query`.

On methods that NEED intermediate column info (e.g. `with_columns` needs current columns to resolve `*`), we have two sub-strategies:

- **B1: Track columns eagerly** — each transform op also updates a `tracked_cols` field, purely from AST analysis
- **B2: Materialize on demand** — if column info is needed mid-chain, flush pending transforms to create a new base relation, then continue

### Pros

- Smaller refactor surface than Option A: transformation methods just append to a list
- B2 variant is pragmatic — complex ops that need column info (explode, selectors) can materialize
- Base data initialization stays untouched
- Incremental adoption: migrate methods one by one, keeping the "flush to relation" escape hatch

### Cons

- **B2 breaks the "defer everything" goal**: If `with_columns` frequently needs columns → frequent materialization → gains are marginal
- **B1 requires the same column tracking work as Option A** without the elegance of a uniform AST
- Two representations to maintain (relation + transform list), more complex reasoning about state
- No sqlglot optimizer benefits: the AST is only constructed at the end, not available for mid-chain analysis
- Not serializable mid-chain since it depends on a live DuckDB base relation

---

## Option C: Thin AST wrapper — current ops emit sqlglot instead of DuckDB calls

### Concept

Keep `ScanSource` and the current architecture, but replace every `self.inner().relation.<method>()` call with its sqlglot equivalent:

- `relation.select(*exprs)` → build `exp.select(*glot_exprs).from_(current_cte)`
- `relation.filter(expr)` → `current.where(glot_expr)`
- `relation.sort(*exprs)` → `current.order_by(*glot_exprs)`
- `relation.limit(n)` → `current.limit(n)`
- etc.

`ScanSource` becomes:

```python
@dataclass
class ScanSource:
    base_relation: duckdb.DuckDBPyRelation  # the raw data
    base_alias: str  # registered name
    query: exp.Select  # the growing AST
    columns: pc.Vec[str]  # tracked names
```

`ExprPlan` context methods return `exp.Select` instead of `DuckDBPyRelation`.

### Pros

- Method-by-method migration: each method can be converted independently
- Natural mapping: `relation.filter(expr)` → `query.where(expr)` is 1:1
- Existing CTE approach for pivot/unpivot/asof already proves the pattern
- `_from_sql_expr` pattern generalizes naturally

### Cons

- Still need manual column tracking (same as Option A)
- ExprPlan rewrite needed (same as Option A, but can be done incrementally)
- Base relation needs registration as temp view (same as Option A)
- Essentially becomes Option A when fully implemented — just a different migration path

---

## Option D: LazyFrame stores `exp.Select`, `ScanSource.from_query()` at terminal only

### Concept

This is the most targeted version of what you described. `LazyFrame._inner` changes to:

```python
@dataclass
class LazyPlan:
    sources: pc.Dict[str, ScanSource]  # named base relations
    ast: exp.Select  # the full query
    columns: pc.Vec[str]  # tracked column names
```

- `sources` holds all base data referenced by the query (main table + joined tables)
- `ast` is the full sqlglot query tree
- `columns` is tracked through transforms

Terminal methods:

```python
def _materialize(self) -> ScanSource:
    # Register all sources as temp views/aliases
    # Generate SQL
    # ScanSource.from_query(self.ast.sql(dialect="duckdb"), **self.sources)
```

This is essentially Option A with explicit source tracking, making `ScanSource.from_query` the single execution pathway.

### Pros

- Clean single-path execution model
- `ScanSource.from_query` already exists and handles relation binding
- `sources` dict makes join scenarios explicit
- Most aligned with your stated goal

### Cons

- Same column tracking burden as all AST-based approaches
- Same ExprPlan rewrite requirement
- Multiple sources management adds complexity for joins (self-joins, multi-way joins)

---

## Cross-cutting concerns (all options)

### 1. Column tracking — the crux

Every AST-based approach needs to answer: "what are the current columns?"

| Operation | Column impact |
|-----------|--------------|
| `select()` | Replaces: output = resolved expr names |
| `with_columns()` | Extends/replaces: existing + new, dedup by name |
| `filter()` | Passthrough |
| `sort()` | Passthrough |
| `limit()/head()/tail()` | Passthrough |
| `drop()` | Removes named columns |
| `rename()` | Remaps names |
| `join()` | Merge of both sides with suffix handling |
| `explode()` | Passthrough (names stay, types change) |
| `pivot()` | Completely new set of columns (data-dependent!) |
| `describe()` | Fixed schema |
| `union()` | Left side wins |

**Pivot** is the hardest: the output columns depend on **data values**, not just the schema. This fundamentally requires a materialization step (or a pre-query to discover values). Current code already requires `on_columns` to be explicit, so this is manageable.

**Multi-column selectors** (`cs.all()`, regex selectors, `*`) require knowing all current columns. This is solvable since `tracked_cols` propagates.

### 2. ExprPlan rewrite strategy

ExprPlan currently outputs `DuckDBPyRelation`. It needs to output `exp.Select` (or AST fragments).

The logic inside `select_context` is:

- All distinct? → `.select(*exprs).distinct()`  
- All pure reducers? → `.aggregate(*exprs)` (= `SELECT agg FROM ... GROUP BY ALL`)
- Mixed? → `.select(*exprs)` with broadcast reducers windowed

This translates naturally:

- `.select(*exprs)` → `exp.select(*glot_exprs).from_(current_cte)`
- `.aggregate(*exprs)` → same but with no GROUP BY (DuckDB auto-groups)
- `.distinct()` → `current.distinct()`
- Broadcast reducers → already handled by `_broadcast_reducers` which adds `OVER()`, which is pure sqlglot AST manipulation

### 3. LazyGroupBy rewrite

`_aggregator = partial(frame.inner().relation.aggregate, group_expr=...)` → becomes `lambda exprs: exp.select(*exprs).from_(base).group_by(...)`.

### 4. `Marker.windowed()`

Currently: `lf.select(row_nb, sql.all().into_duckdb())` on the relation. Would become: wrapping the current AST as a subquery with the row_number prepended.

---

## Synthesis & Recommendation

Options A, C, and D converge to essentially the same end state:
> LazyFrame stores a sqlglot AST + column names. Terminal methods generate SQL and call `duckdb.from_query()`.

The difference is **migration strategy**:

- **Option A/D** = big-bang rewrite
- **Option C** = incremental method-by-method migration  
- **Option B** = incremental but with a "hybrid" state that adds complexity without payoff

**My recommendation: Option C (incremental) towards the end-state of D.**

Concretely:

**Phase 1 — Foundation (low risk)**

1. Add `columns: pc.Vec[str]` tracking to a new `QueryPlan` dataclass alongside the existing relation
2. `ScanSource.build()` populates both relation and columns (already does)
3. Simple passthrough methods (`filter`, `sort`, `limit`) emit sqlglot and go through `from_query` instead of `relation.<method>()`
4. Keep the `relation` as fallback for methods not yet migrated

**Phase 2 — ExprPlan migration (medium risk)**

1. Add `select_context_ast()` etc. that return `exp.Select` instead of `DuckDBPyRelation`
2. Migrate `select()`, `with_columns()`, `group_by_all()` to use AST path
3. Column tracking gets exercised extensively here

**Phase 3 — Complex operations (higher risk)**

1. Migrate `join()`, `join_cross()`, `join_asof()` (asof already uses AST)
2. Migrate `explode()`, `unique()`, `pivot()`, `unpivot()` (pivot/unpivot already use AST)
3. Migrate `_iter_agg`, `_iter_slct` patterns

**Phase 4 — Cleanup**

1. Remove `duckdb.DuckDBPyRelation` from `ScanSource` for non-base usage
2. `ScanSource` becomes the base-only materialization point
3. Terminal methods unify through `from_query`

This gives you the ability to **migrate and test incrementally**, with each phase leaving the test suite green.

Completed: *Synthesize and compare* (5/5)

Want me to dig deeper into any specific option, or start prototyping Phase 1?
