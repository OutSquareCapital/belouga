# Refactor LazyFrame → sqlglot AST

## Status quo

- `LazyFrame` wraps `ScanSource` = `(duckdb.DuckDBPyRelation, pc.Vec[str])`.
- Every transformation (`select`, `filter`, `sort`, `join`, ...) calls `self.inner().relation.<method>()` immediatement → produit une nouvelle DuckDB relation → wrap dans un nouveau `ScanSource`.
- `SqlExpr` wraps déjà `exp.Expr` (sqlglot AST). La conversion vers `duckdb.Expression` ne se fait qu'au `.into_duckdb()`.
- **Pivot/unpivot/asof join** construisent déjà des ASTs sqlglot et matérialisent via `_from_sql_expr()` → `ScanSource.from_query(ast.sql(dialect="duckdb"))`. Le pattern existe déjà.
- `ExprPlan` résout les métadonnées d'expression et produit `DuckDBPyRelation` via `select_context`, `with_columns_context`, `agg_context`, `group_by_all_context`.

**23 accès directs `self.inner().relation`** dans `_frame.py`, plus l'agrégateur de `_groupby.py`.

---

## Objectif

`LazyFrame` transporte un AST sqlglot (`exp.Select`) au lieu de déléguer chaque opération à `DuckDBPyRelation`. La relation DuckDB n'est créée qu'aux terminaux (`collect`, `lazy`, `dtypes`, `shape`, `explain`, `show`, `sink_*`, `fetch_all`) via `ScanSource.from_query`.

`ScanSource` reste inchangé : il gère la relation de base et les colonnes. C'est `LazyFrame` qui porte l'arbre.

---

## Architecture cible

### ScanSource (gère relation + colonnes)

```python
@dataclass(slots=True)
class ScanSource:
    relation: duckdb.DuckDBPyRelation   # source de données brute
    columns: pc.Vec[str]                 # noms de colonnes, mutable (Vec), muté in-place par les ops
```

Rôle : stocker la relation de base et les colonnes. `columns` est un `Vec` (mutable) — les transformations LazyFrame le mutent in-place, pas de recréation de `ScanSource` à chaque opération. C'est aussi le point de matérialisation via `from_query`.

### LazyFrame (porte l'AST)

```python
@dataclass(slots=True, init=False, repr=False)
class LazyFrame(sql.CoreHandler[ScanSource]):
    _inner: ScanSource      # relation de base + colonnes mutables
    _ast: exp.Select         # arbre SQL construit par les transformations
```

- `_inner` = le `ScanSource`. Contient la relation de base et les colonnes. Les colonnes sont mutées in-place par chaque transformation (le `Vec` est mutable, c'est pour ça qu'il est là).
- `_ast` = l'arbre sqlglot qui grandit à chaque transformation. Initialisé à `SELECT * FROM base_alias`.
- Les colonnes s'accèdent via `self.inner().columns` — exactement comme aujourd'hui.

Matérialisation = `ScanSource.from_query(self._ast.sql(dialect="duckdb"), base=self.inner().relation)`.

---

## Mapping des transformations

Chaque méthode LazyFrame qui fait aujourd'hui `self.inner().relation.<method>()` manipule `_ast` à la place :

| Méthode actuelle | Pattern actuel | Pattern cible |
|---|---|---|
| `filter(expr)` | `relation.filter(expr.into_duckdb())` | `_ast.where(expr.inner())` |
| `sort(*exprs)` | `relation.sort(*exprs)` | `_ast.order_by(*glot_exprs)` |
| `limit(n)` | `relation.limit(n)` | `_ast.limit(n)` |
| `select(*exprs)` | `ExprPlan.select_context(relation)` | `ExprPlan.select_context_ast(_ast)` |
| `with_columns(...)` | `ExprPlan.with_columns_context(relation)` | `ExprPlan.with_columns_context_ast(_ast)` |
| `group_by_all(...)` | `ExprPlan.group_by_all_context(relation)` | `exp.select(*aggs).from_(_ast_as_cte)` |
| `drop(*cols)` | `relation.select(all_except)` | `exp.select(*remaining).from_(_ast)` |
| `union(other)` | `relation.union(other.relation)` | `exp.Union(this=self._ast, expression=other._ast)` |
| `join(other, ...)` | `relation.join(other.relation, ...)` | `_ast.join(other_alias, on=condition, join_type=...)` |
| `rename(mapping)` | `_iter_slct(alias logic)` | `exp.select(*aliased_cols).from_(_ast)` |

Les terminaux matérialisent l'AST :

| Terminal | Implémentation |
|---|---|
| `collect()` | `_materialize().relation.pl()` |
| `lazy()` | `_materialize().relation.pl(lazy=True)` |
| `columns` | retourne `self.inner().columns` (tracké dans ScanSource, pas de matérialisation) |
| `dtypes` | matérialisation nécessaire |
| `schema` | = `self.inner().columns` + `dtypes` |
| `shape` | matérialisation nécessaire |
| `explain()` | `_materialize().relation.explain()` |
| `show()` | `_materialize().relation.show()` |
| `sink_*()` | `_materialize().relation.write_*()` |
| `fetch_all()` | `_materialize().relation.fetchall()` |
| `sql_query()` | `_ast.sql(dialect="duckdb")` directement |
| `describe()` | matérialisation nécessaire |

Avec `_materialize()` = `ScanSource.from_query(self._ast.sql(dialect="duckdb"), base=self.inner().relation)`.

---

## Column tracking (mutation in-place du Vec)

Chaque transformation LazyFrame mute `self.inner().columns` in-place (c'est un `pc.Vec`, mutable). Pas de recréation de `ScanSource`.

| Opération | Impact sur `ScanSource.columns` |
|-----------|-------------------------------|
| `select()` | Remplace : output = noms des expressions résolues |
| `with_columns()` | Étend/remplace : existant + nouveau, dédup par nom |
| `filter()` | Passthrough |
| `sort()` | Passthrough |
| `limit()/head()/tail()` | Passthrough |
| `drop()` | Supprime les colonnes nommées |
| `rename()` | Remap des noms |
| `join()` | Merge des deux côtés avec gestion suffix |
| `explode()` | Passthrough (noms restent, types changent) |
| `pivot()` | Nouveau set (data-dependent mais `on_columns` est explicite) |
| `union()` | Left side wins |

La seule différence avec aujourd'hui : on ne peut plus faire `relation.columns` pour vérifier — on se fie au tracking dans `ScanSource.columns`. C'est correct tant que le tracking est fidèle.

Multi-column selectors (`cs.all()`, `*`) résolvent contre `self.inner().columns` : déjà le cas via `ExprPlan`.

---

## ExprPlan : rewrite des context methods

`ExprPlan` produit actuellement `DuckDBPyRelation`. Doit produire `exp.Select` (ou fragments AST).

La logique de `select_context` :

- All distinct? → `exp.select(*exprs).from_(current).distinct()`
- All pure reducers? → `exp.select(*aggs).from_(current)` (DuckDB auto-groups)
- Mixed? → `exp.select(*exprs_with_broadcast).from_(current)`

`_broadcast_reducers` ajoute `OVER()` — c'est déjà de la manipulation AST sqlglot pure.

`with_columns_context` → `exp.select(*existing_plus_new).from_(current)`

`agg_context` → `exp.select(*keys, *aggs).from_(current).group_by(*keys_str)`

`group_by_all_context` → `exp.select(*aggs).from_(current)` + `GROUP BY ALL`

`Marker.windowed()` → wrap l'AST courant en sous-requête avec `row_number()` prepended.

---

## LazyGroupBy : rewrite

Actuellement : `_aggregator = partial(frame.inner().relation.aggregate, group_expr=...)`.
Cible : `lambda exprs: exp.select(*exprs).from_(base_cte).group_by(...)`.

Les `keys` sont déjà des `SqlExpr` (= sqlglot AST). L'aggregateur reçoit des expressions sqlglot au lieu de `duckdb.Expression`.

---

## Gestion des sources pour les joins

Un join implique deux `ScanSource` distincts. La source `relation` du right side doit être accessible lors de la matérialisation.

Approche : au moment du join, le LazyFrame résultant accumule les références aux sources des deux côtés. `ScanSource.from_query` accepte déjà `**relations: IntoRel` — on passe les relations nommées.

Pour les self-joins : aliasing via `set_alias` (déjà utilisé aujourd'hui).

---

## Cas spéciaux

### `_from_sql_expr` (pivot/unpivot/asof)

Déjà le pattern cible : construit un AST sqlglot, passe à `from_query`. Pas de changement nécessaire — ces méthodes deviennent juste le cas standard au lieu de l'exception.

### `_iter_agg` / `_iter_slct`

Ces helpers itèrent sur `self.inner().columns` et construisent des expressions. Actuellement appellent `relation.aggregate` / `select`. Cible : construisent un `exp.Select` à la place.

### `describe()` / `shape`

Nécessitent une vraie relation DuckDB. Matérialisent via `_materialize()` puis délèguent.

---

## Migration incrémentale

Chaque phase laisse la test suite verte.

**Phase 1 — Foundation**

1. Ajouter `_ast: exp.Select` à `LazyFrame`. Init à `SELECT * FROM base_alias`. Les colonnes restent dans `ScanSource.columns` comme aujourd'hui.
2. `ScanSource` reste inchangé.
3. Migrer les passthrough simples : `filter`, `sort`, `limit` → manipulent `_ast`. Les colonnes ne changent pas (passthrough), donc pas de mutation du Vec.
4. Les terminaux matérialisent via `_materialize()` → `ScanSource.from_query(_ast.sql(...))`.

**Phase 2 — ExprPlan**

1. Ajouter des variantes `*_context_ast()` qui retournent `exp.Select` au lieu de `DuckDBPyRelation`.
2. Migrer `select()`, `with_columns()`, `group_by_all()` vers le path AST.
3. Le column tracking est exercé à fond ici.

**Phase 3 — Opérations complexes**

1. Migrer `join()`, `join_cross()`, `join_asof()` (asof utilise déjà l'AST).
2. Migrer `explode()`, `unique()`, `pivot()`, `unpivot()` (pivot/unpivot utilisent déjà l'AST).
3. Migrer `_iter_agg`, `_iter_slct`, `LazyGroupBy`.

**Phase 4 — Cleanup**

1. `ScanSource` ne sert plus qu'à stocker la source de base et à matérialiser via `from_query`.
2. Tous les terminaux passent par `_materialize()`.
3. Supprimer `glot_into_duckdb` et les chemins de conversion `SqlExpr → duckdb.Expression` pour les contextes LazyFrame (reste nécessaire pour `ScanSource.from_query` callback interne).
