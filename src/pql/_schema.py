from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from pyochain import Dict, Vec
from pyochain.traits import PyoMutableMapping

from ._datatypes import DataType

if TYPE_CHECKING:
    from .sql.typing import IntoDict


@dataclass(slots=True, init=False, repr=False)
class Schema(PyoMutableMapping[str, DataType]):  # noqa: PLW1641
    _keys: Vec[str]
    _dtypes: Vec[DataType]
    _data: Dict[str, DataType]

    def __init__(self, data: IntoDict[str, DataType]) -> None:
        self._data = Dict(data)
        pairs = self._data.items().iter().unzip()
        self._keys = pairs.left.collect(Vec)
        self._dtypes = pairs.right.collect(Vec)

    @override
    def __eq__(self, other: object) -> bool:
        match other:
            case Schema():
                return self._data == other._data
            case Mapping():
                return self._data == other
            case _:
                return NotImplemented

    @override
    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    @override
    def __len__(self) -> int:
        return len(self._keys)

    @override
    def __getitem__(self, key: str) -> DataType:
        return self._data[key]

    @override
    def __setitem__(self, key: str, value: DataType) -> None:
        match key in self._data:
            case True:
                self._data[key] = value
                self._dtypes[self._keys.index(key)] = value
            case False:
                self._data[key] = value
                self._keys.append(key)
                self._dtypes.append(value)

    @override
    def __delitem__(self, key: str) -> None:
        pos = self._keys.index(key)
        _ = self._keys.pop(pos)
        _ = self._dtypes.pop(pos)
        del self._data[key]

    @override
    def keys(self) -> Vec[str]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self._keys

    @override
    def values(self) -> Vec[DataType]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self._dtypes

    def insert_at(self, pos: int, name: str, dtype: DataType) -> None:
        """Insert a key-value pair at a specific position."""
        self._keys.insert(pos, name)
        self._dtypes.insert(pos, dtype)
        self._data[name] = dtype
