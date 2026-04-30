from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Self

from pyochain import NONE, Dict, Iter, NoneOption as Null, Option, Seq, Set, Some

from .._utils import Builtins, Pql, get_attr
from ._parse import annotations_compatible, extract_last_name
from ._rules import IGNORED_PARAMS, Status

type MapInfo = Dict[str, ParamInfo]


@dataclass(slots=True)
class ParamInfo:
    """Information about a function parameter."""

    name: str
    is_var_positional: bool
    is_var_keyword: bool
    has_default: bool
    annotation: Option[str]

    @classmethod
    def from_signature(cls, param: inspect.Parameter) -> Self:
        """Create ParamInfo from inspect.Parameter.

        Args:
            param (inspect.Parameter): The parameter to create ParamInfo from.

        Returns:
            ParamInfo: The created ParamInfo instance.
        """
        return cls(
            name=param.name,
            is_var_positional=param.kind == inspect.Parameter.VAR_POSITIONAL,
            is_var_keyword=param.kind == inspect.Parameter.VAR_KEYWORD,
            has_default=param.default is not inspect.Parameter.empty,  # pyright: ignore[reportAny]
            annotation=_get_annotation_str(param.annotation),  # pyright: ignore[reportAny]
        )

    def param_name(self) -> str:
        match (self.is_var_positional, self.is_var_keyword):
            case (True, _):
                return f"*{self.name}"
            case (_, True):
                return f"**{self.name}"
            case _:
                return self.name


@dataclass(slots=True)
class MethodInfo:
    """Information about a method."""

    name: str
    params: Seq[ParamInfo]
    return_annotation: Option[str]
    is_property: bool = False

    @classmethod
    def from_signature(cls, name: str, sig: inspect.Signature) -> Self:
        """Create MethodInfo from inspect.Signature.

        Args:
            name (str): The name of the method.
            sig (inspect.Signature): The signature to create MethodInfo from.

        Returns:
            MethodInfo: The created MethodInfo instance.
        """
        return cls(
            name=name,
            params=Iter(sig.parameters.values())
            .map(ParamInfo.from_signature)
            .collect(),
            return_annotation=_get_annotation_str(sig.return_annotation),  # pyright: ignore[reportAny]
        )

    def signature_str(self, highlight_names: Option[Set[str]] = NONE) -> str:
        """Generate a human-readable signature string.

        Args:
            highlight_names (Option[Set[str]]): Optional set of parameter names to highlight.

        Returns:
            str: The formatted signature string.
        """
        highlights = highlight_names.unwrap_or_else(Set[str].new)
        params_str = (
            self.params
            .iter()
            .filter(lambda p: p.name != Builtins.SELF)
            .map(lambda p: _format_param_str(p, highlights))
            .join(", ")
        )
        ret = self.return_annotation.map(lambda r: f" -> {r}").unwrap_or("")
        return f"({params_str}){ret}"

    def to_map(self) -> MapInfo:
        """Convert parameters to a dictionary mapping names to ParamInfo.

        Returns:
            MapInfo: A dictionary mapping parameter names to ParamInfo instances.
        """
        return (
            self.params
            .iter()
            .filter(lambda p: p.name != Builtins.SELF)
            .map(lambda p: (p.name, p))
            .collect(Dict)
        )


@dataclass(slots=True)
class ComparisonInfos:
    """Holds MethodInfo for Polars and pql."""

    polars: Option[MethodInfo] = field(default_factory=lambda: NONE)
    pql_info: Option[MethodInfo] = field(default_factory=lambda: NONE)
    ignored_params: Set[str] = field(default_factory=Set[str].new)

    def has_reference(self) -> bool:
        return self.polars.is_some()

    def status(self) -> Option[Status]:
        match (self.polars, self.pql_info):
            case (Some(_), Null()):
                return Some(Status.MISSING)
            case (Null(), Some(_)):
                return Some(Status.EXTRA)
            case (Some(reference), Some(pql_info)):
                return Some(
                    Status.SIGNATURE_MISMATCH
                    if _mismatch_against(
                        pql_info.to_map(),
                        reference.to_map(),
                        self.ignored_params,
                    )
                    else Status.MATCH
                )
            case _:
                return NONE

    def to_status(self) -> Status:
        """Classify the method comparison result.

        Returns:
            Status: The status of the method comparison.
        """
        match (self.pql_info, self.polars):
            case (NONE, Some(_)):  # noqa: N806  # pyright: ignore[reportUnusedVariable]
                return Status.MISSING
            case (Some(_), NONE):  # noqa: F841, N806  # pyright: ignore[reportConstantRedefinition, reportUnusedVariable]
                return Status.EXTRA
            case (Some(target), Some(pl_info)):
                is_mismatch = _mismatch_against(
                    target.to_map(), pl_info.to_map(), self.ignored_params
                )
                match is_mismatch:
                    case True:
                        return Status.SIGNATURE_MISMATCH
                    case False:
                        return Status.MATCH
            case _:
                return Status.MISSING


def _mismatch_against(target: MapInfo, other: MapInfo, ignored: Set[str]) -> bool:
    target_filtered = _without_ignored_params(target, ignored)
    other_filtered = _without_ignored_params(other, ignored)
    on_params = (
        other_filtered.keys().symmetric_difference(target_filtered.keys()).length() > 0
    )
    on_ann = (
        other_filtered
        .keys()
        .intersection(target_filtered.keys())
        .any(
            lambda name: annotations_differ(
                other_filtered.get_item(name).unwrap(),
                target_filtered.get_item(name).unwrap(),
            )
        )
    )
    return on_params or on_ann


def ignored_params_for(class_name: Pql, method_name: str) -> Set[str]:
    return (
        IGNORED_PARAMS
        .get_item(class_name)
        .and_then(lambda method_map: method_map.get_item(method_name))
        .unwrap_or_else(Set.new)
    )


@dataclass(slots=True, init=False)
class ComparisonResult:
    """Result of comparing a single method."""

    method_name: str
    classification: Status
    infos: ComparisonInfos

    def __init__(
        self,
        polars_cls: object,
        pql_cls: object,
        method_name: str,
        class_name: Pql,
    ) -> None:
        """Compare a single method between Polars and pql."""
        infos = ComparisonInfos(
            polars=_get_method_info(polars_cls, method_name),
            pql_info=_get_method_info(pql_cls, method_name),
            ignored_params=ignored_params_for(class_name, method_name),
        )
        self.method_name = method_name
        self.classification = infos.to_status()
        self.infos = infos

    def to_format(self, *, status: Status) -> Iter[str]:
        """Format a single comparison result as markdown lines.

        Args:
            status (Status): The status to filter the results by.

        Returns:
            Iter[str]: An iterator over the formatted markdown lines.
        """
        match (status, self.infos.polars, self.infos.pql_info):
            case (Status.MISSING, _, _):
                return Iter.once(f"- `{self.method_name}`").chain(
                    self.infos.polars.map(
                        lambda info: Iter.once(
                            f"  - **Polars**: {info.signature_str()}"
                        )
                    ).unwrap_or(Iter(()))
                )
            case (Status.SIGNATURE_MISMATCH, Some(pl_info), Some(pql_info)):
                return Iter((
                    f"- `{self.method_name}`",
                    f"  - **Polars**: {_signature_with_diff(pl_info, pql_info, self.infos.ignored_params)}",
                    f"  - **pql**: {_signature_with_diff(pql_info, pl_info, self.infos.ignored_params)}",
                ))
            case _:
                return Iter.once(f"- `{self.method_name}`")


def _get_method_info(cls: object, name: str) -> Option[MethodInfo]:
    return get_attr(cls, name).and_then(_build_method_info, name)


def _build_method_info(attr: object, name: str) -> Option[MethodInfo]:
    match attr:
        case property() as prop:
            match prop.fget:
                case None:
                    return Some(
                        MethodInfo(name, Seq[ParamInfo].new(), NONE, is_property=True)
                    )
                case getter:
                    return Some(
                        MethodInfo(
                            name,
                            Seq[ParamInfo].new(),
                            _get_annotation_str(
                                inspect.signature(getter).return_annotation  # pyright: ignore[reportAny]
                            ),
                            is_property=True,
                        )
                    )
        case attr if callable(attr):
            try:
                return Some(
                    MethodInfo.from_signature(name=name, sig=inspect.signature(attr))
                )
            except (ValueError, TypeError):
                return NONE
        case _:
            return NONE


def _get_annotation_str(annotation: object) -> Option[str]:
    """Convert annotation to string representation.

    Args:
        annotation (object): The annotation to convert.

    Returns:
        Option[str]: The string representation of the annotation, or NONE if not available.
    """
    match annotation:
        case inspect.Parameter.empty | inspect.Signature.empty:
            return NONE
        case type():
            return Option(annotation.__name__)
        case _:
            return Option(extract_last_name(str(annotation)))


def _format_param_str(param: ParamInfo, highlight_names: Set[str]) -> str:
    rendered = param.annotation.map(lambda a: f"{param.param_name()}: {a}").unwrap_or(
        param.param_name() + ("=..." if param.has_default else "")
    )
    return rendered if not highlight_names.contains(param.name) else f"`{rendered}`"


def _signature_with_diff(base: MethodInfo, other: MethodInfo, ignored: Set[str]) -> str:
    return base.signature_str(Some(_diff_param_names(base, other, ignored)))


def _diff_param_names(
    base: MethodInfo, other: MethodInfo, ignored: Set[str]
) -> Set[str]:
    base_map = _without_ignored_params(base.to_map(), ignored)
    other_map = _without_ignored_params(other.to_map(), ignored)
    return (
        base_map
        .keys()
        .symmetric_difference(other_map.keys())
        .iter()
        .chain(
            base_map
            .keys()
            .intersection(other_map.keys())
            .iter()
            .filter(
                lambda name: annotations_differ(
                    base_map.get_item(name).unwrap(), other_map.get_item(name).unwrap()
                )
            )
        )
        .collect(Set)
    )


def _without_ignored_params(mapping: MapInfo, ignored: Set[str]) -> MapInfo:
    def _get_fn(current: Dict[str, ParamInfo], param: ParamInfo) -> ParamInfo:
        key = (
            param.name.removeprefix("more_")
            if param.is_var_positional and param.name.startswith("more_")
            else param.name
        )
        return current.setdefault(key, param)

    return (
        mapping
        .items()
        .iter()
        .filter_star(lambda k, _v: not ignored.contains(k))
        .map_star(lambda _name, param: param)
        .fold(
            Dict[str, ParamInfo].new(),
            lambda acc, param: acc.inspect(_get_fn, param),
        )
    )


def annotations_differ(pl_param: ParamInfo, pql_param: ParamInfo) -> bool:
    match (pl_param.annotation, pql_param.annotation):
        case (Some(pl_ann), Some(pql_ann)):
            return not annotations_compatible(pl_ann, pql_ann)
        case _:
            return False
