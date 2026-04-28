from __future__ import annotations

"""Portable model specification helpers for pymars."""

import copy
import json
import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
from sklearn.preprocessing import LabelEncoder

from ._basis import (
    BasisFunction,
    CategoricalBasisFunction,
    ConstantBasisFunction,
    HingeBasisFunction,
    InteractionBasisFunction,
    LinearBasisFunction,
    MissingnessBasisFunction,
)
from ._categorical import CategoricalImputer

MODEL_SPEC_VERSION = "1.0"
_MODEL_SPEC_VERSION_RE = re.compile(r"^(?P<major>\d+)\.(?P<minor>\d+)$")
_MODEL_SPEC_SUPPORTED_MAJOR_VERSION = 1


def _parse_model_spec_version(spec_version: Any) -> tuple[int, int]:
    """Parse a portable model spec version string."""
    if not isinstance(spec_version, str):
        raise ValueError(
            "Model spec field 'spec_version' must be a '<major>.<minor>' string."
        )

    match = _MODEL_SPEC_VERSION_RE.fullmatch(spec_version)
    if match is None:
        raise ValueError(
            "Model spec field 'spec_version' must be a '<major>.<minor>' string."
        )

    return int(match.group("major")), int(match.group("minor"))


def validate_model_spec(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a portable model spec payload."""
    if not isinstance(payload, dict):
        raise ValueError("Model spec payload must be a JSON object.")

    spec_version = payload.get("spec_version")
    major_version, _minor_version = _parse_model_spec_version(spec_version)
    if major_version != _MODEL_SPEC_SUPPORTED_MAJOR_VERSION:
        raise ValueError(
            "Unsupported model spec version "
            f"{spec_version!r}; expected a compatible "
            f"'{_MODEL_SPEC_SUPPORTED_MAJOR_VERSION}.x' payload."
        )

    required_fields = ("params", "feature_schema", "basis_terms", "coefficients")
    missing_fields = [field for field in required_fields if field not in payload]
    if missing_fields:
        raise ValueError(
            "Model spec is missing required fields: "
            + ", ".join(sorted(missing_fields))
        )

    if not isinstance(payload["params"], dict):
        raise ValueError("Model spec field 'params' must be an object.")
    if not isinstance(payload["feature_schema"], dict):
        raise ValueError("Model spec field 'feature_schema' must be an object.")
    if not isinstance(payload["basis_terms"], list):
        raise ValueError("Model spec field 'basis_terms' must be an array.")
    if not isinstance(payload["coefficients"], list):
        raise ValueError("Model spec field 'coefficients' must be an array.")

    feature_schema = payload["feature_schema"]
    n_features = feature_schema.get("n_features")
    if n_features is not None and (not isinstance(n_features, int) or n_features < 0):
        raise ValueError(
            "Model spec field 'feature_schema.n_features' must be a non-negative integer or null."
        )

    if len(payload["coefficients"]) != len(payload["basis_terms"]):
        raise ValueError(
            "Model spec must contain one coefficient per basis term."
        )

    for idx, term in enumerate(payload["basis_terms"]):
        if not isinstance(term, dict):
            raise ValueError(f"Basis term at index {idx} must be an object.")
        kind = term.get("kind")
        if not isinstance(kind, str) or not kind:
            raise ValueError(
                f"Basis term at index {idx} is missing a valid 'kind' field."
            )

    return payload


@dataclass(frozen=True)
class BasisTermSpec:
    """Portable representation of a fitted basis function."""

    kind: str
    variable_idx: int | None = None
    variable_name: str | None = None
    knot_val: float | None = None
    is_right_hinge: bool | None = None
    category: Any | None = None
    gcv_score: float = 0.0
    rss_score: float = 0.0
    parent1: BasisTermSpec | None = None
    parent2: BasisTermSpec | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the basis term spec to a dictionary."""
        payload = {
            "kind": self.kind,
            "variable_idx": self.variable_idx,
            "variable_name": self.variable_name,
            "knot_val": self.knot_val,
            "is_right_hinge": self.is_right_hinge,
            "category": self.category,
            "gcv_score": self.gcv_score,
            "rss_score": self.rss_score,
            "parent1": self.parent1.to_dict() if self.parent1 is not None else None,
            "parent2": self.parent2.to_dict() if self.parent2 is not None else None,
        }
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BasisTermSpec:
        """Build a basis term spec from a dictionary payload."""
        return cls(
            kind=payload["kind"],
            variable_idx=payload.get("variable_idx"),
            variable_name=payload.get("variable_name"),
            knot_val=payload.get("knot_val"),
            is_right_hinge=payload.get("is_right_hinge"),
            category=payload.get("category"),
            gcv_score=float(payload.get("gcv_score", 0.0)),
            rss_score=float(payload.get("rss_score", 0.0)),
            parent1=cls.from_dict(payload["parent1"])
            if payload.get("parent1") is not None
            else None,
            parent2=cls.from_dict(payload["parent2"])
            if payload.get("parent2") is not None
            else None,
        )


def basis_function_to_spec(basis_function: BasisFunction) -> BasisTermSpec:
    """Convert a fitted basis function to a portable spec."""
    if isinstance(basis_function, ConstantBasisFunction):
        kind = "constant"
    elif isinstance(basis_function, HingeBasisFunction):
        kind = "hinge"
    elif isinstance(basis_function, LinearBasisFunction):
        kind = "linear"
    elif isinstance(basis_function, CategoricalBasisFunction):
        kind = "categorical"
    elif isinstance(basis_function, InteractionBasisFunction):
        kind = "interaction"
    elif isinstance(basis_function, MissingnessBasisFunction):
        kind = "missingness"
    else:  # pragma: no cover - defensive branch for unknown basis classes
        raise TypeError(f"Unsupported basis function type: {type(basis_function)!r}")

    return BasisTermSpec(
        kind=kind,
        variable_idx=basis_function.variable_idx,
        variable_name=getattr(basis_function, "variable_name", None),
        knot_val=basis_function.knot_val,
        is_right_hinge=getattr(basis_function, "is_right_hinge", None),
        category=getattr(basis_function, "category", None),
        gcv_score=float(getattr(basis_function, "gcv_score_", 0.0)),
        rss_score=float(getattr(basis_function, "rss_score_", 0.0)),
        parent1=basis_function_to_spec(basis_function.parent1)
        if basis_function.parent1 is not None
        else None,
        parent2=basis_function_to_spec(basis_function.parent2)
        if basis_function.parent2 is not None
        else None,
    )


def basis_function_from_spec(term_spec: BasisTermSpec) -> BasisFunction:
    """Rebuild a basis function from its portable spec."""
    parent1 = (
        basis_function_from_spec(term_spec.parent1)
        if term_spec.parent1 is not None
        else None
    )
    parent2 = (
        basis_function_from_spec(term_spec.parent2)
        if term_spec.parent2 is not None
        else None
    )

    if term_spec.kind == "constant":
        basis_function = ConstantBasisFunction()
    elif term_spec.kind == "hinge":
        basis_function = HingeBasisFunction(
            variable_idx=cast(int, term_spec.variable_idx),
            knot_val=cast(float, term_spec.knot_val),
            is_right_hinge=bool(term_spec.is_right_hinge),
            variable_name=term_spec.variable_name,
            parent_bf=parent1,
        )
    elif term_spec.kind == "linear":
        basis_function = LinearBasisFunction(
            variable_idx=cast(int, term_spec.variable_idx),
            variable_name=term_spec.variable_name,
            parent_bf=parent1,
        )
    elif term_spec.kind == "categorical":
        basis_function = CategoricalBasisFunction(
            variable_idx=cast(int, term_spec.variable_idx),
            category=term_spec.category,
            variable_name=term_spec.variable_name,
            parent_bf=parent1,
        )
    elif term_spec.kind == "interaction":
        if parent1 is None or parent2 is None:
            raise ValueError("Interaction basis terms require two parent terms.")
        basis_function = InteractionBasisFunction(parent1, parent2)
    elif term_spec.kind == "missingness":
        basis_function = MissingnessBasisFunction(
            variable_idx=cast(int, term_spec.variable_idx),
            variable_name=term_spec.variable_name,
        )
    else:  # pragma: no cover - defensive branch for invalid spec payloads
        raise ValueError(f"Unsupported basis term kind: {term_spec.kind!r}")

    basis_function.gcv_score_ = term_spec.gcv_score
    basis_function.rss_score_ = term_spec.rss_score
    return basis_function


def categorical_imputer_to_spec(imputer: CategoricalImputer | None) -> dict[str, Any] | None:
    """Serialize a categorical imputer to a dictionary."""
    if imputer is None:
        return None

    encoders = {
        str(idx): list(encoder.classes_)
        for idx, encoder in imputer.encoders.items()
    }
    return {
        "encoders": encoders,
        "most_frequent": {
            str(idx): value for idx, value in imputer.most_frequent_.items()
        },
    }


def categorical_imputer_from_spec(payload: dict[str, Any] | None) -> CategoricalImputer | None:
    """Rebuild a categorical imputer from its portable spec."""
    if payload is None:
        return None

    imputer = CategoricalImputer()
    for idx_str, classes in payload.get("encoders", {}).items():
        idx = int(idx_str)
        encoder = imputer.encoders.setdefault(idx, LabelEncoder())
        encoder.classes_ = np.asarray(classes, dtype=object)
    for idx_str, value in payload.get("most_frequent", {}).items():
        imputer.most_frequent_[int(idx_str)] = value
    return imputer


def model_to_spec(model: Any) -> dict[str, Any]:
    """Export a fitted Earth-like model into a portable dictionary spec."""
    if not getattr(model, "fitted_", False):
        raise ValueError("Model must be fitted before exporting a portable spec.")
    if model.basis_ is None or model.coef_ is None:
        raise ValueError("Fitted model is missing basis functions or coefficients.")

    record = getattr(model, "record_", None)
    feature_names = None
    if record is not None and hasattr(record, "feature_names_in_"):
        feature_names = list(record.feature_names_in_)
    elif hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)
    elif record is not None:
        feature_names = [
            f"x{i}" for i in range(getattr(record, "n_features", len(model.coef_)))
        ]

    return {
        "spec_version": MODEL_SPEC_VERSION,
        "model_type": model.__class__.__name__,
        "module_version": getattr(model, "__version__", None),
        "params": {
            "max_degree": model.max_degree,
            "penalty": model.penalty,
            "max_terms": model.max_terms,
            "minspan_alpha": model.minspan_alpha,
            "endspan_alpha": model.endspan_alpha,
            "minspan": model.minspan,
            "endspan": model.endspan,
            "allow_linear": model.allow_linear,
            "allow_missing": model.allow_missing,
            "feature_importance_type": model.feature_importance_type,
            "categorical_features": list(model.categorical_features or []),
        },
        "feature_schema": {
            "n_features": getattr(record, "n_features", None),
            "feature_names": feature_names,
        },
        "basis_terms": [basis_function_to_spec(bf).to_dict() for bf in model.basis_],
        "coefficients": np.asarray(model.coef_, dtype=float).tolist(),
        "metrics": {
            "rss": model.rss_,
            "mse": model.mse_,
            "gcv": model.gcv_,
        },
        "categorical_imputer": categorical_imputer_to_spec(
            getattr(model, "categorical_imputer_", None)
        ),
    }


def spec_to_model(payload: dict[str, Any], earth_cls: type[Any]) -> Any:
    """Rehydrate an Earth-compatible estimator from a portable spec."""
    payload = validate_model_spec(payload)
    params = copy.deepcopy(payload.get("params", {}))
    if "allow_missing" not in params:
        # Legacy fixtures omitted this flag. Infer missing-value support from
        # basis terms that require missing-aware runtime evaluation.
        params["allow_missing"] = any(
            term_payload.get("kind") in {"categorical", "missingness"}
            for term_payload in payload["basis_terms"]
        )
    model = earth_cls(**params)
    model.model_spec_ = copy.deepcopy(payload)
    model.basis_ = [
        basis_function_from_spec(BasisTermSpec.from_dict(term_payload))
        for term_payload in payload["basis_terms"]
    ]
    model.coef_ = np.asarray(payload["coefficients"], dtype=float)
    metrics = payload.get("metrics", {})
    model.rss_ = metrics.get("rss")
    model.mse_ = metrics.get("mse")
    model.gcv_ = metrics.get("gcv")
    feature_schema = payload.get("feature_schema", {})
    model.feature_names_in_ = np.asarray(
        feature_schema.get("feature_names")
        or [f"x{i}" for i in range(feature_schema.get("n_features", 0))],
        dtype=object,
    )
    model.n_features_in_ = int(feature_schema.get("n_features", len(model.feature_names_in_)))
    model.record_ = SimpleNamespace(
        n_features=model.n_features_in_,
        feature_names_in_=model.feature_names_in_,
    )
    model.categorical_imputer_ = categorical_imputer_from_spec(
        payload.get("categorical_imputer")
    )
    model.fitted_ = True
    return model


def spec_to_json(payload: dict[str, Any]) -> str:
    """Serialize a model spec dictionary to JSON."""
    return json.dumps(payload, indent=2, sort_keys=True)


def spec_from_json(payload: str) -> dict[str, Any]:
    """Parse a JSON model spec payload."""
    return validate_model_spec(cast(dict[str, Any], json.loads(payload)))
