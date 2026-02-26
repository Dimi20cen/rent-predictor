"""Reusable ML pipeline utilities for training/evaluation/prediction."""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import category_encoders as ce
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DROP_COLUMNS = ["ID", "Description", "City", "Date_Created", "Link", "Title", "tax_source"]
ONE_HOT_COLUMNS = ["Canton", "SubType"]
TARGET_COLUMN = "Rent"
ZIP_COLUMN = "Zip"
REQUIRED_CATEGORICAL_COLUMNS = ["Canton", "SubType", ZIP_COLUMN]
REQUIRED_NUMERIC_COLUMNS = [
    "Rooms",
    "Area_m2",
    "Floor",
    "Lat",
    "Lon",
    "dist_to_Zurich_HB",
    "dist_to_Geneva_Cornavin",
    "dist_to_Basel_SBB",
    "dist_to_Bern_HB",
    "dist_to_Lausanne_Gare",
    "dist_to_closest_hub",
    "tax_rate",
    "is_rent_estimated",
    "year_built_is_missing",
    "is_renovated",
    "Balcony",
    "Elevator",
    "Parking",
    "View",
    "Fireplace",
    "Child_Friendly",
    "CableTV",
    "New_Building",
    "Minergie",
    "Wheelchair",
    "has_lake_view",
    "is_attic",
    "is_quiet",
    "is_sunny",
]
REQUIRED_RAW_FEATURE_COLUMNS = REQUIRED_NUMERIC_COLUMNS + REQUIRED_CATEGORICAL_COLUMNS

DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "objective": "reg:absoluteerror",
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_jobs": -1,
    "random_state": 42,
    "early_stopping_rounds": 50,
}


class ContractValidationError(ValueError):
    """Raised when an input dataframe violates a required data contract."""


class ArtifactValidationError(ValueError):
    """Raised when runtime model artifacts are missing or have integrity mismatch."""


def load_featured_data(path: str | Path) -> pd.DataFrame:
    return pd.read_pickle(path)


def prepare_model_df(df: pd.DataFrame) -> pd.DataFrame:
    drop = [col for col in DROP_COLUMNS if col in df.columns]
    return df.drop(columns=drop).copy()


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    model_df = prepare_model_df(df)
    validate_training_dataframe(model_df)
    x = model_df.drop(columns=[TARGET_COLUMN]).copy()
    y = model_df[TARGET_COLUMN].copy()
    return x, y


def _missing_columns(df: pd.DataFrame, required: list[str]) -> list[str]:
    return [col for col in required if col not in df.columns]


def _ensure_non_null(df: pd.DataFrame, columns: list[str], context: str) -> None:
    cols = [col for col in columns if col in df.columns]
    if not cols:
        return
    null_cols = [col for col in cols if df[col].isnull().any()]
    if null_cols:
        raise ContractValidationError(
            f"{context}: columns contain null values: {', '.join(null_cols)}. "
            "Fill missing values before running this command."
        )


def _ensure_numeric(df: pd.DataFrame, columns: list[str], context: str) -> None:
    cols = [col for col in columns if col in df.columns]
    if not cols:
        return
    non_numeric = [col for col in cols if not pd.api.types.is_numeric_dtype(df[col])]
    if non_numeric:
        raise ContractValidationError(
            f"{context}: expected numeric columns but found non-numeric values in: "
            f"{', '.join(non_numeric)}."
        )


def validate_prediction_dataframe(df: pd.DataFrame, context: str = "prediction input") -> None:
    if df.empty:
        raise ContractValidationError(f"{context}: dataset is empty.")

    missing = _missing_columns(df, REQUIRED_RAW_FEATURE_COLUMNS)
    if missing:
        raise ContractValidationError(
            f"{context}: missing required columns: {', '.join(missing)}. "
            "Expected raw preprocessing columns including Zip, Canton, and SubType."
        )

    _ensure_non_null(df, REQUIRED_CATEGORICAL_COLUMNS, context)
    _ensure_non_null(df, REQUIRED_NUMERIC_COLUMNS, context)
    _ensure_numeric(df, REQUIRED_NUMERIC_COLUMNS, context)


def validate_training_dataframe(df: pd.DataFrame, context: str = "training data") -> None:
    if TARGET_COLUMN not in df.columns:
        raise ContractValidationError(
            f"{context}: missing target column: {TARGET_COLUMN}."
        )
    validate_prediction_dataframe(df, context=context)
    _ensure_non_null(df, [TARGET_COLUMN], context)
    _ensure_numeric(df, [TARGET_COLUMN], context)


def encode_train_test(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, ce.CatBoostEncoder]:
    if ZIP_COLUMN not in x_train.columns:
        raise ValueError(f"Missing required column for target encoding: {ZIP_COLUMN}")

    x_train = x_train.copy()
    x_test = x_test.copy()
    x_train = add_derived_features(x_train)
    x_test = add_derived_features(x_test)

    # Leakage-resistant target encoding for high-cardinality ZIP.
    encoder = ce.CatBoostEncoder(cols=[ZIP_COLUMN], sigma=0.05, a=5)
    x_train["Zip_encoded"] = encoder.fit_transform(x_train[ZIP_COLUMN], y_train)
    x_test["Zip_encoded"] = encoder.transform(x_test[ZIP_COLUMN])

    x_train = x_train.drop(columns=[ZIP_COLUMN])
    x_test = x_test.drop(columns=[ZIP_COLUMN])

    x_train = pd.get_dummies(x_train, columns=[c for c in ONE_HOT_COLUMNS if c in x_train.columns], drop_first=True)
    x_test = pd.get_dummies(x_test, columns=[c for c in ONE_HOT_COLUMNS if c in x_test.columns], drop_first=True)
    x_train, x_test = x_train.align(x_test, join="left", axis=1, fill_value=0)
    return x_train, x_test, encoder


def transform_features_for_model(
    x_raw: pd.DataFrame,
    encoder: ce.CatBoostEncoder,
    feature_names: list[str],
) -> pd.DataFrame:
    validate_prediction_dataframe(x_raw)

    x_raw = add_derived_features(x_raw.copy())
    x_raw["Zip_encoded"] = encoder.transform(x_raw[ZIP_COLUMN])
    x_raw = x_raw.drop(columns=[ZIP_COLUMN])
    x_raw = pd.get_dummies(x_raw, columns=[c for c in ONE_HOT_COLUMNS if c in x_raw.columns], drop_first=True)

    x_final = pd.DataFrame(0, index=x_raw.index, columns=feature_names, dtype=float)
    for col in x_raw.columns:
        if col in x_final.columns:
            x_final[col] = x_raw[col]
    return x_final


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {"Area_m2", "Rooms"}.issubset(out.columns):
        denom = out["Rooms"].replace(0, np.nan)
        out["area_per_room"] = (out["Area_m2"] / denom).replace([np.inf, -np.inf], np.nan)
        out["area_per_room"] = out["area_per_room"].fillna(out["Area_m2"])
    if "Area_m2" in out.columns:
        out["log_area_m2"] = np.log1p(out["Area_m2"].clip(lower=0))
    if {"tax_rate", "dist_to_closest_hub"}.issubset(out.columns):
        out["tax_distance_interaction"] = out["tax_rate"] * out["dist_to_closest_hub"]
    if {"Floor"}.issubset(out.columns):
        out["is_top_floor_proxy"] = (out["Floor"] >= 6).astype(int)
    return out


def split_train_test(
    x: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    strategy: str = "random",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if strategy == "random":
        return train_test_split(x, y, test_size=test_size, random_state=random_state)
    if strategy == "group_zip":
        if ZIP_COLUMN not in x.columns:
            raise ValueError(f"Missing required group column for strategy '{strategy}': {ZIP_COLUMN}")
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(x, y, groups=x[ZIP_COLUMN]))
        return x.iloc[train_idx].copy(), x.iloc[test_idx].copy(), y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
    raise ValueError(f"Unknown split strategy: {strategy}. Use 'random' or 'group_zip'.")


def resolve_feature_names(model: xgb.XGBRegressor, feature_columns_path: str | Path | None = None) -> list[str]:
    if feature_columns_path:
        cols_path = Path(feature_columns_path)
        if cols_path.exists():
            return json.loads(cols_path.read_text(encoding="utf-8"))

    booster_names = model.get_booster().feature_names
    if booster_names is None:
        raise ValueError(
            "Feature names unavailable. Run scripts/train.py to generate "
            "models/feature_columns.json or pass --feature-columns."
        )
    return list(booster_names)


def train_xgb_log_target(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    params: dict[str, Any] | None = None,
) -> xgb.XGBRegressor:
    train_log = np.log1p(y_train)
    valid_log = np.log1p(y_valid)

    merged = dict(DEFAULT_XGB_PARAMS)
    if params:
        merged.update(params)

    model = xgb.XGBRegressor(**merged)
    model.fit(
        x_train,
        train_log,
        eval_set=[(x_valid, valid_log)],
        verbose=False,
    )
    return model


def predict_rent(model: xgb.XGBRegressor, x: pd.DataFrame) -> np.ndarray:
    pred_log = model.predict(x)
    return np.expm1(pred_log)


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse),
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def dataframe_fingerprint(df: pd.DataFrame) -> str:
    """Return deterministic SHA256 fingerprint for dataframe contents."""
    row_hashes = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(row_hashes).hexdigest()


def sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_of_file(path: str | Path) -> str:
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rent_bands(y: pd.Series) -> pd.Series:
    bins = [-np.inf, 1500, 2500, 4000, np.inf]
    labels = ["low", "mid", "upper_mid", "high"]
    return pd.cut(y, bins=bins, labels=labels, include_lowest=True)


def segmented_regression_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    segments: pd.Series,
    segment_name: str,
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(
        {
            "y_true": y_true.values,
            "y_pred": y_pred,
            "segment": segments.astype("string").fillna("UNKNOWN"),
        }
    )
    out: list[dict[str, Any]] = []
    for segment_value, group in frame.groupby("segment"):
        if len(group) < 2:
            continue
        metrics = regression_metrics(group["y_true"], group["y_pred"].to_numpy())
        out.append(
            {
                segment_name: str(segment_value),
                "count": int(len(group)),
                **metrics,
            }
        )
    out.sort(key=lambda x: x["count"], reverse=True)
    return out


def build_model_manifest(
    model_path: str | Path,
    encoder_path: str | Path,
    feature_columns_path: str | Path,
    metrics_path: str | Path,
    training_payload: dict[str, Any],
) -> dict[str, Any]:
    model_path = Path(model_path)
    encoder_path = Path(encoder_path)
    feature_columns_path = Path(feature_columns_path)
    metrics_path = Path(metrics_path)

    model_version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return {
        "model_version": model_version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": {
            "model": {
                "path": str(model_path),
                "sha256": sha256_of_file(model_path),
            },
            "encoder": {
                "path": str(encoder_path),
                "sha256": sha256_of_file(encoder_path),
            },
            "feature_columns": {
                "path": str(feature_columns_path),
                "sha256": sha256_of_file(feature_columns_path),
            },
            "training_metrics": {
                "path": str(metrics_path),
                "sha256": sha256_of_file(metrics_path),
            },
        },
        "training": training_payload,
    }


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def get_model_version(manifest_path: str | Path = "models/model_manifest.json") -> str:
    path = Path(manifest_path)
    if not path.exists():
        return "unknown"
    payload = load_json(path)
    return str(payload.get("model_version", "unknown"))


def validate_model_artifacts(manifest_path: str | Path = "models/model_manifest.json") -> dict[str, Any]:
    payload = load_json(manifest_path)
    artifacts = payload.get("artifacts", {})
    if not artifacts:
        raise ArtifactValidationError("model manifest missing 'artifacts' section.")

    for name, artifact in artifacts.items():
        path = artifact.get("path")
        expected_sha = artifact.get("sha256")
        if not path or not expected_sha:
            raise ArtifactValidationError(f"artifact '{name}' missing path or sha256 in manifest.")

        artifact_path = Path(path)
        if not artifact_path.exists():
            raise ArtifactValidationError(f"artifact '{name}' missing file: {artifact_path}")

        actual_sha = sha256_of_file(artifact_path)
        if actual_sha != expected_sha:
            raise ArtifactValidationError(
                f"artifact '{name}' checksum mismatch: expected {expected_sha}, got {actual_sha}."
            )

    return payload
