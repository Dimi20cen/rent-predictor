import unittest
import tempfile
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml_pipeline import (
    add_derived_features,
    ArtifactValidationError,
    build_model_manifest,
    ContractValidationError,
    dataframe_fingerprint,
    get_model_version,
    encode_train_test,
    predict_rent,
    regression_metrics,
    resolve_feature_names,
    rent_bands,
    segmented_regression_metrics,
    split_train_test,
    split_features_target,
    transform_features_for_model,
    validate_model_artifacts,
    validate_prediction_dataframe,
    validate_training_dataframe,
)


class DummyModel:
    def predict(self, x):
        return np.log1p(np.array([1000.0] * len(x)))


class DummyBooster:
    def __init__(self, feature_names):
        self.feature_names = feature_names


class DummyFeatureModel:
    def __init__(self, feature_names):
        self._booster = DummyBooster(feature_names)

    def get_booster(self):
        return self._booster


class MlPipelineTests(unittest.TestCase):
    def setUp(self):
        base_rows = [
            {"Rent": 2000.0, "Zip": 8001, "Canton": "ZH", "SubType": "FLAT", "Rooms": 2.5, "Area_m2": 60.0},
            {"Rent": 2200.0, "Zip": 8001, "Canton": "ZH", "SubType": "FLAT", "Rooms": 3.0, "Area_m2": 70.0},
            {"Rent": 1800.0, "Zip": 1200, "Canton": "GE", "SubType": "STUDIO", "Rooms": 1.5, "Area_m2": 35.0},
        ]
        self.df = pd.DataFrame(base_rows)

        defaults = {
            "Floor": 2.0,
            "Lat": 47.3769,
            "Lon": 8.5417,
            "dist_to_Zurich_HB": 1.0,
            "dist_to_Geneva_Cornavin": 220.0,
            "dist_to_Basel_SBB": 85.0,
            "dist_to_Bern_HB": 125.0,
            "dist_to_Lausanne_Gare": 175.0,
            "dist_to_closest_hub": 1.0,
            "tax_rate": 30.0,
            "is_rent_estimated": 0.0,
            "year_built_is_missing": 1.0,
            "is_renovated": 0.0,
            "Balcony": 0.0,
            "Elevator": 1.0,
            "Parking": 0.0,
            "View": 0.0,
            "Fireplace": 0.0,
            "Child_Friendly": 0.0,
            "CableTV": 0.0,
            "New_Building": 0.0,
            "Minergie": 0.0,
            "Wheelchair": 0.0,
            "has_lake_view": 0.0,
            "is_attic": 0.0,
            "is_quiet": 1.0,
            "is_sunny": 1.0,
        }
        for col, value in defaults.items():
            self.df[col] = value

    def test_split_encode_and_transform(self):
        x, y = split_features_target(self.df)
        self.assertEqual(len(x), 3)
        self.assertEqual(len(y), 3)

        x_train_raw = x.iloc[:2].copy()
        x_test_raw = x.iloc[2:].copy()
        y_train = y.iloc[:2].copy()

        x_train, x_test, encoder = encode_train_test(x_train_raw, x_test_raw, y_train)
        self.assertIn("Zip_encoded", x_train.columns)
        self.assertNotIn("Zip", x_train.columns)

        feature_names = list(x_train.columns)
        x_transformed = transform_features_for_model(x_test_raw, encoder, feature_names)
        self.assertEqual(set(x_transformed.columns), set(feature_names))
        self.assertEqual(len(x_transformed), 1)

    def test_predict_and_metrics(self):
        model = DummyModel()
        x = pd.DataFrame({"a": [1.0, 2.0]})
        preds = predict_rent(model, x)
        self.assertTrue(np.allclose(preds, [1000.0, 1000.0]))

        metrics = regression_metrics(pd.Series([900.0, 1100.0]), preds)
        self.assertIn("mae", metrics)
        self.assertIn("rmse", metrics)
        self.assertIn("r2", metrics)

    def test_validate_training_dataframe_missing_target(self):
        invalid = self.df.drop(columns=["Rent"])
        with self.assertRaises(ContractValidationError):
            validate_training_dataframe(invalid)

    def test_validate_prediction_dataframe_missing_required_column(self):
        invalid = self.df.drop(columns=["Canton", "Rent"])
        with self.assertRaises(ContractValidationError):
            validate_prediction_dataframe(invalid)

    def test_fingerprint_is_deterministic(self):
        fp_1 = dataframe_fingerprint(self.df)
        fp_2 = dataframe_fingerprint(self.df.copy())
        self.assertEqual(fp_1, fp_2)

    def test_segmented_metrics_outputs_rows(self):
        y_true = pd.Series([1000.0, 1200.0, 2500.0, 2600.0])
        y_pred = np.array([1050.0, 1100.0, 2450.0, 2700.0])
        segments = pd.Series(["A", "A", "B", "B"])
        result = segmented_regression_metrics(y_true, y_pred, segments, "group")
        self.assertEqual(len(result), 2)
        self.assertIn("mae", result[0])
        self.assertIn("count", result[0])

        bands = rent_bands(y_true)
        self.assertEqual(len(bands), len(y_true))

    def test_resolve_feature_names_prefers_file(self):
        model = DummyFeatureModel(["from_booster"])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "feature_columns.json"
            path.write_text('["from_file"]', encoding="utf-8")
            names = resolve_feature_names(model, path)
        self.assertEqual(names, ["from_file"])

    def test_resolve_feature_names_fallback_to_booster(self):
        model = DummyFeatureModel(["a", "b"])
        names = resolve_feature_names(model, "missing.json")
        self.assertEqual(names, ["a", "b"])

    def test_add_derived_features_creates_expected_columns(self):
        features = add_derived_features(self.df.drop(columns=["Rent"]))
        self.assertIn("area_per_room", features.columns)
        self.assertIn("log_area_m2", features.columns)
        self.assertIn("tax_distance_interaction", features.columns)
        self.assertIn("is_top_floor_proxy", features.columns)

    def test_group_zip_split_has_no_zip_overlap(self):
        x, y = split_features_target(self.df)
        x_train, x_test, _, _ = split_train_test(x, y, test_size=0.5, random_state=42, strategy="group_zip")
        self.assertEqual(set(x_train["Zip"]).intersection(set(x_test["Zip"])), set())

    def test_manifest_build_and_validate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model = root / "model.pkl"
            encoder = root / "encoder.pkl"
            features = root / "feature_columns.json"
            metrics = root / "training_metrics.json"
            manifest = root / "model_manifest.json"

            model.write_bytes(b"model-bytes")
            encoder.write_bytes(b"encoder-bytes")
            features.write_text('["a","b"]', encoding="utf-8")
            metrics.write_text('{"mae": 1.0}', encoding="utf-8")

            payload = build_model_manifest(model, encoder, features, metrics, {"data_path": "x"})
            manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            validated = validate_model_artifacts(manifest)
            self.assertIn("model_version", validated)
            self.assertEqual(get_model_version(manifest), payload["model_version"])

    def test_manifest_validation_fails_on_tampered_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model = root / "model.pkl"
            encoder = root / "encoder.pkl"
            features = root / "feature_columns.json"
            metrics = root / "training_metrics.json"
            manifest = root / "model_manifest.json"

            model.write_bytes(b"model-bytes")
            encoder.write_bytes(b"encoder-bytes")
            features.write_text('["a","b"]', encoding="utf-8")
            metrics.write_text('{"mae": 1.0}', encoding="utf-8")

            payload = build_model_manifest(model, encoder, features, metrics, {"data_path": "x"})
            manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            model.write_bytes(b"tampered")
            with self.assertRaises(ArtifactValidationError):
                validate_model_artifacts(manifest)


if __name__ == "__main__":
    unittest.main()
