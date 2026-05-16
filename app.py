from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd

from src.ml_pipeline import (
    ArtifactValidationError,
    ContractValidationError,
    get_model_version,
    predict_rent,
    resolve_feature_names,
    transform_features_for_model,
    validate_model_artifacts,
)


PROJECT_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = PROJECT_ROOT / "static"
DEFAULT_PORT = 8501
AREA_MIN = 10
AREA_MAX = 500
ROOMS_MIN = 1.0
ROOMS_MAX = 10.0
FLOOR_MIN = -1
FLOOR_MAX = 20

HUBS = {
    "Zurich_HB": (47.378177, 8.540192),
    "Geneva_Cornavin": (46.210226, 6.142456),
    "Basel_SBB": (47.547412, 7.589556),
    "Bern_HB": (46.948833, 7.439122),
    "Lausanne_Gare": (46.516777, 6.629095),
}


def haversine_distance(lat1, lon1, lat2, lon2):
    earth_radius_km = 6371
    lat1_radians = np.radians(lat1)
    lat2_radians = np.radians(lat2)
    lat_delta_radians = np.radians(lat2 - lat1)
    lon_delta_radians = np.radians(lon2 - lon1)
    haversine_value = (
        np.sin(lat_delta_radians / 2) ** 2
        + np.cos(lat1_radians) * np.cos(lat2_radians) * np.sin(lon_delta_radians / 2) ** 2
    )
    angular_distance = 2 * np.arctan2(np.sqrt(haversine_value), np.sqrt(1 - haversine_value))
    return earth_radius_km * angular_distance


@lru_cache(maxsize=1)
def load_resources():
    validate_model_artifacts(PROJECT_ROOT / "models/model_manifest.json")

    with open(PROJECT_ROOT / "models/xgb_rent_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    with open(PROJECT_ROOT / "models/zip_encoder.pkl", "rb") as encoder_file:
        encoder = pickle.load(encoder_file)

    feature_columns = resolve_feature_names(model, PROJECT_ROOT / "models/feature_columns.json")
    reference_dataframe = pd.read_pickle(PROJECT_ROOT / "data/processed/02_featured_data.pkl")
    model_version = get_model_version(PROJECT_ROOT / "models/model_manifest.json")
    return model, encoder, feature_columns, reference_dataframe, model_version


def get_options_payload():
    _, _, _, reference_dataframe, model_version = load_resources()
    cantons = sorted(str(canton) for canton in reference_dataframe["Canton"].dropna().unique())
    subtypes = sorted(str(subtype) for subtype in reference_dataframe["SubType"].dropna().unique())
    zip_codes_by_canton = {}

    for canton in cantons:
        canton_zip_codes = reference_dataframe.loc[reference_dataframe["Canton"] == canton, "Zip"].dropna().unique()
        zip_codes_by_canton[canton] = sorted(int(zip_code) for zip_code in canton_zip_codes)

    return {
        "cantons": cantons,
        "defaultCanton": "ZH" if "ZH" in cantons else cantons[0],
        "defaultSubtype": "FLAT" if "FLAT" in subtypes else subtypes[0],
        "subtypes": subtypes,
        "zipCodesByCanton": zip_codes_by_canton,
        "modelVersion": model_version,
    }


def get_required_request_value(request_payload, field_name):
    if field_name not in request_payload:
        raise KeyError(field_name)
    return request_payload[field_name]


def parse_number_request_value(request_payload, field_name, display_name, minimum_value, maximum_value):
    raw_value = get_required_request_value(request_payload, field_name)
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        raise ValueError(f"{display_name} must be a number.")

    parsed_value = float(raw_value)
    if not np.isfinite(parsed_value):
        raise ValueError(f"{display_name} must be a finite number.")

    if parsed_value < minimum_value or parsed_value > maximum_value:
        raise ValueError(f"{display_name} must be between {minimum_value} and {maximum_value}.")

    return parsed_value


def parse_integer_request_value(request_payload, field_name, display_name, minimum_value, maximum_value):
    parsed_value = parse_number_request_value(
        request_payload,
        field_name,
        display_name,
        minimum_value,
        maximum_value,
    )
    if not parsed_value.is_integer():
        raise ValueError(f"{display_name} must be a whole number.")

    return int(parsed_value)


def parse_boolean_request_value(request_payload, field_name, default_value=False):
    raw_value = request_payload.get(field_name, default_value)
    if not isinstance(raw_value, bool):
        raise ValueError(f"{field_name} must be true or false.")
    return raw_value


def validate_prediction_request(request_payload, reference_dataframe):
    if not isinstance(request_payload, dict):
        raise ValueError("Request body must be a JSON object.")

    cantons = set(str(canton) for canton in reference_dataframe["Canton"].dropna().unique())
    subtypes = set(str(subtype) for subtype in reference_dataframe["SubType"].dropna().unique())

    selected_canton = get_required_request_value(request_payload, "canton")
    if not isinstance(selected_canton, str) or selected_canton not in cantons:
        raise ValueError("Canton must be one of the available canton options.")

    selected_property_type = get_required_request_value(request_payload, "propertyType")
    if not isinstance(selected_property_type, str) or selected_property_type not in subtypes:
        raise ValueError("Property type must be one of the available property type options.")

    selected_zip = parse_integer_request_value(request_payload, "zipCode", "Zip code", 1000, 9999)
    canton_zip_codes = reference_dataframe.loc[reference_dataframe["Canton"] == selected_canton, "Zip"].dropna().unique()
    zip_codes_for_selected_canton = set(int(zip_code) for zip_code in canton_zip_codes)
    if selected_zip not in zip_codes_for_selected_canton:
        raise ValueError("Zip code must belong to the selected canton.")

    return {
        "area": parse_number_request_value(request_payload, "area", "Living area", AREA_MIN, AREA_MAX),
        "rooms": parse_number_request_value(request_payload, "rooms", "Rooms", ROOMS_MIN, ROOMS_MAX),
        "floor": parse_integer_request_value(request_payload, "floor", "Floor", FLOOR_MIN, FLOOR_MAX),
        "canton": selected_canton,
        "zipCode": selected_zip,
        "propertyType": selected_property_type,
        "hasLake": parse_boolean_request_value(request_payload, "hasLake"),
        "isNew": parse_boolean_request_value(request_payload, "isNew"),
        "isQuiet": parse_boolean_request_value(request_payload, "isQuiet"),
    }


def build_prediction_payload(request_payload):
    model, encoder, model_feature_names, reference_dataframe, model_version = load_resources()
    validated_request_payload = validate_prediction_request(request_payload, reference_dataframe)

    selected_canton = validated_request_payload["canton"]
    selected_property_type = validated_request_payload["propertyType"]
    selected_zip = validated_request_payload["zipCode"]
    area = validated_request_payload["area"]
    rooms = validated_request_payload["rooms"]
    floor = validated_request_payload["floor"]
    has_lake = validated_request_payload["hasLake"]
    is_new = validated_request_payload["isNew"]
    is_quiet = validated_request_payload["isQuiet"]

    matching_reference_rows = reference_dataframe[reference_dataframe["Zip"] == selected_zip]
    if matching_reference_rows.empty:
        raise ValueError(f"No reference data found for zip code {selected_zip}.")

    reference_row = matching_reference_rows.iloc[0]
    lat = reference_row["Lat"]
    lon = reference_row["Lon"]
    tax_rate = reference_row["tax_rate"]

    distances_to_hubs = {}
    for hub_name, hub_coordinates in HUBS.items():
        distances_to_hubs[f"dist_to_{hub_name}"] = haversine_distance(
            lat,
            lon,
            hub_coordinates[0],
            hub_coordinates[1],
        )
    minimum_distance_to_hub = min(distances_to_hubs.values())

    input_data = {
        "Rooms": [rooms],
        "Area_m2": [area],
        "Floor": [floor],
        "Canton": [selected_canton],
        "SubType": [selected_property_type],
        "Lat": [lat],
        "Lon": [lon],
        "dist_to_Zurich_HB": [distances_to_hubs["dist_to_Zurich_HB"]],
        "dist_to_Geneva_Cornavin": [distances_to_hubs["dist_to_Geneva_Cornavin"]],
        "dist_to_Basel_SBB": [distances_to_hubs["dist_to_Basel_SBB"]],
        "dist_to_Bern_HB": [distances_to_hubs["dist_to_Bern_HB"]],
        "dist_to_Lausanne_Gare": [distances_to_hubs["dist_to_Lausanne_Gare"]],
        "dist_to_closest_hub": [minimum_distance_to_hub],
        "tax_rate": [tax_rate],
        "is_rent_estimated": [0],
        "year_built_is_missing": [1],
        "is_renovated": [1 if is_new else 0],
        "Balcony": [1 if has_lake else 0],
        "Elevator": [1],
        "Parking": [0],
        "View": [1 if has_lake else 0],
        "Fireplace": [0],
        "Child_Friendly": [0],
        "CableTV": [0],
        "New_Building": [1 if is_new else 0],
        "Minergie": [0],
        "Wheelchair": [0],
        "has_lake_view": [1 if has_lake else 0],
        "is_attic": [0],
        "is_quiet": [1 if is_quiet else 0],
        "is_sunny": [0],
        "Zip": [selected_zip],
    }

    prediction_input_dataframe = pd.DataFrame(input_data)
    model_input_dataframe = transform_features_for_model(prediction_input_dataframe, encoder, model_feature_names)
    monthly_rent = int(predict_rent(model, model_input_dataframe)[0])

    return {
        "monthlyRent": monthly_rent,
        "annualRent": monthly_rent * 12,
        "details": {
            "zipCode": selected_zip,
            "taxIndex": float(tax_rate),
            "distanceToNearestHubKm": int(minimum_distance_to_hub),
            "modelVersion": model_version,
            "modelType": "XGBoost",
        },
    }


class RentPredictionRequestHandler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_static_file(STATIC_ROOT / "index.html", head_only=True)
            return

        if self.path.startswith("/static/"):
            requested_static_path = STATIC_ROOT / self.path.removeprefix("/static/")
            self.send_static_file(requested_static_path, head_only=True)
            return

        self.send_error_response(HTTPStatus.NOT_FOUND, "Not found.")

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_static_file(STATIC_ROOT / "index.html")
            return

        if self.path == "/api/options":
            self.send_json_response(get_options_payload())
            return

        if self.path.startswith("/static/"):
            requested_static_path = STATIC_ROOT / self.path.removeprefix("/static/")
            self.send_static_file(requested_static_path)
            return

        self.send_error_response(HTTPStatus.NOT_FOUND, "Not found.")

    def do_POST(self):
        if self.path != "/api/predict":
            self.send_error_response(HTTPStatus.NOT_FOUND, "Not found.")
            return

        try:
            request_payload = self.read_json_body()
            prediction_payload = build_prediction_payload(request_payload)
        except KeyError as exc:
            self.send_error_response(HTTPStatus.BAD_REQUEST, f"Missing required field: {exc.args[0]}.")
        except (ValueError, json.JSONDecodeError) as exc:
            self.send_error_response(HTTPStatus.BAD_REQUEST, str(exc))
        except ContractValidationError as exc:
            self.send_error_response(HTTPStatus.BAD_REQUEST, f"Input contract error: {exc}")
        except ArtifactValidationError as exc:
            self.send_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, f"Artifact integrity error: {exc}")
        except Exception as exc:
            self.send_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, f"Prediction failed: {exc}")
        else:
            self.send_json_response(prediction_payload)

    def read_json_body(self):
        content_length = int(self.headers.get("Content-Length", "0"))
        request_body = self.rfile.read(content_length).decode("utf-8")
        return json.loads(request_body)

    def send_static_file(self, static_file_path, head_only=False):
        resolved_static_file_path = static_file_path.resolve()
        if not str(resolved_static_file_path).startswith(str(STATIC_ROOT.resolve())):
            self.send_error_response(HTTPStatus.NOT_FOUND, "Not found.")
            return

        if not resolved_static_file_path.is_file():
            self.send_error_response(HTTPStatus.NOT_FOUND, "Not found.")
            return

        response_body = resolved_static_file_path.read_bytes()
        content_type = mimetypes.guess_type(resolved_static_file_path)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        if not head_only:
            self.wfile.write(response_body)

    def send_json_response(self, response_payload, status=HTTPStatus.OK):
        response_body = json.dumps(response_payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)

    def send_error_response(self, status, message):
        self.send_json_response({"error": message}, status=status)

    def log_message(self, format, *args):
        sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), format % args))


def main():
    port = int(os.environ.get("PORT", DEFAULT_PORT))
    server = ThreadingHTTPServer(("0.0.0.0", port), RentPredictionRequestHandler)
    print(f"rent-predictor web app running at http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
