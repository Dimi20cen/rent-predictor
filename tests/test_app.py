import unittest

import pandas as pd

from app import validate_prediction_request


class AppValidationTests(unittest.TestCase):
    def setUp(self):
        self.reference_dataframe = pd.DataFrame(
            [
                {"Canton": "ZH", "SubType": "FLAT", "Zip": 8001},
                {"Canton": "GE", "SubType": "STUDIO", "Zip": 1200},
            ]
        )
        self.valid_request_payload = {
            "area": 65,
            "rooms": 2.5,
            "floor": 2,
            "canton": "ZH",
            "zipCode": 8001,
            "propertyType": "FLAT",
            "hasLake": False,
            "isNew": False,
            "isQuiet": True,
        }

    def test_validate_prediction_request_accepts_valid_payload(self):
        validated_request_payload = validate_prediction_request(
            self.valid_request_payload,
            self.reference_dataframe,
        )

        self.assertEqual(validated_request_payload["zipCode"], 8001)
        self.assertEqual(validated_request_payload["floor"], 2)
        self.assertEqual(validated_request_payload["area"], 65.0)

    def test_validate_prediction_request_rejects_string_boolean(self):
        invalid_request_payload = dict(self.valid_request_payload)
        invalid_request_payload["hasLake"] = "false"

        with self.assertRaisesRegex(ValueError, "hasLake must be true or false"):
            validate_prediction_request(invalid_request_payload, self.reference_dataframe)

    def test_validate_prediction_request_rejects_zip_from_different_canton(self):
        invalid_request_payload = dict(self.valid_request_payload)
        invalid_request_payload["zipCode"] = 1200

        with self.assertRaisesRegex(ValueError, "Zip code must belong to the selected canton"):
            validate_prediction_request(invalid_request_payload, self.reference_dataframe)

    def test_validate_prediction_request_rejects_area_outside_browser_bounds(self):
        invalid_request_payload = dict(self.valid_request_payload)
        invalid_request_payload["area"] = 5

        with self.assertRaisesRegex(ValueError, "Living area must be between 10 and 500"):
            validate_prediction_request(invalid_request_payload, self.reference_dataframe)


if __name__ == "__main__":
    unittest.main()
