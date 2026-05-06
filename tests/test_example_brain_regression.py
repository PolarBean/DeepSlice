import json
from pathlib import Path

import numpy as np
import pandas as pd

from DeepSlice import DSModel


EXAMPLE_DIR = Path("examples/example_brain/GLTa")
REFERENCE_PREFIX = EXAMPLE_DIR / "MyResults"


def test_example_brain_predictions_match_reference(tmp_path):
    output_prefix = tmp_path / "PR80Results"

    model = DSModel("mouse")
    model.predict(str(EXAMPLE_DIR) + "/", ensemble=True, section_numbers=True)
    model.set_bad_sections(bad_sections=["_s094", "s199"])
    model.propagate_angles()
    model.enforce_index_order()
    model.enforce_index_spacing(section_thickness=None)
    model.save_predictions(str(output_prefix))

    expected = pd.read_csv(str(REFERENCE_PREFIX) + ".csv")
    actual = pd.read_csv(str(output_prefix) + ".csv")

    pd.testing.assert_series_equal(actual["Filenames"], expected["Filenames"])
    pd.testing.assert_series_equal(actual["bad_section"], expected["bad_section"])
    pd.testing.assert_series_equal(actual["nr"], expected["nr"])
    pd.testing.assert_series_equal(actual["width"], expected["width"])
    pd.testing.assert_series_equal(actual["height"], expected["height"])

    numeric_columns = [
        "ox",
        "oy",
        "oz",
        "ux",
        "uy",
        "uz",
        "vx",
        "vy",
        "vz",
        "depths",
    ]
    for column in numeric_columns:
        actual_values = actual[column].to_numpy()
        expected_values = expected[column].to_numpy()
        max_delta = np.max(np.abs(actual_values - expected_values))
        assert np.allclose(actual_values, expected_values, rtol=1e-5, atol=1e-3), (
            f"{column} drifted from {REFERENCE_PREFIX.name}; "
            f"max abs delta: {max_delta}"
        )

    expected_json = json.loads((REFERENCE_PREFIX.with_suffix(".json")).read_text())
    actual_json = json.loads((output_prefix.with_suffix(".json")).read_text())

    assert actual_json["target"] == expected_json["target"]
    assert actual_json["aligner"] == expected_json["aligner"]
    assert [s["filename"] for s in actual_json["slices"]] == [
        s["filename"] for s in expected_json["slices"]
    ]
