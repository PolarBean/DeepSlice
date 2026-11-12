from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from DeepSlice.coord_post_processing import angle_methods
from DeepSlice.coord_post_processing.plane_alignment_functions import plane_alignment


RESULTS = Path(__file__).parents[1] / "examples" / "example_brain" / "GLTa" / "MyResults.csv"
MAX_DEPTH = {"mouse": 528, "rat": 1024}


@pytest.mark.parametrize("species", ["mouse", "rat"])
@pytest.mark.parametrize("depths", [[300.4], [300.4, 410.7], [0.0, 527.9]])
def test_weighted_mean_of_one_or_two_sections_is_the_mean(species, depths):
    DV_list, ML_list = [1.0, 3.0][: len(depths)], [-2.0, 4.0][: len(depths)]
    DV, ML = angle_methods.get_mean_angle(DV_list, ML_list, "weighted_mean", depths, species)
    assert DV == pytest.approx(np.mean(DV_list))
    assert ML == pytest.approx(np.mean(ML_list))


@pytest.mark.parametrize("species", ["mouse", "rat"])
def test_weighted_mean_clamps_depths_to_the_atlas(species):
    max_depth = MAX_DEPTH[species]
    weights = plane_alignment.make_gaussian_weights(max_depth)
    DV_list, ML_list = [1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]
    depths = [-5.0, 100.0, max_depth, max_depth + 10.0]
    expected_weights = [weights[0], weights[100], weights[max_depth - 1], weights[max_depth - 1]]
    DV, ML = angle_methods.get_mean_angle(DV_list, ML_list, "weighted_mean", depths, species)
    assert DV == pytest.approx(np.average(DV_list, weights=expected_weights))
    assert ML == pytest.approx(np.average(ML_list, weights=expected_weights))


def tilted_sections(n_sections):
    # the example results already have propagated angles, so give every section different angles first
    df = pd.read_csv(RESULTS).iloc[:n_sections]
    rows = [angle_methods.set_angles(df.iloc[[i]].copy(), 2.0 * i, -1.5 * i) for i in range(n_sections)]
    return pd.concat(rows)


@pytest.mark.parametrize("n_sections", [1, 2, 3])
def test_propagate_angles_of_a_few_sections(n_sections):
    df = tilted_sections(n_sections)
    DV_in, ML_in = angle_methods.calculate_angles(df)
    out = angle_methods.propagate_angles(df.copy(), "weighted_mean", "mouse")
    DV, ML = angle_methods.calculate_angles(out)
    # every section now has the same angles
    assert np.allclose(DV, DV[0], atol=1e-2)
    assert np.allclose(ML, ML[0], atol=1e-2)
    if n_sections <= 2:
        # with equal weights, that is the mean of the input angles
        assert DV[0] == pytest.approx(np.mean(DV_in), abs=1e-2)
        assert ML[0] == pytest.approx(np.mean(ML_in), abs=1e-2)


def test_propagate_angles_with_all_but_two_sections_bad():
    df = pd.read_csv(RESULTS).iloc[:5].copy()
    df["bad_section"] = [True, True, True, False, False]
    out = angle_methods.propagate_angles(df, "weighted_mean", "mouse")
    assert np.isfinite(out[["ox", "oy", "oz", "ux", "uy", "uz", "vx", "vy", "vz"]].to_numpy()).all()
