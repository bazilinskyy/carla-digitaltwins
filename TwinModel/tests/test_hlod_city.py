"""ue/hlod_city.py: which HLOD layer each baked twin actor gets (no editor)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ue"))
import hlod_city as hc  # noqa: E402


def test_layers_by_label_and_folder():
    assert hc.layer_for("Windows_3_-2", "Twin/Windows") == hc.LAYER_INSTANCED
    assert hc.layer_for("SM_EhvCity_L0_building_3_4_facade_brick_old_0", "Twin/building_3dbag") == hc.LAYER_MERGED
    assert hc.layer_for("SM_EhvCity_L0_drivable_1_1", "Twin/drivable") == hc.LAYER_MERGED
    assert hc.layer_for("SM_EhvCity_boundary", "Twin/boundary") is None          # invisible collision wall
    assert hc.layer_for("PlayerStart", "") is None
    assert hc.layer_for("VehicleSpawnPoint12", "Twin/SpawnPoints") is None
    assert hc.layer_for("SM_x", "Twin/roof", spatially_loaded=False) is None     # always loaded: no HLOD


def test_plan_counts():
    rows = [("Windows_0_0", "Twin/Windows", True), ("SM_a", "Twin/drivable", True), ("BP_Carla_Sky", "", False)]
    assert hc.plan(rows) == {hc.LAYER_INSTANCED: 1, hc.LAYER_MERGED: 1, "none": 1}
