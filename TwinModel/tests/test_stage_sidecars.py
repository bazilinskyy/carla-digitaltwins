"""tools/stage_sidecars.py: per-level server copies of the twins' OpenDRIVE (light-free, sign-free) and the manifest."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "tools"))
sys.path.insert(0, str(HERE / "tests"))
import stage_sidecars as ss  # noqa: E402
from test_server_signs import XODR  # noqa: E402


def _twin(content, name, country):
    d = content / "Carla" / "Maps" / "Twins" / name / "OpenDrive"
    d.mkdir(parents=True)
    (d / f"{name}.xodr").write_text(XODR.format(c=country))


def test_light_free_and_sign_free_server_copies(tmp_path):
    c = tmp_path / "Content"
    _twin(c, "EhvCity", "ES")          # baked before the country stamp: the level default says NL, lights none
    _twin(c, "TUeCampus", "NL")
    _twin(c, "ShibuyaCrossing", "JP")
    man = ss.stage(c, {"TUeCampus": {"light_actors": "carla"}})
    od = c / "Carla" / "Maps" / "OpenDrive"
    city, campus, shibuya = ((od / f"{n}.xodr").read_text() for n in ("EhvCity", "TUeCampus", "ShibuyaCrossing"))
    assert 'type="1000001"' not in city and 'type="274"' not in city              # no lights, no signs
    assert 'type="1000001"' in campus and 'type="274"' not in campus              # lights kept, signs out (NL)
    assert 'type="1000001"' in shibuya and 'type="274"' in shibuya                # JP: CARLA's own, untouched
    assert man["EhvCity"]["light_actors"] == "none" and man["EhvCity"]["country"] == "NL"
    assert man["EhvCity"]["dropped"] == {"274": 1, "205": 1, "1000001": 1}
    assert man["TUeCampus"]["light_actors"] == "carla" and man["TUeCampus"]["signs"] == "client"
    assert man["ShibuyaCrossing"]["signs"] == "carla" and man["ShibuyaCrossing"]["dropped"] == {}
    assert json.dumps(man)                                                         # the manifest serialises


def test_bad_option(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        ss.plan("X", {"light_actors": "maybe"}, "NL")
