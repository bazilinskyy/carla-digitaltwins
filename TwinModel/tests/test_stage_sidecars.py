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
    man = ss.stage(c, {"TUeCampus": {"light_actors": "carla"}}, lookup=lambda lat, lon: None)
    od = c / "Carla" / "Maps" / "OpenDrive"
    city, campus, shibuya = ((od / f"{n}.xodr").read_text() for n in ("EhvCity", "TUeCampus", "ShibuyaCrossing"))
    assert 'type="1000001"' not in city and 'type="274"' not in city              # no lights, no signs
    assert 'type="1000001"' in campus and 'type="274"' not in campus              # lights kept, signs out (NL)
    assert 'type="1000001"' in shibuya and 'type="274"' not in shibuya            # JP: lights kept, US plates out
    assert man["EhvCity"]["light_actors"] == "none" and man["EhvCity"]["country"] == "NL"
    assert man["EhvCity"]["dropped"] == {"274": 1, "205": 1, "1000001": 1}
    assert man["TUeCampus"]["light_actors"] == "carla" and man["TUeCampus"]["signs"] == "client"
    assert man["ShibuyaCrossing"]["signs"] == "client"      # JP: no US plates either
    assert json.dumps(man)                                                         # the manifest serialises


def test_bad_option(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        ss.plan("X", {"light_actors": "maybe"}, "NL")


GEO = ("  <header name=\"{n}\"><geoReference><![CDATA[+proj=tmerc +lat_0={lat} +lon_0={lon} +k=1 +x_0=0 +y_0=0 "
       "+datum=WGS84 +units=m +no_defs]]></geoReference></header>\n")


def _geo_twin(content, name, lat, lon, stamp="ES", place=None):
    d = content / "Carla" / "Maps" / "Twins" / name
    (d / "OpenDrive").mkdir(parents=True)
    x = XODR.format(c=stamp).replace("<OpenDRIVE>\n", "<OpenDRIVE>\n" + GEO.format(n=name, lat=lat, lon=lon), 1)
    (d / "OpenDrive" / f"{name}.xodr").write_text(x)
    if place:
        (d / "place.json").write_text(json.dumps({"name": name, "country": place}))
    return d


def test_country_from_place_record_then_origin_not_the_profile_stamp(tmp_path):
    c = tmp_path / "Content"
    _geo_twin(c, "HelsinkiSenate", 60.169, 24.948, place="fi")      # stamped ES by the eu_dense profile
    _geo_twin(c, "ZurichHB", 47.378, 8.540)                          # no place record yet: the OpenDRIVE origin
    _geo_twin(c, "SalamancaPlaza", 40.965, -5.664)                   # lookup has no answer: the stamp
    asked = []

    def lookup(lat, lon):
        asked.append((lat, lon))
        return "ch" if abs(lat - 47.378) < 1e-6 else None

    man = ss.stage(c, {}, lookup=lookup)
    assert (man["HelsinkiSenate"]["country"], man["HelsinkiSenate"]["country_source"]) == ("FI", "place record")
    assert (man["ZurichHB"]["country"], man["ZurichHB"]["country_source"]) == ("CH", "OpenDRIVE origin")
    assert (man["SalamancaPlaza"]["country"], man["SalamancaPlaza"]["country_source"]) == ("ES", "OpenDRIVE signals")
    assert (60.169, 24.948) not in asked                              # the place record answered first
    assert man["HelsinkiSenate"]["signs"] == "client"                 # FI: the client draws the signs


def test_levels_file_country_beats_place_record(tmp_path):
    c = tmp_path / "Content"
    _geo_twin(c, "EhvCity", 51.44, 5.47, place="BE")
    man = ss.stage(c, {"EhvCity": {"country": "NL"}}, lookup=lambda lat, lon: None)
    assert (man["EhvCity"]["country"], man["EhvCity"]["country_source"]) == ("NL", "levels file")
    assert man["EhvCity"]["signs"] == "client"


def test_xodr_origin(tmp_path):
    d = _geo_twin(tmp_path, "X", 60.169, 24.948)
    assert ss.xodr_origin(d / "OpenDrive" / "X.xodr") == (60.169, 24.948)
    (tmp_path / "plain.xodr").write_text(XODR.format(c="NL"))
    assert ss.xodr_origin(tmp_path / "plain.xodr") is None
