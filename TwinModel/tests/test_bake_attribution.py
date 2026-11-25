"""ue/bake_level.py's credits step outside Unreal (a stand-in `unreal` module): the twin's ATTRIBUTION.md travels with
the level even when an older manifest does not name it (NycMidtown, credited afterwards on 2026-11-25)."""
import importlib.util
import json
import sys
import types
from pathlib import Path

UE = Path(__file__).resolve().parents[1] / "ue"


def _bake_level(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "unreal", types.ModuleType("unreal"))
    monkeypatch.syspath_prepend(str(UE))
    spec = importlib.util.spec_from_file_location("bake_level_under_test", UE / "bake_level.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "content_dir_for", lambda root: str(tmp_path / "Content"))
    said = []
    monkeypatch.setattr(mod, "log", said.append)
    monkeypatch.setattr(mod, "warn", lambda m: said.append("WARN " + m))
    return mod, said


def test_attribution_beside_the_manifest_is_copied_even_unnamed(monkeypatch, tmp_path):
    bl, said = _bake_level(monkeypatch, tmp_path)
    ue = tmp_path / "ue"
    ue.mkdir()
    (ue / "manifest.json").write_text(json.dumps({"name": "NycMidtown"}))
    man = {"name": "NycMidtown", "_path": str(ue / "manifest.json")}
    assert bl.copy_attribution(man, "NycMidtown") is None and any("WARN no ATTRIBUTION.md" in s for s in said)
    (ue / "ATTRIBUTION.md").write_text("# NycMidtown: data sources and licences\n")
    dst = bl.copy_attribution(man, "NycMidtown")
    assert dst and Path(dst).read_text().startswith("# NycMidtown")
