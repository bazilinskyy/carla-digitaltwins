"""ue/server_signs.py + tools/xodr_drop_signals.py: the level's OpenDRIVE keeps no sign signals where the client
draws the real signs (no US SPEED LIMIT plates on a Dutch twin)."""
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "ue"))
sys.path.insert(0, str(HERE / "tools"))
import server_signs as ss  # noqa: E402
import xodr_drop_signals  # noqa: E402

XODR = """<?xml version="1.0"?>
<OpenDRIVE>
  <road id="1">
    <signals>
      <signal id="s1" s="0" t="-4" name="Speed_30" dynamic="no" orientation="+" zOffset="0" country="{c}" type="274" subtype="30" value="30" unit="km/h"/>
      <signal id="s2" s="5" t="-4" name="Sign_Yield" dynamic="no" orientation="+" zOffset="0" country="{c}" type="205" subtype="-1"/>
      <signal id="t1" s="9" t="-4" name="Signal_3Light_Post01" dynamic="yes" orientation="+" zOffset="0" country="{c}" type="1000001" subtype="-1">
        <validity fromLane="-1" toLane="-1"/>
      </signal>
    </signals>
  </road>
</OpenDRIVE>
"""


def test_modes_and_country(tmp_path):
    nl = tmp_path / "nl.xodr"
    nl.write_text(XODR.format(c="NL"))
    assert ss.xodr_country(nl) == "NL"
    assert ss.types_to_drop("auto", "NL") == ss.SIGN_TYPES
    assert ss.types_to_drop("auto", "ES") == ss.SIGN_TYPES   # CARLA's US plates are not Spain's: the client draws them
    assert ss.types_to_drop("auto", "JP") == ss.SIGN_TYPES and ss.types_to_drop("auto", "FI") == ss.SIGN_TYPES
    assert ss.types_to_drop("auto", "US") == () and ss.types_to_drop("auto", "ca") == ()   # US designs: stock stays
    assert ss.types_to_drop("auto", None) == ()          # unknown country: nothing dropped
    assert ss.types_to_drop("none", "US") == ss.SIGN_TYPES and ss.types_to_drop("stock", "NL") == ()
    with pytest.raises(ValueError):
        ss.types_to_drop("regional", "NL")


def test_server_copy_without_signs_keeps_traffic_lights(tmp_path):
    src, dst = tmp_path / "nl.xodr", tmp_path / "server.xodr"
    src.write_text(XODR.format(c="NL"))
    dropped = xodr_drop_signals.drop_signals(str(src), str(dst), set(ss.types_to_drop("auto", ss.xodr_country(src))))
    out = dst.read_text()
    assert dict(dropped) == {"274": 1, "205": 1}
    assert 'type="274"' not in out and 'type="205"' not in out
    assert 'type="1000001"' in out and "<validity" in out
