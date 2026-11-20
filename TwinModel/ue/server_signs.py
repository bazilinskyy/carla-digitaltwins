"""Which OpenDRIVE sign signals the baked level's server copy keeps (pure Python; bake_level.copy_xodr uses it).

At load a CARLA 1.0 server spawns its stock sign blueprints at the level's OpenDRIVE sign signals
(ATrafficLightManager::SpawnSignals): BP_SpeedLimit* is a US "SPEED LIMIT" plate, wrong on a Dutch twin, and with
tens of thousands of signals (EhvCity) loading never ends (tools/xodr_drop_signals.py). Coupled CARLA draws the real
signs itself where a sign register exists (the Netherlands: NDW, coupledcarla/ndwsigns.py), so there the server copy
drops them. Traffic-light signals (1000001, 1000002) always stay. The client keeps the full OpenDRIVE.

    mode "auto"  : "none" where the signals' country has a client sign register (CLIENT_SIGN_COUNTRIES), else "stock"
    mode "none"  : drop speed limit (274), yield (205), stop (206), priority road (306)
    mode "stock" : keep every signal (CARLA's stock signs)
"""
import re

SIGN_TYPES = ("274", "205", "206", "306")
CLIENT_SIGN_COUNTRIES = frozenset({"NL"})       # countries whose real signs Coupled CARLA places from a register
_COUNTRY = re.compile(r'<signal\s[^>]*?\bcountry="([^"]*)"')


def xodr_country(path, max_lines=2_000_000):
    """The country attribute of the first signal in an OpenDRIVE file (upper case), or None."""
    with open(path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            m = _COUNTRY.search(line)
            if m:
                return m.group(1).upper() or None
            if i > max_lines:
                break
    return None


def types_to_drop(mode, country):
    """The sign signal types the server copy leaves out for ``mode`` and the signals' ``country``."""
    if mode == "stock":
        return ()
    if mode == "none":
        return SIGN_TYPES
    if mode == "auto":
        return SIGN_TYPES if (country or "").upper() in CLIENT_SIGN_COUNTRIES else ()
    raise ValueError("server signs mode %r (auto, none, stock)" % mode)
