"""Which OpenDRIVE sign signals the baked level's server copy keeps (pure Python; bake_level.copy_xodr uses it).

At load a CARLA 1.0 server spawns its stock sign blueprints at the level's OpenDRIVE sign signals
(ATrafficLightManager::SpawnSignals): BP_SpeedLimit* is a US "SPEED LIMIT" plate, wrong on a Dutch twin, and with
tens of thousands of signals (EhvCity) loading never ends (tools/xodr_drop_signals.py). Coupled CARLA draws the real
signs itself where a sign register exists (the Netherlands: NDW, coupledcarla/ndwsigns.py), so there the server copy
drops them. Traffic-light signals (1000001, 1000002) always stay. The client keeps the full OpenDRIVE.

    mode "auto"  : "stock" only where CARLA's stock plates are the country's own design (STOCK_SIGN_COUNTRIES: the
                   US "SPEED LIMIT" / YIELD family), else "none": the client draws the real signs, from the
                   register in the Netherlands (NDW) and from OSM elsewhere, in the country's design (coupled-carla
                   streetlook.osm_face: the Vienna Convention's roundel and triangle; no octagon in Japan)
    mode "none"  : drop speed limit (274), yield (205), stop (206), priority road (306)
    mode "stock" : keep every signal (CARLA's stock signs)
"""
import re

SIGN_TYPES = ("274", "205", "206", "306")
CLIENT_SIGN_COUNTRIES = frozenset({"NL"})       # countries whose real signs Coupled CARLA places from a register
# countries whose signs are the US designs CARLA's stock blueprints show (coupled-carla placement.YIELD_TEXT and
# streetlook.SPEED_RECT, ISO2): only there the server keeps its own plates (user, 2026-11-22: local signs in every
# twin, no US plates in Eindhoven, Salamanca, Helsinki, Zurich or Tokyo)
STOCK_SIGN_COUNTRIES = frozenset({"US", "CA", "PR", "GU", "VI", "AS", "MP", "PH", "LR"})
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
        c = (country or "").upper()
        return () if (not c or c in STOCK_SIGN_COUNTRIES) else SIGN_TYPES
    raise ValueError("server signs mode %r (auto, none, stock)" % mode)
