"""Write a copy of a twin's OpenDRIVE without the signals of some types (a server copy for a city-sized level).

    python tools/xodr_drop_signals.py build/ehv_city/EhvCity.xodr EhvCity_server.xodr --types 274,205,306

Why: when a CARLA 1.0 server loads a level, ATrafficLightManager::SpawnSignals spawns one sign actor per OpenDRIVE
sign signal and, for every signal, looks for the closest existing sign by listing every sign actor in the world
(GetAllActorsOfClass): quadratic in the number of signs. EhvCity has 15,736 speed-limit (274), 11,139 yield (205)
and 3,454 priority-road (306) signals: the packaged server never finished loading it. A simulator that draws its own
signs and runs its own traffic (Coupled CARLA: NDW sign register, SUMO) does not need the server's sign actors;
traffic-light signals (1000001, 1000002) stay, so CARLA's traffic-light actors and their controllers are kept.

Streams the file (no XML tree of half a gigabyte in memory): a <signal> of a dropped type is left out with its
children (validity), any <signalReference> to it and any <control signalId=...> of it. Prints what it dropped.
"""
import argparse
import re
import sys
from collections import Counter

SIGNAL_OPEN = re.compile(r'<signal\s[^>]*?\bid="([^"]*)"[^>]*?\btype="([^"]*)"')
SELF_CLOSING = re.compile(r"/>\s*$")


def drop_signals(src, dst, types):
    """Copy src to dst without the signals whose type is in ``types``; returns Counter of dropped types."""
    dropped, ids = Counter(), set()
    skipping = False
    with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as out:
        for line in f:
            if skipping:
                if "</signal>" in line:
                    skipping = False
                continue
            s = line.lstrip()
            if s.startswith("<signal ") or s.startswith("<signal\t"):
                m = SIGNAL_OPEN.search(line)
                if m and m.group(2) in types:
                    dropped[m.group(2)] += 1
                    ids.add(m.group(1))
                    if not SELF_CLOSING.search(line.rstrip("\n")) and "</signal>" not in line:
                        skipping = True
                    continue
            if s.startswith("<signalReference") or s.startswith("<control "):
                m = re.search(r'\b(?:id|signalId)="([^"]*)"', line)
                if m and m.group(1) in ids:
                    continue
            out.write(line)
    return dropped


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--types", default="274,205,306", help="signal types to leave out (comma list)")
    args = ap.parse_args(argv)
    types = {t.strip() for t in args.types.split(",") if t.strip()}
    dropped = drop_signals(args.src, args.dst, types)
    print("dropped %d signals: %s" % (sum(dropped.values()), dict(dropped)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
