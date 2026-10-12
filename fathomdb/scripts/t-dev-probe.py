"""Helper process of t-dev-lib-major.sh and the g8-*.sh scripts.

Usage: python3 -I t-dev-probe.py PATTERN NAME...

dlopens each NAME in order (a leading "!" means the name must NOT open),
then prints the mapped files whose path contains PATTERN. Exit status 1 if
any name opened or failed against expectation.
"""

import ctypes
import os
import sys


def main(argv):
    pattern, names = argv[0], argv[1:]
    ok = True
    for n in names:
        want = not n.startswith("!")
        n = n.lstrip("!")
        try:
            ctypes.CDLL(n, mode=os.RTLD_NOW)
            opened = True
        except OSError:
            opened = False
        print(f"HELPER dlopen({n}) {'opened' if opened else 'failed'}")
        ok &= opened == want
    with open("/proc/self/maps") as f:
        paths = sorted({line.split()[-1] for line in f if pattern in line})
    for p in paths:
        print(f"HELPER maps {p}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
