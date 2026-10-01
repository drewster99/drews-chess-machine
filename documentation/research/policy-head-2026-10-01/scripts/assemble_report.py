#!/usr/bin/env python3
"""Fill the generated-table blocks of REPORT.md in place.

REPORT.md marks each generated table with
    <!-- begin:NAME -->
    ...
    <!-- end:NAME -->
and this script replaces everything between the markers with
results/tables/NAME.md. The narrative around the markers is left untouched,
so the report can be re-rendered after `run_all.py`, `findings.py` and
`render_tables.py` are re-run.
"""
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(SCRIPT_DIR, "..", "REPORT.md")
TABLES = os.path.join(SCRIPT_DIR, "..", "results", "tables")

BLOCK = re.compile(r"(<!-- begin:([a-z_]+) -->\n)(.*?)(<!-- end:\2 -->)", re.DOTALL)


def main():
    text = open(REPORT).read()
    missing = []

    def fill(match):
        name = match.group(2)
        path = os.path.join(TABLES, name + ".md")
        if not os.path.exists(path):
            missing.append(name)
            return match.group(0)
        return match.group(1) + "\n" + open(path).read().rstrip("\n") + "\n\n" + match.group(4)

    text = BLOCK.sub(fill, text)
    if missing:
        sys.exit(f"missing generated tables: {', '.join(missing)}")
    open(REPORT, "w").write(text)
    print("REPORT.md assembled")


if __name__ == "__main__":
    main()
