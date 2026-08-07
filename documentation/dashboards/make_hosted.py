#!/usr/bin/env python3
"""Convert the standalone dcm_master.html (a full HTML document emitted by
master.py) into Artifact-host page-body content: strip the outer
<!doctype>/<html>/<head>/<body>/</html> wrapper, keep the <style> block, the
body content, and the inline chart <script> verbatim. The Artifact host wraps
the result in its own doctype/head/body skeleton and runs the inline JS, so the
canvas charts render on the hosted URL. Fully self-contained (no external refs).

Usage: python3 make_hosted.py  ->  writes dcm_master_hosted.html next to the doc
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(os.path.dirname(HERE)), "dcm_master.html")
DST = os.path.join(HERE, "dcm_master_hosted.html")

html = open(SRC, encoding="utf-8").read()

# keep from the first <style> onward (drops doctype/html/head/meta/title)
i = html.index("<style>")
body = html[i:]
# collapse the head->body boundary and drop the closing document tags
body = body.replace("</style></head><body>", "</style>", 1)
body = body.replace("</body></html>", "", 1).rstrip() + "\n"

open(DST, "w", encoding="utf-8").write(body)
print(f"wrote {DST} ({len(body)/1024:.0f} KB)")
