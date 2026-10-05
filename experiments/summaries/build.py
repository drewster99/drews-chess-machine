#!/usr/bin/env python3
"""Regenerate experiments.js — the index and previous/next data — from the summary pages.

Each summary page is the single source of truth for its own metadata, held in
<meta name="dcm-…" content="…"> tags (README.md lists them). This script reads every
`E-NNNN_YYYY-MM-DD_slug.html` here, validates the metadata, checks that the page's visible
header and takeaway say the same thing, and writes `experiments.js`. It refuses (exit 1,
naming the file and the problem) rather than skipping anything it cannot read: a malformed
page, a dcm- tag written in a form it does not parse, or an HTML file not named as a summary,
so the index can never silently omit or misdescribe one.

Usage: python3 experiments/summaries/build.py
"""
import datetime
import fnmatch
import html
import html.parser
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NOT_SUMMARIES = {"index.html", "TEMPLATE.html"}
NAME = re.compile(r"^(E-\d{4})_(\d{4}-\d{2}-\d{2})_([a-z0-9]+(?:-[a-z0-9]+)*)\.html$")
COMMENT = re.compile(r"<!--.*?-->", re.S)
META_TAG = re.compile(r"<meta\b[^>]*>", re.I)
DCM_NAME = re.compile(r"""\bname\s*=\s*["']?dcm-""", re.I)
META = re.compile(r'<meta\s+name="(dcm-[a-z-]+)"\s+content="([^"]*)"\s*/?>')
TITLE = re.compile(r"<title>(.*?)</title>", re.S)
REQUIRED = ("dcm-id", "dcm-date", "dcm-title", "dcm-status", "dcm-tags", "dcm-takeaway")
OPTIONAL = ("dcm-related", "dcm-superseded-by")
STATUSES = {"complete", "running", "stopped", "superseded"}
TAG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
ID = re.compile(r"^E-\d{4}$")
ASSETS = ('href="style.css"', 'src="experiments.js"', 'src="summaries.js"')


def normalized(text):
    return " ".join(text.split())


def split_list(value):
    """Items of a comma-separated value; an empty value is an empty list, an empty item is kept."""
    return [item.strip() for item in value.split(",")] if value.strip() else []


class PageStructure(html.parser.HTMLParser):
    """The visible text that must repeat the metadata: the header's ID line, heading and tag
    chips, and the takeaway paragraph. Character references arrive already decoded."""

    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}

    def __init__(self):
        super().__init__()
        self.open = []  # (tag, classes, text record or None) per open element
        self.headers = 0
        self.id_spans, self.headings, self.tags, self.takeaways = [], [], [], []
        self.unmatched_end_tags = []

    def within(self, tag, cls):
        return any(t == tag and cls in classes for t, classes, _ in self.open)

    def handle_starttag(self, tag, attrs):
        classes = set((dict(attrs).get("class") or "").split())
        record = None
        if tag == "header" and "exp" in classes:
            self.headers += 1
        elif self.within("header", "exp"):
            parent = self.open[-1]
            if tag == "span" and parent[0] == "div" and "exp-id" in parent[1]:
                record = (classes, [])
                self.id_spans.append(record)
            elif tag == "h1":
                record = (classes, [])
                self.headings.append(record)
            elif tag == "span" and "tag" in classes and self.within("div", "tags"):
                record = (classes, [])
                self.tags.append(record)
        if tag == "p" and "takeaway" in classes:
            record = (classes, [])
            self.takeaways.append(record)
        if tag not in self.VOID:
            self.open.append((tag, classes, record))

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        for depth in range(len(self.open) - 1, -1, -1):
            if self.open[depth][0] == tag:
                del self.open[depth:]
                return
        self.unmatched_end_tags.append(tag)

    def handle_data(self, data):
        for _, _, record in reversed(self.open):
            if record is not None:
                record[1].append(data)
                return


def text_of(records):
    return [normalized("".join(parts)) for _, parts in records]


def structure_problems(name, text, meta, tags):
    """Differences between the page's visible header / takeaway and its metadata."""
    page = PageStructure()
    page.feed(text)
    page.close()
    problems = [f"{name}: </{tag}> closes no open element" for tag in page.unmatched_end_tags]
    if page.headers != 1:
        return problems + [f'{name}: needs exactly one <header class="exp">, found {page.headers}']
    status = meta["dcm-status"]
    expected = [meta["dcm-id"], meta["dcm-date"], status]
    found = text_of(page.id_spans)
    if found != expected:
        problems.append(f"{name}: header .exp-id spans read {found}, metadata gives {expected}")
    elif page.id_spans[2][0] != {"status", status}:
        problems.append(f"{name}: header status span has class '{' '.join(sorted(page.id_spans[2][0]))}', "
                        f"metadata gives 'status {status}'")
    title = normalized(meta["dcm-title"])
    if text_of(page.headings) != [title]:
        problems.append(f"{name}: header <h1> reads {text_of(page.headings)}, dcm-title gives ['{title}']")
    if text_of(page.tags) != tags:
        problems.append(f"{name}: header tag chips read {text_of(page.tags)}, dcm-tags gives {tags}")
    takeaway = normalized(meta["dcm-takeaway"])
    if text_of(page.takeaways) != [takeaway]:
        problems.append(f'{name}: <p class="takeaway"> reads {text_of(page.takeaways)}, '
                        f"dcm-takeaway gives ['{takeaway}']")
    return problems


def read_page(name):
    """Metadata of one page, or a list of the problems found in it."""
    problems = []
    match = NAME.match(name)
    if not match:
        return None, [f"{name}: file name is not E-NNNN_YYYY-MM-DD_slug.html (lowercase slug)"]
    file_id, file_date, _ = match.groups()
    try:
        datetime.date.fromisoformat(file_date)
    except ValueError as error:
        problems.append(f"{name}: {file_date} is not a calendar date ({error})")
    with open(os.path.join(HERE, name), encoding="utf-8") as page:
        text = COMMENT.sub("", page.read())
    meta = {}
    for tag in META_TAG.findall(text):
        if not DCM_NAME.search(tag):
            continue
        parsed = META.fullmatch(tag)
        if not parsed:
            problems.append(f'{name}: cannot parse {tag} — write it as <meta name="dcm-…" content="…">')
            continue
        key, value = parsed.groups()
        if key not in REQUIRED + OPTIONAL:
            problems.append(f"{name}: unknown metadata key {key}")
        if key in meta:
            problems.append(f"{name}: {key} given twice")
        meta[key] = html.unescape(value)
    for key in REQUIRED:
        if not meta.get(key, "").strip():
            problems.append(f"{name}: missing or empty {key}")
    if problems:
        return None, problems
    if meta["dcm-id"] != file_id:
        problems.append(f"{name}: dcm-id {meta['dcm-id']} does not match the file name's {file_id}")
    if meta["dcm-date"] != file_date:
        problems.append(f"{name}: dcm-date {meta['dcm-date']} does not match the file name's {file_date}")
    status = meta["dcm-status"]
    if status not in STATUSES:
        problems.append(f"{name}: dcm-status '{status}' is not one of {sorted(STATUSES)}")
    tags = split_list(meta["dcm-tags"])
    if "" in tags:
        problems.append(f"{name}: dcm-tags has an empty item")
    for t in tags:
        if t and not TAG.match(t):
            problems.append(f"{name}: tag '{t}' is not lowercase-hyphenated")
    for dup in sorted({t for t in tags if t and tags.count(t) > 1}):
        problems.append(f"{name}: tag '{dup}' is listed more than once")
    title_tag = TITLE.search(text)
    expected_title = f"{file_id} · {meta['dcm-title']}"
    if not title_tag or html.unescape(title_tag.group(1).strip()) != expected_title:
        problems.append(f"{name}: <title> must be exactly '{expected_title}'")
    related = split_list(meta.get("dcm-related", ""))
    superseded_by = meta.get("dcm-superseded-by", "").strip()
    if "" in related:
        problems.append(f"{name}: dcm-related has an empty item")
    for ref in [r for r in related if r] + ([superseded_by] if superseded_by else []):
        if not ID.match(ref):
            problems.append(f"{name}: reference '{ref}' is not an E-NNNN id")
        elif ref == file_id:
            problems.append(f"{name}: refers to itself")
    for dup in sorted({r for r in related if r and related.count(r) > 1}):
        problems.append(f"{name}: dcm-related lists {dup} more than once")
    if superseded_by and status != "superseded":
        problems.append(f"{name}: dcm-superseded-by is set but dcm-status is '{status}'")
    if status == "superseded" and not superseded_by:
        problems.append(f"{name}: dcm-status is 'superseded' but dcm-superseded-by is empty")
    for asset in ASSETS:
        if asset not in text:
            problems.append(f"{name}: does not load {asset}")
    problems += structure_problems(name, text, meta, tags)
    if problems:
        return None, problems
    return {
        "id": file_id, "date": file_date, "file": name, "title": meta["dcm-title"],
        "status": status, "tags": tags, "takeaway": meta["dcm-takeaway"],
        "related": related, "supersededBy": superseded_by or None,
    }, []


def main():
    names = sorted(n for n in os.listdir(HERE)
                   if fnmatch.fnmatch(n.lower(), "*.htm*") and n not in NOT_SUMMARIES)
    entries, problems = [], []
    for name in names:
        entry, page_problems = read_page(name)
        problems += page_problems
        if entry:
            entries.append(entry)
    ids = [e["id"] for e in entries]
    for dup in sorted({i for i in ids if ids.count(i) > 1}):
        problems.append(f"{dup} is used by more than one page")
    known = {match.group(1) for match in map(NAME.match, names) if match}
    for e in entries:
        for ref in e["related"] + ([e["supersededBy"]] if e["supersededBy"] else []):
            if ref not in known:
                problems.append(f"{e['file']}: references {ref}, which has no page")
    if problems:
        print("\n".join(problems), file=sys.stderr)
        sys.exit(1)
    entries.sort(key=lambda e: e["id"])
    body = json.dumps(entries, ensure_ascii=False, indent=1)
    with open(os.path.join(HERE, "experiments.js"), "w", encoding="utf-8") as out:
        out.write("// Generated by build.py from the summary pages' <meta> tags. Do not edit.\n")
        out.write(f"window.DCM_EXPERIMENTS = {body};\n")
    print(f"experiments.js: {len(entries)} experiments")


if __name__ == "__main__":
    main()
