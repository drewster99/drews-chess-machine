# Experiment summaries

One page per experiment: what was asked, how, what came out, what it means, and where the
full data lives. The experiment folders keep the long write-ups and the data; these pages
are the short version a reader can scan in a minute and click through.

Open `index.html` in a browser (it works from the local file, no server): latest ID first,
with a search box and tag filters. Every page has previous / next / index links.

## Naming

`E-NNNN_YYYY-MM-DD_slug.html`

- `E-NNNN` — global, four-digit, assigned in the order summaries are written, never reused.
  Cite experiments by it ("E-0012"). A gap in the sequence is an ID not yet written (reserved
  for a write-up in progress) or withdrawn; it is never handed to another experiment.
- `YYYY-MM-DD` — when the experiment ran (its first day). A summary written later for an
  older experiment keeps its own date, so numbers and dates need not sort the same way.
- `slug` — lowercase words joined by hyphens.

## Writing a page

Copy `TEMPLATE.html`. Its header (ID, date, status, `<h1>`, tag chips in order) and its
takeaway paragraph repeat the metadata word for word — `build.py` refuses a page where they
differ (compared after decoding entities and collapsing whitespace). Sections, in order:

1. **Question** — the impetus and hypothesis.
2. **Setup** — exact identities: run names and model IDs, checkpoints and steps, build and
   git hash, corpus, probe set, method. "R7 at 33k" is ambiguous a month later; the model ID
   and build are not.
3. **Findings** — whatever presents the result best: one or two labeled tables, a chart, or a
   few sentences. Charts are declared inline as JSON (see the template) and drawn by
   `summaries.js`; a missing value is `null` and shows as a gap, never interpolated (a point
   with no neighbour on either side shows as a dot). `y.min` / `y.max` are optional and
   independent; an end that is not given gets a little padding. A chart with no values, a value
   outside `y.min`…`y.max`, or a `vlines` entry outside the data's x range is not drawn: the
   figure shows the error instead.
4. **Interpretation** — a few sentences, including how sure we are and what it does not show.
5. **Takeaway** — one sentence a reader can act on; exactly the `dcm-takeaway` text, which the
   index shows.
6. **Artifacts** — links to the experiment folder (write-up, data, scripts, Reproduce section).
   Anything a summary relies on lives in the repo, not in a scratch folder.
7. **Related** — filled from `dcm-related`.

Aim for one page. Put depth in the artifacts.

## Metadata

The page's `<meta>` tags are the only source of its index entry, its related list and its
superseded banner. Write each exactly as `<meta name="dcm-…" content="…">` (name first, double
quotes, `&quot;` for a quote in the value); a `dcm-` tag in any other form, or a key not listed
here, is refused. Commented-out markup is ignored.

| tag | content |
|---|---|
| `dcm-id` | `E-NNNN`, same as the file name |
| `dcm-date` | `YYYY-MM-DD`, same as the file name; a real calendar date |
| `dcm-title` | the title; `<title>` must be exactly `E-NNNN · <title>` |
| `dcm-status` | `complete`, `running`, `stopped` or `superseded` |
| `dcm-tags` | comma-separated, lowercase-hyphenated; at least one, no empty items, no repeats |
| `dcm-takeaway` | one sentence |
| `dcm-related` | optional, comma-separated `E-NNNN` ids of existing pages; no empty items, no repeats, not the page itself |
| `dcm-superseded-by` | `E-NNNN` of an existing page, not the page itself; set exactly when the status is `superseded`, and the page shows a banner |

When a later experiment overturns a conclusion, mark the old page `superseded` with
`dcm-superseded-by`, rather than editing its findings away.

## Index

After adding or changing a page, run

    python3 experiments/summaries/build.py

It validates every page (name, metadata, `<title>`, header and takeaway against the
metadata, references, and that it loads `style.css`, `experiments.js` and `summaries.js`) and
rewrites `experiments.js`. It refuses, naming the page and the problem, instead of skipping one —
including any `.htm*` file here other than `index.html` and `TEMPLATE.html` that is not named
`E-NNNN_YYYY-MM-DD_slug.html`. A page missing from `experiments.js`, or a missing
`experiments.js`, shows a "run build.py" warning in the browser.
Commit `experiments.js` with the pages.
