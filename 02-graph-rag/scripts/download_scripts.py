"""Fetch movie scripts from IMSDb and cache them locally.

Downloads each of the 11 movies' shooting drafts from imsdb.com and
writes the plaintext to data/scripts/<slug>.txt. The data/ folder is
gitignored, so the scripts never ride along in the repo: each user
fetches them once.

Usage:
    python scripts/download_scripts.py            # fetch missing only
    python scripts/download_scripts.py --force    # re-fetch everything

After running this, rebuild the Kuzu DB so the Script nodes get
populated:
    python scripts/build_corpus.py --force

About the source: IMSDb has hosted these for ~20 years as the
de-facto public archive. We're attaching real text to a teaching
graph, not redistributing it: the cache stays local. If you're
deploying anything based on this, do your own clearance.
"""
from __future__ import annotations

import re
import sys
import time
import urllib.error
import urllib.request
from html import unescape
from pathlib import Path


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
SCRIPTS_DIR = DATA_DIR / "scripts"

USER_AGENT = "rag-unpacked/0.2 (educational; +https://github.com/allllc/rag-unpacked)"
REQUEST_TIMEOUT = 20
DELAY_BETWEEN_SECONDS = 1.0  # be polite

# Mirrors the MOVIES list in build_corpus.py: (title, imsdb_slug).
# Kept duplicated to avoid an import-time dependency on Kuzu.
SCRIPTS: list[tuple[str, str]] = [
    ("Pulp Fiction", "Pulp-Fiction"),
    ("Reservoir Dogs", "Reservoir-Dogs"),
    ("Django Unchained", "Django-Unchained"),
    ("The Matrix", "Matrix,-The"),
    ("The Matrix Reloaded", "Matrix-Reloaded,-The"),
    ("John Wick", "John-Wick"),
    ("Memento", "Memento"),
    ("The Prestige", "Prestige,-The"),
    ("Inception", "Inception"),
    ("Get Out", "Get-Out"),
    ("Eternal Sunshine of the Spotless Mind", "Eternal-Sunshine-of-the-Spotless-Mind"),
]


def _url_for(slug: str) -> str:
    return f"https://imsdb.com/scripts/{slug}.html"


def _output_path(slug: str) -> Path:
    return SCRIPTS_DIR / f"{slug}.txt"


def _fetch_html(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        return resp.read().decode("utf-8", errors="replace")


_SCRIPT_OPEN_RE = re.compile(r'<td\s+class="scrtext"\s*>', re.IGNORECASE)
_PRE_OPEN_RE = re.compile(r"<pre>", re.IGNORECASE)
_PRE_CLOSE_RE = re.compile(r"</pre>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")


def _extract_script(html: str, source_url: str) -> str:
    """Pull the script body out of an IMSDb script page.

    Expected shape: <td class="scrtext"><pre>...script lines...</pre></td>.
    If the page doesn't match (404 landing, layout change), raise.
    """
    td_match = _SCRIPT_OPEN_RE.search(html)
    if not td_match:
        raise ValueError(
            f"Could not find '<td class=scrtext>' container at {source_url}. "
            "Either IMSDb's layout changed or this page is a stub/404 landing."
        )
    pre_open = _PRE_OPEN_RE.search(html, td_match.end())
    if not pre_open:
        raise ValueError(f"No <pre> after script container at {source_url}.")
    pre_close = _PRE_CLOSE_RE.search(html, pre_open.end())
    if not pre_close:
        raise ValueError(f"No </pre> closing tag at {source_url}.")

    body = html[pre_open.end():pre_close.start()]
    body = _TAG_RE.sub("", body)          # strip <b>, <i>, anchors, etc.
    body = unescape(body)                  # &amp; -> &, &nbsp; -> space, etc.
    # Collapse Windows line endings, leave the rest alone (preserves
    # screenplay formatting: indentation, blank lines for scene breaks).
    body = body.replace("\r\n", "\n").replace("\r", "\n")
    return body.strip()


def _download_one(title: str, slug: str, force: bool) -> tuple[str, str]:
    """Download one script. Returns (status, detail) for logging."""
    out = _output_path(slug)
    if out.exists() and not force:
        return ("skip", f"{title} (cached, {out.stat().st_size:,} bytes)")
    url = _url_for(slug)
    try:
        html = _fetch_html(url)
        text = _extract_script(html, url)
    except urllib.error.HTTPError as e:
        return ("fail", f"{title}: HTTP {e.code} at {url}")
    except ValueError as e:
        return ("fail", f"{title}: {e}")
    except Exception as e:
        return ("fail", f"{title}: {type(e).__name__}: {e}")
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return ("ok", f"{title} ({len(text):,} chars -> {out.name})")


def download_all(force: bool = False) -> dict[str, list[str]]:
    """Fetch every script. Returns a dict bucketed by status."""
    summary: dict[str, list[str]] = {"ok": [], "skip": [], "fail": []}
    for i, (title, slug) in enumerate(SCRIPTS):
        status, detail = _download_one(title, slug, force=force)
        summary[status].append(detail)
        print(f"  [{status:4s}] {detail}")
        # Polite delay only when we actually fetched
        if status == "ok" and i < len(SCRIPTS) - 1:
            time.sleep(DELAY_BETWEEN_SECONDS)
    return summary


def main() -> int:
    force = "--force" in sys.argv
    print(f"Downloading {len(SCRIPTS)} scripts to {SCRIPTS_DIR}")
    print(f"(force={force})")
    print()
    summary = download_all(force=force)
    print()
    print(
        f"Done: {len(summary['ok'])} fetched, "
        f"{len(summary['skip'])} skipped, {len(summary['fail'])} failed"
    )
    if summary["fail"]:
        print()
        print("Failures (re-run with --force to retry):")
        for line in summary["fail"]:
            print(f"  - {line}")
        return 1
    print()
    print("Next step: rebuild the Kuzu graph to load these into Script nodes.")
    print("    python scripts/build_corpus.py --force")
    return 0


if __name__ == "__main__":
    sys.exit(main())
