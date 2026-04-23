"""Render the Pinecone + OpenAI RAG cheat sheet as SVG and PNG.

Dense, DataCamp-style one-pager: pink banner, three columns of syntax sections,
each section a dark header bar + REPL-style `>>>` lines with green `# comments`.

Run:
    python scripts/render_cheatsheet.py

Outputs:
    assets/cheatsheet.svg  (source of truth, embeddable in notebooks)
    assets/cheatsheet.png  (for the README)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

ASSETS = Path(__file__).resolve().parent.parent / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

# --- palette -------------------------------------------------------------
BG = "#faf7f2"              # page cream
BANNER = "#ff3d7f"          # DataCamp-ish pink
BANNER_INK = "#ffffff"
HEADER = "#1d1f2b"          # near-black for section header bars
HEADER_INK = "#ffffff"
INK = "#1d1f2b"
MUTED = "#6a7180"
SUBTLE = "#e7e3db"
CODE_FG = "#1d1f2b"
PROMPT = "#d92d6a"          # pink-ish for the >>> prompts, echoes banner
COMMENT = "#2f9e44"         # green for inline comments
STRING = "#7a4ec7"          # purple for string literals (sparingly)

# --- layout primitives ---------------------------------------------------

PAGE_W, PAGE_H = 22.0, 16.5     # axes units; figure size is scaled from this
MARGIN = 0.35
BANNER_H = 1.35
FOOTER_H = 0.35
COL_GAP = 0.35


def draw_banner(ax):
    y0 = PAGE_H - BANNER_H
    ax.add_patch(Rectangle((0, y0), PAGE_W, BANNER_H, facecolor=BANNER, edgecolor="none"))
    ax.text(MARGIN + 0.15, PAGE_H - 0.45, "rag-unpacked",
            color=BANNER_INK, fontsize=11, fontweight="bold", va="top",
            family="DejaVu Sans")
    ax.text(MARGIN + 0.15, PAGE_H - 0.80, "01. Intro to RAG",
            color=BANNER_INK, fontsize=9.5, va="top", family="DejaVu Sans")
    ax.text(PAGE_W - MARGIN - 0.15, PAGE_H - 0.35,
            "Pinecone + OpenAI",
            color=BANNER_INK, fontsize=14, fontweight="bold", va="top", ha="right",
            family="DejaVu Sans")
    ax.text(PAGE_W - MARGIN - 0.15, PAGE_H - 0.90,
            "RAG Cheat Sheet",
            color=BANNER_INK, fontsize=20, fontweight="bold", va="top", ha="right",
            family="DejaVu Sans")


def draw_footer(ax):
    ax.text(MARGIN, 0.18, "github.com/allllc/rag-unpacked",
            color=MUTED, fontsize=7.8, va="center", family="DejaVu Sans")
    ax.text(PAGE_W - MARGIN, 0.18,
            "module 01  ·  intro-to-rag  ·  Pinecone SDK 7 + OpenAI",
            color=MUTED, fontsize=7.8, va="center", ha="right", family="DejaVu Sans")


class Column:
    """A vertical cursor that lays out section blocks top-down in a column."""

    def __init__(self, x, width):
        self.x = x
        self.width = width
        self.y = PAGE_H - BANNER_H - 0.3   # start just below banner

    def advance(self, dy):
        self.y -= dy


def draw_section_header(ax, col, title):
    h = 0.38
    y = col.y - h
    ax.add_patch(
        FancyBboxPatch(
            (col.x, y), col.width, h,
            boxstyle="round,pad=0.0,rounding_size=0.05",
            linewidth=0, facecolor=HEADER,
        )
    )
    ax.text(col.x + 0.15, y + h / 2, "▸", color=BANNER, fontsize=10,
            va="center", family="DejaVu Sans", fontweight="bold")
    ax.text(col.x + 0.42, y + h / 2, title, color=HEADER_INK, fontsize=10.0,
            va="center", fontweight="bold", family="DejaVu Sans")
    col.advance(h + 0.08)


def draw_subheader(ax, col, text):
    """A small caption between a section header and its first code line."""
    ax.text(col.x + 0.1, col.y - 0.16, text, color=MUTED,
            fontsize=7.8, va="top", style="italic", family="DejaVu Sans")
    col.advance(0.24)


def _render_code_line(ax, x, y, line):
    """Render a single monospace code line with token colouring.

    Colours:
      >>> / ...     prompt  (PROMPT)
      # comment     comment (COMMENT, italic)
      everything    code    (CODE_FG)

    Inline trailing comments are intentionally not supported. Put the
    comment on its own preceding line instead; it looks cleaner and avoids
    font-width measurement headaches.
    """
    if line.startswith(">>> "):
        ax.text(x, y, ">>> ", color=PROMPT, fontsize=8.4, family="monospace",
                va="top", fontweight="bold")
        rest = line[4:]
        x_rest = x + 0.33
    elif line.startswith("... "):
        ax.text(x, y, "... ", color=PROMPT, fontsize=8.4, family="monospace",
                va="top")
        rest = line[4:]
        x_rest = x + 0.33
    else:
        rest = line
        x_rest = x

    if rest.lstrip().startswith("#"):
        ax.text(x_rest, y, rest, color=COMMENT, fontsize=8.4, family="monospace",
                va="top", style="italic")
        return

    ax.text(x_rest, y, rest, color=CODE_FG, fontsize=8.4, family="monospace",
            va="top")


def draw_code_block(ax, col, lines):
    """Render lines as code, with a subtle left accent strip."""
    line_h = 0.22
    block_h = line_h * len(lines) + 0.10
    top = col.y
    bottom = top - block_h
    # subtle tinted background band
    ax.add_patch(Rectangle((col.x + 0.05, bottom + 0.02),
                           col.width - 0.1, block_h - 0.04,
                           facecolor="#f1eee6", edgecolor="none"))
    # accent strip
    ax.add_patch(Rectangle((col.x + 0.05, bottom + 0.02), 0.06, block_h - 0.04,
                           facecolor=BANNER, edgecolor="none"))

    x = col.x + 0.22
    y = top - 0.12
    for line in lines:
        _render_code_line(ax, x, y, line)
        y -= line_h
    col.advance(block_h + 0.10)


def draw_section(ax, col, title, lines, subtitle=None):
    draw_section_header(ax, col, title)
    if subtitle:
        draw_subheader(ax, col, subtitle)
    draw_code_block(ax, col, lines)


# --- content -------------------------------------------------------------

def content_column_1(ax, col):
    draw_section(ax, col, "Setup", [
        "# .env  (gitignored)",
        "PINECONE_API_KEY=pcsk_...",
        "OPENAI_API_KEY=sk-...",
        "",
        ">>> from helpers import load_env, \\",
        "...     get_pinecone_client, get_openai_client",
        ">>> cfg = load_env()",
        ">>> pc = get_pinecone_client(cfg)",
        ">>> client = get_openai_client(cfg)",
    ])

    draw_section(ax, col, "Client & Index", [
        ">>> from pinecone import Pinecone, ServerlessSpec",
        ">>> pc = Pinecone(api_key=\"pcsk_...\")",
        "",
        "# Create a serverless index (dim must match model)",
        ">>> pc.create_index(",
        "...     name=\"my-index\",",
        "...     dimension=1536,",
        "...     metric=\"cosine\",",
        "...     spec=ServerlessSpec(",
        "...         cloud=\"aws\", region=\"us-east-1\"))",
        "",
        "# Idempotent helper: create if missing",
        ">>> from helpers import ensure_index",
        ">>> index = ensure_index(pc, \"my-index\", 1536)",
        "",
        "# Delete an index",
        ">>> if pc.has_index(\"my-index\"):",
        "...     pc.delete_index(\"my-index\")",
    ])

    draw_section(ax, col, "Inspect", [
        "# Cheap existence check (True / False)",
        ">>> pc.has_index(\"my-index\")",
        "",
        "# All indexes in the project",
        ">>> pc.list_indexes()",
        "",
        "# Metadata for one index",
        ">>> pc.describe_index(\"my-index\")",
        "",
        "# Live stats on a connected index",
        ">>> index.describe_index_stats()",
        "# -> {'total_vector_count': 20,",
        "#     'dimension': 1536, ...}",
    ])

    draw_section(ax, col, "Upsert", [
        "# Records are dicts with id, values, metadata",
        ">>> index.upsert(vectors=[",
        "...   {\"id\": \"a\", \"values\": [...],",
        "...    \"metadata\": {\"year\": 2024}},",
        "...   {\"id\": \"b\", \"values\": [...]},",
        "... ])",
        "",
        "# Same call with a namespace",
        ">>> index.upsert(vectors=recs,",
        "...              namespace=\"prod\")",
    ])


def content_column_2(ax, col):
    draw_section(ax, col, "Querying", [
        ">>> index.query(",
        "...     vector=query_vec,",
        "...     top_k=5,",
        "...     include_metadata=True)",
        "",
        "# Response shape",
        "# {'matches': [",
        "#    {'id': 'a', 'score': 0.87,",
        "#     'metadata': {...}}, ...]}",
    ])

    draw_section(ax, col, "Metadata Filters", [
        "# Operators: $eq $ne $gt $gte $lt $lte",
        "#            $in $nin $exists",
        "",
        "# Single key",
        ">>> index.query(vector=v, top_k=5,",
        "...   filter={\"genre\": \"thriller\"})",
        "",
        "# Compound (implicit AND)",
        ">>> filt = {\"genre\": \"thriller\",",
        "...         \"year\": {\"$lt\": 2018}}",
        "",
        "# $in across multiple values",
        ">>> filt = {\"genre\":",
        "...   {\"$in\": [\"sci-fi\", \"thriller\"]}}",
        "",
        "# Explicit OR",
        ">>> filt = {\"$or\": [",
        "...   {\"genre\": \"sci-fi\"},",
        "...   {\"rating\": {\"$gte\": 8.0}}]}",
    ], subtitle="Filter runs before similarity ranking")

    draw_section(ax, col, "Fetch / Update", [
        "# Fetch by id (no similarity)",
        ">>> index.fetch(ids=[\"a\", \"b\"])",
        "",
        "# Replace values, keep metadata",
        ">>> index.update(id=\"a\", values=new_vec)",
        "",
        "# Merge metadata, keep vector",
        ">>> index.update(id=\"a\",",
        "...   set_metadata={\"rating\": 9.1})",
    ])

    draw_section(ax, col, "Deleting", [
        "# By id (targeted, safe)",
        ">>> index.delete(ids=[\"a\", \"b\"])",
        "",
        "# By filter (careful!)",
        ">>> index.delete(",
        "...   filter={\"genre\": \"horror\"})",
        "",
        "# Nuke a whole namespace",
        ">>> index.delete(delete_all=True,",
        "...              namespace=\"staging\")",
    ])

    draw_section(ax, col, "Namespaces", [
        "# Logical partitions inside ONE index",
        ">>> index.upsert(vectors=recs,",
        "...              namespace=\"tenant_42\")",
        ">>> index.query(vector=v, top_k=5,",
        "...             namespace=\"tenant_42\")",
    ])


def content_column_3(ax, col):
    draw_section(ax, col, "Performance", [
        "# Batch: chunks helper",
        ">>> from helpers import chunks",
        ">>> for batch in chunks(recs, 100):",
        "...     index.upsert(vectors=list(batch))",
        "",
        "# Parallel upsert with pool_threads",
        ">>> with pc.Index(name,",
        "...               pool_threads=20) as idx:",
        "...     futs = [idx.upsert(",
        "...         vectors=list(b),",
        "...         async_req=True)",
        "...         for b in chunks(recs, 100)]",
        "...     for f in futs: f.get()",
    ], subtitle="~100 vectors / request, 2 MB payload cap")

    draw_section(ax, col, "Embedding (OpenAI)", [
        "# text -> 1536-dim vector",
        ">>> resp = client.embeddings.create(",
        "...     input=[\"what is a vector DB?\"],",
        "...     model=\"text-embedding-3-small\")",
        ">>> query_vec = resp.data[0].embedding",
    ])

    draw_section(ax, col, "Generating (OpenAI)", [
        "# temperature=0 for grounded RAG answers",
        ">>> client.chat.completions.create(",
        "...   model=\"gpt-4o-mini\",",
        "...   messages=msgs,",
        "...   temperature=0)",
    ])

    draw_section(ax, col, "The RAG Loop", [
        "# 1. Embed the question",
        ">>> qv = embed(query)",
        "",
        "# 2. Retrieve top-k",
        ">>> res = index.query(vector=qv,",
        "...    top_k=5, include_metadata=True)",
        "",
        "# 3. Augment: build the prompt",
        ">>> ctx = \"\\n\".join(m.metadata[\"text\"]",
        "...                 for m in res.matches)",
        ">>> msgs = [",
        "...   {\"role\":\"system\", \"content\": SYS},",
        "...   {\"role\":\"user\",",
        "...    \"content\": f\"{ctx}\\n\\nQ: {q}\"}]",
        "",
        "# 4. Generate (see above)",
    ], subtitle="Embed -> Retrieve -> Augment -> Generate")


def render(out_svg: Path = ASSETS / "cheatsheet.svg",
           out_png: Path = ASSETS / "cheatsheet.png") -> tuple[Path, Path]:
    # Match figure aspect to page so `set_aspect("equal")` doesn't leave gutters.
    in_w = 15.4
    in_h = in_w * (PAGE_H / PAGE_W)
    fig, ax = plt.subplots(figsize=(in_w, in_h), facecolor=BG)
    ax.set_xlim(0, PAGE_W)
    ax.set_ylim(0, PAGE_H)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])  # fill the figure, no padding

    ax.add_patch(Rectangle((0, 0), PAGE_W, PAGE_H, facecolor=BG, edgecolor="none"))

    draw_banner(ax)

    col_width = (PAGE_W - 2 * MARGIN - 2 * COL_GAP) / 3
    cols = [
        Column(MARGIN + i * (col_width + COL_GAP), col_width)
        for i in range(3)
    ]

    content_column_1(ax, cols[0])
    content_column_2(ax, cols[1])
    content_column_3(ax, cols[2])

    draw_footer(ax)

    fig.savefig(out_svg, format="svg", facecolor=BG, pad_inches=0)
    fig.savefig(out_png, format="png", facecolor=BG, dpi=170, pad_inches=0)
    plt.close(fig)
    return out_svg, out_png


if __name__ == "__main__":
    svg, png = render()
    print(f"Wrote {svg}")
    print(f"Wrote {png}")
