"""Render the BM25 + Hybrid Vectorless RAG cheat sheet as SVG and PNG.

Same DataCamp-style one-pager as modules 01 and 02: pink banner, three
columns of syntax sections, dark header bars, REPL-style `>>>` lines,
green `# comments`. Layout primitives are lifted verbatim from module
02's cheatsheet renderer.

Run:
    python scripts/render_cheatsheet.py

Outputs:
    assets/cheatsheet.svg
    assets/cheatsheet.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

ASSETS = Path(__file__).resolve().parent.parent / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

# --- palette -------------------------------------------------------------
BG = "#faf7f2"
BANNER = "#ff3d7f"
BANNER_INK = "#ffffff"
HEADER = "#1d1f2b"
HEADER_INK = "#ffffff"
INK = "#1d1f2b"
MUTED = "#6a7180"
SUBTLE = "#e7e3db"
CODE_FG = "#1d1f2b"
PROMPT = "#d92d6a"
COMMENT = "#2f9e44"
STRING = "#7a4ec7"

# --- layout primitives ---------------------------------------------------

PAGE_W, PAGE_H = 22.0, 18.0
MARGIN = 0.35
BANNER_H = 1.35
COL_GAP = 0.35


def draw_banner(ax):
    y0 = PAGE_H - BANNER_H
    ax.add_patch(Rectangle((0, y0), PAGE_W, BANNER_H, facecolor=BANNER, edgecolor="none"))
    ax.text(MARGIN + 0.15, PAGE_H - 0.45, "rag-unpacked",
            color=BANNER_INK, fontsize=11, fontweight="bold", va="top",
            family="DejaVu Sans")
    ax.text(MARGIN + 0.15, PAGE_H - 0.80, "03. Vectorless RAG",
            color=BANNER_INK, fontsize=9.5, va="top", family="DejaVu Sans")
    ax.text(PAGE_W - MARGIN - 0.15, PAGE_H - 0.35,
            "BM25 + Hybrid",
            color=BANNER_INK, fontsize=14, fontweight="bold", va="top", ha="right",
            family="DejaVu Sans")
    ax.text(PAGE_W - MARGIN - 0.15, PAGE_H - 0.90,
            "Vectorless RAG Cheat Sheet",
            color=BANNER_INK, fontsize=20, fontweight="bold", va="top", ha="right",
            family="DejaVu Sans")


def draw_footer(ax):
    ax.text(MARGIN, 0.18, "github.com/allllc/rag-unpacked",
            color=MUTED, fontsize=7.8, va="center", family="DejaVu Sans")
    ax.text(PAGE_W - MARGIN, 0.18,
            "module 03  ·  vectorless-rag  ·  NB1-NB3  ·  bm25s 0.3 + OpenAI",
            color=MUTED, fontsize=7.8, va="center", ha="right", family="DejaVu Sans")


class Column:
    def __init__(self, x, width):
        self.x = x
        self.width = width
        self.y = PAGE_H - BANNER_H - 0.3

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
    ax.text(col.x + 0.1, col.y - 0.16, text, color=MUTED,
            fontsize=7.8, va="top", style="italic", family="DejaVu Sans")
    col.advance(0.24)


def _render_code_line(ax, x, y, line):
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
    line_h = 0.22
    block_h = line_h * len(lines) + 0.10
    top = col.y
    bottom = top - block_h
    ax.add_patch(Rectangle((col.x + 0.05, bottom + 0.02),
                           col.width - 0.1, block_h - 0.04,
                           facecolor="#f1eee6", edgecolor="none"))
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
        "# .env  (gitignored, NB3 only)",
        "OPENAI_API_KEY=sk-...",
        "# Optional, for NB3 head-to-head:",
        "# PINECONE_API_KEY=pcsk-...",
        "",
        ">>> from helpers import load_env, \\",
        "...     get_openai_client, build_bm25_index, \\",
        "...     retrieve, rrf_combine",
        ">>> cfg = load_env()",
        ">>> client = get_openai_client(cfg)  # NB3 only",
    ])

    draw_section(ax, col, "BM25 in 5 lines", [
        ">>> import bm25s",
        ">>> tokens = bm25s.tokenize(docs)",
        ">>> m = bm25s.BM25()",
        ">>> m.index(tokens)",
        "",
        ">>> q = bm25s.tokenize(\"query string\")",
        ">>> ids, scores = m.retrieve(q, k=5)",
        "",
        "# Or use the helper (1 line each):",
        ">>> retr, _ = build_bm25_index(docs)",
        ">>> ids, scores = retrieve(retr, \"query\", k=5)",
    ], subtitle="Raw API and the helper wrapper")

    draw_section(ax, col, "Tokenization", [
        "# Default: lowercase, English stopwords",
        ">>> bm25s.tokenize(docs)",
        "",
        "# Disable stopwords (code/log search)",
        ">>> bm25s.tokenize(docs, stopwords=None)",
        "",
        "# Stem with PyStemmer",
        ">>> import Stemmer",
        ">>> st = Stemmer.Stemmer(\"english\")",
        ">>> bm25s.tokenize(docs, stemmer=st)",
    ])

    draw_section(ax, col, "Scoring intuition", [
        "# score(doc) = IDF * TF_sat * len_norm",
        "#",
        "# IDF      : rare query words count more",
        "# TF_sat   : repeated matches help, with",
        "#            diminishing returns",
        "# len_norm : shorter docs win ties",
    ], subtitle="The whole algorithm, in 6 lines")


def content_column_2(ax, col):
    draw_section(ax, col, "Parameters (k1, b)", [
        "# Defaults are good. Rarely need to tune.",
        ">>> bm25s.BM25(k1=1.5, b=0.75)",
        "",
        "# k1 : TF saturation",
        "#      higher -> more reward for repeats",
        "#      lower  -> saturates faster",
        "",
        "# b  : length normalization",
        "#      0    -> ignore doc length",
        "#      1    -> fully normalize",
    ])

    draw_section(ax, col, "Phrase queries", [
        "# BM25 ignores word order: same scores",
        "# for 'sort list dict' and 'dict list sort'",
        "",
        "# If order matters, use a phrase-aware",
        "# retriever or post-filter candidates.",
    ], subtitle="Bag of tokens, not phrases")

    draw_section(ax, col, "Metadata filtering", [
        "# bm25s has no built-in filter.",
        "# Retrieve more, filter in pandas.",
        ">>> ids, scores = retrieve(retr, q, k=20)",
        ">>> top = pd.DataFrame({",
        "...   \"id\": [df.iloc[i][\"id\"] for i in ids],",
        "...   \"title\": [df.iloc[i][\"title\"] for i in ids],",
        "...   \"score\": scores})",
        ">>> top[top[\"title\"].str.contains(\"dict\")]",
    ])

    draw_section(ax, col, "Persistence", [
        ">>> retr.save(\"data/bm25_index\")",
        ">>> reloaded = bm25s.BM25.load(",
        "...   \"data/bm25_index\", load_corpus=False)",
        "# Same scores; index round-trips cleanly",
    ])

    draw_section(ax, col, "When BM25 wins / loses", [
        "# WINS on rare distinctive tokens:",
        "#   __slots__, pool_threads, $in,",
        "#   error codes, version numbers, IDs",
        "",
        "# LOSES on synonyms / paraphrase:",
        "#   query 'make my code faster'",
        "#   misses doc 'speed up loops'",
        "#   (no shared non-stopword tokens)",
    ])


def content_column_3(ax, col):
    draw_section(ax, col, "The Vectorless RAG Loop", [
        "# Same 4-step shape as vector RAG",
        "# (module 01); only step 1 changed.",
        "#",
        "#   1. Tokenize the query",
        "#   2. Retrieve top-k via BM25",
        "#   3. Augment a grounded prompt",
        "#   4. Generate the answer (LLM)",
        "",
        ">>> ids, _ = retrieve(retr, q, k=3)",
        ">>> ctx = \"\\n\\n\".join(",
        "...   df.iloc[i][\"text\"] for i in ids)",
        ">>> client.chat.completions.create(",
        "...   model=\"gpt-4o-mini\", temperature=0,",
        "...   messages=[{\"role\":\"user\",",
        "...     \"content\": f\"Ctx:\\n{ctx}\\n\\nQ:{q}\"}])",
    ], subtitle="Tokenize -> Retrieve -> Augment -> Generate")

    draw_section(ax, col, "Hybrid via RRF", [
        "# Reciprocal Rank Fusion combines ranks,",
        "# not scores. k=60 is the standard pick.",
        "#",
        "# For each id, sum 1/(k + rank) across",
        "# every retriever's list. Sort descending.",
        "",
        ">>> def rrf(rankings, k=60):",
        "...   fused = {}",
        "...   for r in rankings:",
        "...     for rank, d in enumerate(r, 1):",
        "...       fused[d] = fused.get(d, 0) \\",
        "...         + 1 / (k + rank)",
        "...   return sorted(fused.items(),",
        "...     key=lambda x: x[1], reverse=True)",
        "",
        "# Or use the helper:",
        ">>> rrf_combine([bm25_ids, vector_ids])",
    ], subtitle="2 extra lines on top of two retrievers")

    draw_section(ax, col, "When to choose what", [
        "# BM25 alone:",
        "#   small corpus, keyword-shaped queries,",
        "#   latency-sensitive, no embedding budget",
        "",
        "# Vector alone:",
        "#   paraphrased queries, conceptual search,",
        "#   synonyms dominate your traffic",
        "",
        "# Hybrid (BM25 + vector + RRF):",
        "#   robust across query shapes,",
        "#   production default for most apps",
    ])

    draw_section(ax, col, "The portfolio takeaway", [
        "# BM25 is the baseline you should beat.",
        "# Vector handles the paraphrases BM25 misses.",
        "# Hybrid is a robustness move, not a quality",
        "# move: it combines ranks, not correctness.",
        "# A cross-encoder reranker on top of the",
        "# fused list gets you back to per-query best.",
        "# (That's module 05.)",
    ])


def render(out_svg: Path = ASSETS / "cheatsheet.svg",
           out_png: Path = ASSETS / "cheatsheet.png") -> tuple[Path, Path]:
    in_w = 15.4
    in_h = in_w * (PAGE_H / PAGE_W)
    fig, ax = plt.subplots(figsize=(in_w, in_h), facecolor=BG)
    ax.set_xlim(0, PAGE_W)
    ax.set_ylim(0, PAGE_H)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])

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
