"""Render the RAGAS + hand-rolled Evaluating RAG cheat sheet as SVG and PNG.

Same DataCamp-style one-pager as modules 01-03: pink banner, three
columns of syntax sections, dark header bars, REPL-style `>>>` lines,
green `# comments`. Layout primitives lifted verbatim from module 03.

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
    ax.text(MARGIN + 0.15, PAGE_H - 0.80, "04. Evaluating RAG",
            color=BANNER_INK, fontsize=9.5, va="top", family="DejaVu Sans")
    ax.text(PAGE_W - MARGIN - 0.15, PAGE_H - 0.35,
            "RAGAS + hand-rolled",
            color=BANNER_INK, fontsize=14, fontweight="bold", va="top", ha="right",
            family="DejaVu Sans")
    ax.text(PAGE_W - MARGIN - 0.15, PAGE_H - 0.90,
            "Evaluating RAG Cheat Sheet",
            color=BANNER_INK, fontsize=20, fontweight="bold", va="top", ha="right",
            family="DejaVu Sans")


def draw_footer(ax):
    ax.text(MARGIN, 0.18, "github.com/allllc/rag-unpacked",
            color=MUTED, fontsize=7.8, va="center", family="DejaVu Sans")
    ax.text(PAGE_W - MARGIN, 0.18,
            "module 04  ·  evaluating-rag  ·  NB1-NB4  ·  ragas 0.4 + OpenAI",
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
        "# .env  (gitignored, NB2/3/4 only)",
        "OPENAI_API_KEY=sk-...",
        "# Optional, for NB3 head-to-head:",
        "# PINECONE_API_KEY=pcsk-...",
        "",
        ">>> from helpers import load_env, \\",
        "...     get_openai_client, get_evaluator_llm, \\",
        "...     get_evaluator_embeddings",
        ">>> cfg = load_env()",
    ])

    draw_section(ax, col, "The golden eval set", [
        "# 4 columns + a shape tag.",
        "#   question_id, question, ground_truth_answer,",
        "#   relevant_doc_ids, shape",
        "#",
        "# Shapes (cheap analytic win):",
        "#   keyword     - rare token in question",
        "#   paraphrase  - synonym, no shared tokens",
        "#   synthesis   - multi-doc",
        "#   adversarial - not in corpus",
        "",
        ">>> golden = pd.read_parquet(",
        "...   \"data/golden.parquet\")",
    ], subtitle="One small parquet, four question shapes")

    draw_section(ax, col, "precision@k", [
        "# |retrieved[:k] ∩ relevant| / k",
        "",
        ">>> precision_at_k(",
        "...   retrieved_ids, relevant_ids, k=3)",
        "",
        "# Edge case: adversarial questions have an",
        "# empty relevant set. We return 1.0 if also",
        "# retrieved nothing, else 0.0. The 'should",
        "# have refused' case is generation, not",
        "# retrieval.",
    ])

    draw_section(ax, col, "recall@k", [
        "# |retrieved[:k] ∩ relevant| / |relevant|",
        "",
        ">>> recall_at_k(",
        "...   retrieved_ids, relevant_ids, k=5)",
        "",
        "# Empty relevant set returns 1.0",
        "# (nothing to recall).",
    ])

    draw_section(ax, col, "Mean Reciprocal Rank", [
        "# mean(1 / rank_of_first_relevant)",
        "",
        ">>> mean_reciprocal_rank(",
        "...   retrieved_lists, relevant_lists)",
        "",
        "# Queries with empty relevant skipped.",
        "# Right metric when there's one right",
        "# answer per question.",
        "#",
        "# The cheap metrics:",
        "#   set arithmetic, no LLM, no money.",
        "#   Run them on every PR.",
    ], subtitle="MRR, the rank-only metric")


def content_column_2(ax, col):
    draw_section(ax, col, "The 4 RAGAS columns", [
        "# EvaluationDataset.from_list([{...}, ...])",
        "#",
        "#   user_input         the question",
        "#   retrieved_contexts list of doc texts",
        "#   response           generated answer",
        "#   reference          ground-truth answer",
        "",
        "# Different metrics need different cols:",
        "#   Faithfulness: response vs contexts",
        "#   ResponseRelevancy: response vs question",
        "#   *WithReference: contexts vs reference",
    ], subtitle="4 columns, picked per metric")

    draw_section(ax, col, "The 4 metrics", [
        "# Faithfulness",
        "#   does every claim in the answer trace",
        "#   to the retrieved context?",
        "",
        "# ResponseRelevancy",
        "#   does the answer match the question?",
        "",
        "# LLMContextRecall",
        "#   contexts cover the ground-truth?",
        "",
        "# LLMContextPrecisionWithReference",
        "#   contexts are precise + well-ranked?",
    ])

    draw_section(ax, col, "Minimal evaluate() call", [
        ">>> from ragas import evaluate, \\",
        "...   EvaluationDataset",
        ">>> from ragas.metrics import (",
        "...   Faithfulness, ResponseRelevancy,",
        "...   LLMContextRecall,",
        "...   LLMContextPrecisionWithReference)",
        "",
        ">>> ds = EvaluationDataset.from_list(rows)",
        ">>> result = evaluate(",
        "...   dataset=ds,",
        "...   metrics=[Faithfulness(),",
        "...     ResponseRelevancy(), ...],",
        "...   llm=get_evaluator_llm(cfg),",
        "...   embeddings=get_evaluator_embeddings(cfg),",
        ")",
        ">>> result.to_pandas()",
    ])

    draw_section(ax, col, "Stochasticity tax", [
        "# Same input, different score per run.",
        "# Mean delta on rerun: ~0.05-0.10.",
        "# Occasional full-flip (0 to 1) happens.",
        "#",
        "# Production fix: average 3 runs, or use",
        "# a stronger judge model (gpt-4o, opus).",
        "# Both cost more, both work.",
    ])

    draw_section(ax, col, "Embedder gotcha", [
        "# ResponseRelevancy needs embeddings.",
        "# Without them you get a runtime error.",
        "",
        ">>> embeddings = get_evaluator_embeddings(cfg)",
        ">>> evaluate(..., embeddings=embeddings)",
    ], subtitle="The one wiring detail RAGAS won't infer")


def content_column_3(ax, col):
    draw_section(ax, col, "The head-to-head pattern", [
        "# Three retrievers, six metrics, one matrix.",
        "#",
        "# 1. Pick retrievers (BM25 / vector / hybrid)",
        "# 2. Run all three on the same eval set",
        "# 3. Cache to disk so reruns are free",
        "# 4. Compute retrieval metrics (cheap)",
        "# 5. Generate answers + RAGAS metrics (expensive)",
        "# 6. Assemble the matrix",
        "",
        "#               BM25   vector  hybrid",
        "#  p@1          0.45   0.75    0.65",
        "#  r@5          0.89   0.99    0.94",
        "#  MRR          0.74   0.96    0.91",
        "#  faithfulness 0.71   0.69    0.78",
        "#  ans_rel      0.54   0.53    0.58",
        "#  ctx_prec     0.63   0.84    0.83",
    ], subtitle="The matrix module 04 exists to produce")

    draw_section(ax, col, "Measure cheap, measure often", [
        "# Cheap metrics (no LLM):",
        "#   p@k, r@k, MRR.",
        "#   Run on every PR.",
        "",
        "# Expensive metrics (LLM judge):",
        "#   faithfulness, answer_relevancy,",
        "#   context_precision, context_recall.",
        "#   Run nightly, or per release.",
        "",
        "# Same eval set serves both cadences.",
    ])

    draw_section(ax, col, "Graph RAG: two extra metrics", [
        "# RAGAS doesn't ship these. ~15 lines each.",
        "",
        "# cypher_hit",
        "#   did the Cypher result contain the",
        "#   ground-truth target (movie/scene)?",
        "",
        "# schema_coverage",
        "#   did the Cypher use the schema tokens",
        "#   (relationship types, properties) the",
        "#   question implies it should have?",
        "",
        "# Three failure modes worth naming:",
        "#   hallucinated Cypher that runs wrong,",
        "#   hallucinated Cypher that won't compile,",
        "#   schema-correct query missing relations.",
    ], subtitle="When the corpus is a graph")

    draw_section(ax, col, "The portfolio takeaway", [
        "# Evaluation is the loop you wrap",
        "# around modules 01-03, not a fourth",
        "# retriever. Same shape works on any",
        "# retriever you build next.",
        "#",
        "# A 20-question eval set is cheap to",
        "# author, fast to run, and decisive",
        "# enough to ship behind. The 200-2000",
        "# question version is the same shape.",
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
