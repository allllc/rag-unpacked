"""Render the Kuzu + LangChain Graph RAG cheat sheet as SVG and PNG.

Same DataCamp-style one-pager as module 01: pink banner, three columns of
syntax sections, dark header bars, REPL-style `>>>` lines, green `# comments`.
Layout primitives are lifted verbatim from module 01's cheatsheet renderer.

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

PAGE_W, PAGE_H = 22.0, 19.5
MARGIN = 0.35
BANNER_H = 1.35
COL_GAP = 0.35


def draw_banner(ax):
    y0 = PAGE_H - BANNER_H
    ax.add_patch(Rectangle((0, y0), PAGE_W, BANNER_H, facecolor=BANNER, edgecolor="none"))
    ax.text(MARGIN + 0.15, PAGE_H - 0.45, "rag-unpacked",
            color=BANNER_INK, fontsize=11, fontweight="bold", va="top",
            family="DejaVu Sans")
    ax.text(MARGIN + 0.15, PAGE_H - 0.80, "02. Graph RAG",
            color=BANNER_INK, fontsize=9.5, va="top", family="DejaVu Sans")
    ax.text(PAGE_W - MARGIN - 0.15, PAGE_H - 0.35,
            "Kuzu + LangChain",
            color=BANNER_INK, fontsize=14, fontweight="bold", va="top", ha="right",
            family="DejaVu Sans")
    ax.text(PAGE_W - MARGIN - 0.15, PAGE_H - 0.90,
            "Graph RAG Cheat Sheet",
            color=BANNER_INK, fontsize=20, fontweight="bold", va="top", ha="right",
            family="DejaVu Sans")


def draw_footer(ax):
    ax.text(MARGIN, 0.18, "github.com/allllc/rag-unpacked",
            color=MUTED, fontsize=7.8, va="center", family="DejaVu Sans")
    ax.text(PAGE_W - MARGIN, 0.18,
            "module 02  ·  graph-rag  ·  NB1-NB4  ·  Kuzu 0.11 + langchain-kuzu 0.4",
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
        "# .env  (gitignored)",
        "OPENAI_API_KEY=sk-...",
        "",
        ">>> from helpers import load_env, \\",
        "...     get_openai_client, get_kuzu_conn",
        ">>> cfg = load_env()",
        ">>> conn = get_kuzu_conn()",
        ">>> client = get_openai_client(cfg)  # NB3 only",
    ])

    draw_section(ax, col, "Schema (DDL)", [
        "# Kuzu requires types up front.",
        ">>> conn.execute(\"\"\"",
        "...   CREATE NODE TABLE Person(",
        "...     name STRING,",
        "...     born_year INT64,",
        "...     PRIMARY KEY (name))\"\"\")",
        "",
        ">>> conn.execute(\"\"\"",
        "...   CREATE NODE TABLE Movie(",
        "...     title STRING, year INT64,",
        "...     PRIMARY KEY (title))\"\"\")",
        "",
        "# Edges: FROM and TO are required.",
        ">>> conn.execute(",
        "...   \"CREATE REL TABLE ACTED_IN(\"",
        "...   \"FROM Person TO Movie)\")",
    ])

    draw_section(ax, col, "Inserts", [
        "# Nodes (parameterized for safety)",
        ">>> conn.execute(",
        "...   \"CREATE (:Person {name: $n})\",",
        "...   {\"n\": \"Keanu Reeves\"})",
        "",
        "# Relationships need a MATCH first",
        ">>> conn.execute(\"\"\"",
        "...   MATCH (p:Person {name: $a}),",
        "...         (m:Movie {title: $t})",
        "...   CREATE (p)-[:ACTED_IN]->(m)",
        "... \"\"\", {\"a\": \"Keanu Reeves\",",
        "...      \"t\": \"The Matrix\"})",
    ])

    draw_section(ax, col, "Inspect", [
        ">>> conn.execute(",
        "...   \"CALL show_tables() RETURN *\"",
        "... ).get_as_df()",
        "",
        "# Count per node label",
        ">>> conn.execute(\"\"\"",
        "...   MATCH (n) RETURN label(n), count(n)",
        "... \"\"\").get_as_df()",
    ])


def content_column_2(ax, col):
    draw_section(ax, col, "MATCH + RETURN", [
        ">>> conn.execute(\"\"\"",
        "...   MATCH (p:Person)-[:ACTED_IN]->(m:Movie)",
        "...   RETURN p.name AS actor,",
        "...          m.title AS movie",
        "...   ORDER BY actor",
        "... \"\"\").get_as_df()",
    ], subtitle="The pattern is the query")

    draw_section(ax, col, "WHERE filters", [
        "# Numeric range",
        ">>> WHERE p.born_year > 1970",
        ">>> WHERE m.year < 2000",
        "",
        "# Inequality",
        ">>> WHERE other.name <> 'Keanu Reeves'",
        "",
        "# String contains",
        ">>> WHERE m.title CONTAINS 'Matrix'",
    ])

    draw_section(ax, col, "Multi-hop traversals", [
        "# Two-hop: co-actors",
        ">>> MATCH (a:Person)-[:ACTED_IN]->",
        "...       (m:Movie)<-[:ACTED_IN]-(b:Person)",
        "...   WHERE a.name = 'Keanu Reeves'",
        "...     AND b.name <> 'Keanu Reeves'",
        "",
        "# Three-hop: directors via films",
        ">>> MATCH (a:Person)-[:ACTED_IN]->",
        "...       (m:Movie)<-[:DIRECTED]-(d:Person)",
    ])

    draw_section(ax, col, "Aggregates", [
        "# Group + tally",
        ">>> RETURN d.name, count(m) AS films",
        ">>> ORDER BY films DESC",
        "",
        "# Group + list (DISTINCT to dedupe)",
        ">>> RETURN d.name,",
        "...   collect(DISTINCT a.name) AS actors",
        "",
        "# Note: 'cast' is a reserved word.",
    ])

    draw_section(ax, col, "OPTIONAL MATCH + params", [
        "# Like SQL LEFT JOIN: keep rows,",
        "# NULL the missing side.",
        ">>> MATCH (d:Person)-[:DIRECTED]->(m:Movie)",
        ">>> OPTIONAL MATCH (m)-[:PRODUCED_BY]->(s)",
        "",
        "# Parameters (safe, faster)",
        ">>> conn.execute(",
        "...   \"MATCH (p:Person {name: $n}) RETURN p\",",
        "...   {\"n\": \"Christopher Nolan\"})",
    ])


def content_column_3(ax, col):
    draw_section(ax, col, "LangChain wrapper", [
        ">>> from langchain_kuzu.graphs.kuzu_graph \\",
        "...   import KuzuGraph",
        ">>> from langchain_kuzu.chains.graph_qa.kuzu \\",
        "...   import KuzuQAChain",
        ">>> from langchain_openai import ChatOpenAI",
        "",
        "# Reuse conn.database (don't open a 2nd DB",
        "# handle: Kuzu file-locks the path).",
        ">>> graph = KuzuGraph(conn.database,",
        "...   allow_dangerous_requests=True)",
        ">>> print(graph.schema)  # what the LLM sees",
    ])

    draw_section(ax, col, "Manual loop (Act I)", [
        "# 1. Generate Cypher",
        ">>> prompt = (INSTRUCTIONS",
        "...   + \"\\nSchema:\\n\" + graph.schema",
        "...   + \"\\nQuestion:\\n\" + question)",
        ">>> cypher = client.chat.completions",
        "...   .create(...).choices[0].message.content",
        "",
        "# 2. Run on Kuzu",
        ">>> rows = conn.execute(cypher).get_as_df()",
        "",
        "# 3. Synthesize answer (same client call",
        "#    with question + rows.to_string())",
    ], subtitle="Don't use .format(): schema has { }")

    draw_section(ax, col, "KuzuQAChain (Act II)", [
        ">>> llm = ChatOpenAI(model=\"gpt-4o-mini\",",
        "...                  temperature=0)",
        ">>> chain = KuzuQAChain.from_llm(",
        "...   llm=llm, graph=graph,",
        "...   verbose=True,",
        "...   allow_dangerous_requests=True)",
        "",
        ">>> chain.invoke({\"query\": q})[\"result\"]",
        "",
        "# Same two LLM calls, hidden. Prompts in",
        "# langchain_kuzu.chains.graph_qa.prompts",
    ])

    draw_section(ax, col, "Text on nodes (NB4)", [
        "# Schema: two text patterns side by side.",
        ">>> CREATE NODE TABLE Script(",
        "...   movie_title STRING, content STRING,",
        "...   PRIMARY KEY (movie_title))",
        ">>> CREATE NODE TABLE Scene(",
        "...   scene_id STRING, heading STRING,",
        "...   body STRING, PRIMARY KEY (scene_id))",
        ">>> CREATE REL TABLE",
        "...   HAS_SCRIPT(FROM Movie TO Script),",
        "...   HAS_SCENE (FROM Movie TO Scene)",
        "",
        "# Whole-document retrieval (baseline)",
        ">>> MATCH (m:Movie {title: $t})",
        "...   -[:HAS_SCRIPT]->(s) RETURN s.content",
        "",
        "# Chunk-level retrieval (production)",
        ">>> MATCH (m:Movie {title: $t})",
        "...   -[:HAS_SCENE]->(sc:Scene)",
        "...   WHERE lower(sc.body)",
        "...         CONTAINS lower($keyword)",
        "...   RETURN sc.heading, sc.body",
        "",
        "# Kuzu CONTAINS is case-sensitive: wrap",
        "# both sides with lower() to be safe.",
    ], subtitle="Chunk on natural boundaries (INT./EXT.)")

    draw_section(ax, col, "Switching to Neo4j", [
        "# Same Cypher; just swap the graph ctor.",
        ">>> from langchain_neo4j import Neo4jGraph",
        ">>> graph = Neo4jGraph(url=..., username=...,",
        "...                    password=...)",
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
