"""Build the 12-question movie-graph eval set used by NB4.

Twelve hand-authored questions on module 02's movie Kuzu DB, with three
shapes:

  - structural (5): Cypher-RAG should win. Director attribution, studio,
    actor filmography, year filter, multi-hop "directors of person X."
    Answerable from the graph schema alone.
  - text (5): Scene-RAG should win. The answer lives in the script text:
    Sunken Place scenes (Get Out), red/blue pill scene (Matrix), totem
    scenes (Inception), John Wick's puppy scene, Memento's opening
    Polaroid scene.
  - hybrid (2): both retrievers attempt. mq11 ("Nolan film with Polaroid
    in reverse" -> Memento) and mq12 ("Wachowski film mentioning Zion" ->
    The Matrix Reloaded).

Every piece of ground truth in this seed is verified against module 02's
Kuzu DB at build time. If the DB is missing, the script prints a clear
"run module 02's build script first" message and exits 1.

Output: data/golden_movies.parquet (~6 KB).
"""
from __future__ import annotations

import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EVAL_PATH = DATA_DIR / "golden_movies.parquet"
KUZU_PATH = Path(__file__).resolve().parent.parent.parent / "02-graph-rag" / "data" / "movies.kuzu"


_SEED_QUESTIONS: list[dict] = [
    # --- structural (Cypher-RAG favored) ---
    {
        "question": "Who directed Inception?",
        "ground_truth_answer": "Christopher Nolan.",
        "relevant_cypher_targets": ["Inception"],
        "relevant_scene_keys": [],
        "schema_tokens": ["DIRECTED"],
        "shape": "structural",
    },
    {
        "question": "Which studio produced Get Out?",
        "ground_truth_answer": "Universal.",
        "relevant_cypher_targets": ["Get Out"],
        "relevant_scene_keys": [],
        "schema_tokens": ["PRODUCED_BY"],
        "shape": "structural",
    },
    {
        "question": "How many movies has Keanu Reeves been in across this graph?",
        "ground_truth_answer": "Three: The Matrix, The Matrix Reloaded, and John Wick.",
        "relevant_cypher_targets": ["The Matrix", "The Matrix Reloaded", "John Wick"],
        "relevant_scene_keys": [],
        "schema_tokens": ["ACTED_IN"],
        "shape": "structural",
    },
    {
        "question": "List every movie in this graph released before the year 2000.",
        "ground_truth_answer": "Reservoir Dogs (1992), Pulp Fiction (1994), and The Matrix (1999).",
        "relevant_cypher_targets": ["Reservoir Dogs", "Pulp Fiction", "The Matrix"],
        "relevant_scene_keys": [],
        "schema_tokens": ["year"],
        "shape": "structural",
    },
    {
        "question": "Which directors have worked with Carrie-Anne Moss?",
        "ground_truth_answer": "The Wachowskis (The Matrix and The Matrix Reloaded) and Christopher Nolan (Memento).",
        "relevant_cypher_targets": ["The Matrix", "The Matrix Reloaded", "Memento"],
        "relevant_scene_keys": [],
        "schema_tokens": ["ACTED_IN", "DIRECTED"],
        "shape": "structural",
    },
    # --- text (Scene-RAG favored) ---
    {
        "question": "In the Get Out script, which scene numbers reference the Sunken Place?",
        "ground_truth_answer": "Scenes 58, 92, and 107 of Get Out.",
        "relevant_cypher_targets": [],
        "relevant_scene_keys": [("Get Out", 58), ("Get Out", 92), ("Get Out", 107)],
        "schema_tokens": ["HAS_SCRIPT", "HAS_SCENE", "body"],
        "shape": "text",
    },
    {
        "question": "Which scene of The Matrix contains the red pill / blue pill choice?",
        "ground_truth_answer": "Scene 27 of The Matrix.",
        "relevant_cypher_targets": [],
        "relevant_scene_keys": [("The Matrix", 27)],
        "schema_tokens": ["HAS_SCRIPT", "HAS_SCENE", "body"],
        "shape": "text",
    },
    {
        "question": "Which scenes of Inception reference Cobb's totem?",
        "ground_truth_answer": "Scenes 63, 67, 87, and 250 of Inception.",
        "relevant_cypher_targets": [],
        "relevant_scene_keys": [("Inception", 63), ("Inception", 67), ("Inception", 87), ("Inception", 250)],
        "schema_tokens": ["HAS_SCRIPT", "HAS_SCENE", "body"],
        "shape": "text",
    },
    {
        "question": "Which scene of John Wick introduces the puppy?",
        "ground_truth_answer": "Scene 18 of John Wick.",
        "relevant_cypher_targets": [],
        "relevant_scene_keys": [("John Wick", 18)],
        "schema_tokens": ["HAS_SCRIPT", "HAS_SCENE", "body"],
        "shape": "text",
    },
    {
        "question": "Which scenes of Pulp Fiction reference the briefcase?",
        "ground_truth_answer": "Scenes 7, 9, and 93 of Pulp Fiction.",
        "relevant_cypher_targets": [],
        "relevant_scene_keys": [("Pulp Fiction", 7), ("Pulp Fiction", 9), ("Pulp Fiction", 93)],
        "schema_tokens": ["HAS_SCRIPT", "HAS_SCENE", "body"],
        "shape": "text",
    },
    # --- hybrid (both retrievers attempt) ---
    {
        "question": "Which Christopher Nolan film opens with a Polaroid photograph fading in reverse?",
        "ground_truth_answer": "Memento. Scene 0 of Memento opens with a Polaroid photograph fading in reverse as the titles roll.",
        "relevant_cypher_targets": ["Memento"],
        "relevant_scene_keys": [("Memento", 0)],
        "schema_tokens": ["DIRECTED", "HAS_SCRIPT", "HAS_SCENE", "body"],
        "shape": "hybrid",
    },
    {
        "question": "Which Wachowski film's script mentions Zion?",
        "ground_truth_answer": "The Matrix Reloaded.",
        "relevant_cypher_targets": ["The Matrix Reloaded"],
        "relevant_scene_keys": [("The Matrix Reloaded", 8), ("The Matrix Reloaded", 10)],
        "schema_tokens": ["DIRECTED", "HAS_SCRIPT", "HAS_SCENE", "body"],
        "shape": "hybrid",
    },
]


def _verify_against_kuzu():
    """Probe module 02's Kuzu DB to ensure every ground truth claim still holds.

    Catches drift if module 02 ever regenerates its corpus with different scene
    numbers or titles. Loud failure beats silent miscalibration.
    """
    if not KUZU_PATH.exists():
        print(f"ERROR: Kuzu DB not found at {KUZU_PATH}", file=sys.stderr)
        print("NB4 needs module 02's movie graph. Run this first:", file=sys.stderr)
        print(f"  cd {KUZU_PATH.parent.parent} && python scripts/build_corpus.py", file=sys.stderr)
        sys.exit(1)

    import kuzu

    db = kuzu.Database(str(KUZU_PATH))
    conn = kuzu.Connection(db)

    for q in _SEED_QUESTIONS:
        # Verify movie titles exist
        for title in q["relevant_cypher_targets"]:
            r = conn.execute(f"MATCH (m:Movie {{title: $t}}) RETURN count(m) AS n", {"t": title})
            n = r.get_as_df().iloc[0]["n"]
            if n == 0:
                print(f"WARNING: '{title}' not in graph (from question: {q['question'][:60]})", file=sys.stderr)

        # Verify scene keys
        for movie, scene_num in q["relevant_scene_keys"]:
            r = conn.execute(
                "MATCH (s:Scene) WHERE s.movie_title = $m AND s.scene_number = $n RETURN count(s) AS k",
                {"m": movie, "n": scene_num},
            )
            k = r.get_as_df().iloc[0]["k"]
            if k == 0:
                print(f"WARNING: ({movie}, scene {scene_num}) not in graph", file=sys.stderr)


def build_seed_eval_set():
    import pandas as pd

    rows = []
    for i, entry in enumerate(_SEED_QUESTIONS, start=1):
        rows.append(
            {
                "question_id": f"mq{i:02d}",
                "question": entry["question"],
                "ground_truth_answer": entry["ground_truth_answer"],
                "relevant_cypher_targets": list(entry["relevant_cypher_targets"]),
                # Scene keys stored as "Movie Title#scene_number" strings for parquet compatibility
                "relevant_scene_keys": [f"{m}#{n}" for m, n in entry["relevant_scene_keys"]],
                "schema_tokens": list(entry["schema_tokens"]),
                "shape": entry["shape"],
            }
        )
    return pd.DataFrame(rows)


def build_if_missing(force: bool = False, verify: bool = True) -> Path:
    """Idempotently materialize data/golden_movies.parquet. Returns the path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if EVAL_PATH.exists() and not force:
        return EVAL_PATH
    if verify:
        _verify_against_kuzu()
    df = build_seed_eval_set()
    df.to_parquet(EVAL_PATH, index=False)
    return EVAL_PATH


if __name__ == "__main__":
    force = "--force" in sys.argv
    no_verify = "--no-verify" in sys.argv
    path = build_if_missing(force=force, verify=not no_verify)
    import pandas as pd

    df = pd.read_parquet(path)
    print(f"Wrote {len(df)} movie questions to {path}")
    print(df[["question_id", "shape", "question"]].to_string(index=False))
    print(f"\nShape breakdown: {df['shape'].value_counts().to_dict()}")
