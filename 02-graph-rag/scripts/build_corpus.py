"""Build the movie graph corpus used by NB1, NB2, NB3, NB4.

Ships with a curated, hand-authored seed list: 11 movies, 30 people, 8
studios, 60+ relationships. Picked so every film has a publicly-available
shooting draft on IMSDb, so NB4 can attach real script text to each
Movie node via a Script node and HAS_SCRIPT edge.

The studio attributions are deliberately simplified: real distribution
credits are messier (joint distribution, regional rights, label vs
studio). Treat this as a teaching graph, not a film database.

Output:
    data/movies.kuzu  (a single Kuzu database file, gitignored)

Scripts themselves are fetched separately by scripts/download_scripts.py
and cached to data/scripts/*.txt (also gitignored). The build_if_missing
function below populates Script nodes from that cache if present.
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "movies.kuzu"
SCRIPTS_DIR = DATA_DIR / "scripts"


# --- Nodes ---------------------------------------------------------------

# (name, born_year). born_year isn't strictly needed for the main demos,
# but NB2 uses it in a WHERE-filter example.
PEOPLE: list[tuple[str, int | None]] = [
    # Directors
    ("Quentin Tarantino", 1963),
    ("The Wachowskis", 1965),         # treated as one node, Lana's birth year
    ("Chad Stahelski", 1968),
    ("Christopher Nolan", 1970),
    ("Jordan Peele", 1979),
    ("Michel Gondry", 1963),
    # Actors
    ("John Travolta", 1954),
    ("Samuel L. Jackson", 1948),
    ("Uma Thurman", 1970),
    ("Bruce Willis", 1955),
    ("Steve Buscemi", 1957),
    ("Tim Roth", 1961),
    ("Harvey Keitel", 1939),
    ("Leonardo DiCaprio", 1974),
    ("Jamie Foxx", 1967),
    ("Keanu Reeves", 1964),
    ("Carrie-Anne Moss", 1967),
    ("Laurence Fishburne", 1961),
    ("Willem Dafoe", 1955),
    ("Guy Pearce", 1967),
    ("Joe Pantoliano", 1951),
    ("Christian Bale", 1974),
    ("Hugh Jackman", 1968),
    ("Scarlett Johansson", 1984),
    ("Joseph Gordon-Levitt", 1981),
    ("Marion Cotillard", 1975),
    ("Daniel Kaluuya", 1989),
    ("Allison Williams", 1988),
    ("Jim Carrey", 1962),
    ("Kate Winslet", 1975),
]

# (title, year, imsdb_slug). imsdb_slug is the script filename on
# imsdb.com/scripts/<slug>.html. Used by scripts/download_scripts.py.
MOVIES: list[tuple[str, int, str]] = [
    ("Pulp Fiction", 1994, "Pulp-Fiction"),
    ("Reservoir Dogs", 1992, "Reservoir-Dogs"),
    ("Django Unchained", 2012, "Django-Unchained"),
    ("The Matrix", 1999, "Matrix,-The"),
    ("The Matrix Reloaded", 2003, "Matrix-Reloaded,-The"),
    ("John Wick", 2014, "John-Wick"),
    ("Memento", 2000, "Memento"),
    ("The Prestige", 2006, "Prestige,-The"),
    ("Inception", 2010, "Inception"),
    ("Get Out", 2017, "Get-Out"),
    ("Eternal Sunshine of the Spotless Mind", 2004,
     "Eternal-Sunshine-of-the-Spotless-Mind"),
]

# (name,)
STUDIOS: list[tuple[str]] = [
    ("Miramax",),
    ("Weinstein",),
    ("Warner Bros",),
    ("Lionsgate",),
    ("Newmarket",),
    ("Touchstone",),
    ("Universal",),
    ("Focus Features",),
]


# --- Relationships -------------------------------------------------------

# (director, movie)
DIRECTED: list[tuple[str, str]] = [
    ("Quentin Tarantino", "Pulp Fiction"),
    ("Quentin Tarantino", "Reservoir Dogs"),
    ("Quentin Tarantino", "Django Unchained"),
    ("The Wachowskis", "The Matrix"),
    ("The Wachowskis", "The Matrix Reloaded"),
    ("Chad Stahelski", "John Wick"),
    ("Christopher Nolan", "Memento"),
    ("Christopher Nolan", "The Prestige"),
    ("Christopher Nolan", "Inception"),
    ("Jordan Peele", "Get Out"),
    ("Michel Gondry", "Eternal Sunshine of the Spotless Mind"),
]

# (actor, movie)
ACTED_IN: list[tuple[str, str]] = [
    # Pulp Fiction
    ("John Travolta", "Pulp Fiction"),
    ("Samuel L. Jackson", "Pulp Fiction"),
    ("Uma Thurman", "Pulp Fiction"),
    ("Bruce Willis", "Pulp Fiction"),
    # Reservoir Dogs
    ("Steve Buscemi", "Reservoir Dogs"),
    ("Tim Roth", "Reservoir Dogs"),
    ("Harvey Keitel", "Reservoir Dogs"),
    # Django Unchained
    ("Leonardo DiCaprio", "Django Unchained"),
    ("Jamie Foxx", "Django Unchained"),
    ("Samuel L. Jackson", "Django Unchained"),
    # The Matrix
    ("Keanu Reeves", "The Matrix"),
    ("Carrie-Anne Moss", "The Matrix"),
    ("Laurence Fishburne", "The Matrix"),
    ("Joe Pantoliano", "The Matrix"),
    # The Matrix Reloaded
    ("Keanu Reeves", "The Matrix Reloaded"),
    ("Carrie-Anne Moss", "The Matrix Reloaded"),
    ("Laurence Fishburne", "The Matrix Reloaded"),
    # John Wick
    ("Keanu Reeves", "John Wick"),
    ("Willem Dafoe", "John Wick"),
    # Memento
    ("Guy Pearce", "Memento"),
    ("Carrie-Anne Moss", "Memento"),
    ("Joe Pantoliano", "Memento"),
    # The Prestige
    ("Christian Bale", "The Prestige"),
    ("Hugh Jackman", "The Prestige"),
    ("Scarlett Johansson", "The Prestige"),
    # Inception
    ("Leonardo DiCaprio", "Inception"),
    ("Joseph Gordon-Levitt", "Inception"),
    ("Marion Cotillard", "Inception"),
    # Get Out
    ("Daniel Kaluuya", "Get Out"),
    ("Allison Williams", "Get Out"),
    # Eternal Sunshine
    ("Jim Carrey", "Eternal Sunshine of the Spotless Mind"),
    ("Kate Winslet", "Eternal Sunshine of the Spotless Mind"),
]

# (movie, studio)
PRODUCED_BY: list[tuple[str, str]] = [
    ("Pulp Fiction", "Miramax"),
    ("Reservoir Dogs", "Miramax"),
    ("Django Unchained", "Weinstein"),
    ("The Matrix", "Warner Bros"),
    ("The Matrix Reloaded", "Warner Bros"),
    ("John Wick", "Lionsgate"),
    ("Memento", "Newmarket"),
    ("The Prestige", "Touchstone"),
    ("Inception", "Warner Bros"),
    ("Get Out", "Universal"),
    ("Eternal Sunshine of the Spotless Mind", "Focus Features"),
]


# --- DDL -----------------------------------------------------------------

SCHEMA_STATEMENTS: list[str] = [
    """
    CREATE NODE TABLE Person(
        name STRING,
        born_year INT64,
        PRIMARY KEY (name)
    )
    """,
    """
    CREATE NODE TABLE Movie(
        title STRING,
        year INT64,
        PRIMARY KEY (title)
    )
    """,
    """
    CREATE NODE TABLE Studio(
        name STRING,
        PRIMARY KEY (name)
    )
    """,
    # Script lives in its own table. Long-form text doesn't belong on Movie:
    # NB1 and NB2 don't need it loaded, and a separate node makes the
    # one-to-zero-or-one nature explicit (some movies may not have a
    # downloaded script yet).
    """
    CREATE NODE TABLE Script(
        movie_title STRING,
        source_url STRING,
        content STRING,
        PRIMARY KEY (movie_title)
    )
    """,
    # Scene is Script chopped on screenplay slug lines (INT./EXT.). Production
    # graph RAG retrieves at this granularity, not at whole-document level:
    # the LLM gets a few KB of focused text instead of 300 KB it has to scan.
    """
    CREATE NODE TABLE Scene(
        scene_id STRING,
        movie_title STRING,
        scene_number INT64,
        heading STRING,
        body STRING,
        PRIMARY KEY (scene_id)
    )
    """,
    "CREATE REL TABLE DIRECTED(FROM Person TO Movie)",
    "CREATE REL TABLE ACTED_IN(FROM Person TO Movie)",
    "CREATE REL TABLE PRODUCED_BY(FROM Movie TO Studio)",
    "CREATE REL TABLE HAS_SCRIPT(FROM Movie TO Script)",
    "CREATE REL TABLE HAS_SCENE(FROM Movie TO Scene)",
]


# Screenplay scene headings start with INT./EXT./INT.EXT./EXT.INT., usually
# at the start of a line. Real-world scripts vary in convention: some are
# preceded by a scene number ("1   INT. ..."), some use a dash instead of a
# period ("EXT - ..." in Tarantino's later drafts). The regex is permissive
# enough to catch all 11 of our IMSDb scripts but still strict on the
# INT/EXT token so we don't match random words mid-paragraph.
_SCENE_HEADING_RE = re.compile(
    r"^[\s\dA-Z]{0,15}(INT\.|EXT\.|INT/EXT\.|EXT/INT\.|INT\s*-|EXT\s*-).*",
    re.MULTILINE,
)


def chunk_script_to_scenes(text: str) -> list[tuple[str, str]]:
    """Split a screenplay into (heading, body) pairs, one per slug line.

    Returns an empty list if no scene headings are found (unusual; means
    the file isn't a screenplay or uses an exotic format).
    """
    matches = list(_SCENE_HEADING_RE.finditer(text))
    if not matches:
        return []
    scenes: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end]
        heading = block.split("\n", 1)[0].strip()
        scenes.append((heading, block))
    return scenes


def _movie_slug_to_url(slug: str) -> str:
    return f"https://imsdb.com/scripts/{slug}.html"


def build_if_missing(force: bool = False) -> Path:
    """Idempotently materialize data/movies.kuzu. Returns the path.

    If scripts have been downloaded to data/scripts/<slug>.txt, they're
    loaded into Script nodes and connected via HAS_SCRIPT edges. Movies
    without a cached script just don't get one. NB4 can detect and
    re-fetch on demand.
    """
    if DB_PATH.exists() and not force:
        return DB_PATH

    # Kuzu writes a sibling .wal file. Both need to go, otherwise re-opening
    # replays the old schema and CREATE TABLE fails with "already exists".
    for path in [DB_PATH, DB_PATH.with_suffix(DB_PATH.suffix + ".wal")]:
        if path.exists():
            if path.is_file():
                path.unlink()
            else:
                shutil.rmtree(path)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    import kuzu

    db = kuzu.Database(str(DB_PATH))
    conn = kuzu.Connection(db)

    for stmt in SCHEMA_STATEMENTS:
        conn.execute(stmt)

    # Nodes: Person
    for name, born in PEOPLE:
        conn.execute(
            "CREATE (:Person {name: $name, born_year: $born})",
            {"name": name, "born": born},
        )

    # Nodes: Movie
    for title, year, _slug in MOVIES:
        conn.execute(
            "CREATE (:Movie {title: $title, year: $year})",
            {"title": title, "year": year},
        )

    # Nodes: Studio
    for (name,) in STUDIOS:
        conn.execute("CREATE (:Studio {name: $name})", {"name": name})

    # Edges
    for director, movie in DIRECTED:
        conn.execute(
            """
            MATCH (p:Person {name: $director}), (m:Movie {title: $movie})
            CREATE (p)-[:DIRECTED]->(m)
            """,
            {"director": director, "movie": movie},
        )
    for actor, movie in ACTED_IN:
        conn.execute(
            """
            MATCH (p:Person {name: $actor}), (m:Movie {title: $movie})
            CREATE (p)-[:ACTED_IN]->(m)
            """,
            {"actor": actor, "movie": movie},
        )
    for movie, studio in PRODUCED_BY:
        conn.execute(
            """
            MATCH (m:Movie {title: $movie}), (s:Studio {name: $studio})
            CREATE (m)-[:PRODUCED_BY]->(s)
            """,
            {"movie": movie, "studio": studio},
        )

    # Scripts: load any cached files into Script nodes + HAS_SCRIPT edges.
    # Missing scripts are fine: NB4 will tell the user to run
    # download_scripts.py if it can't find what it needs.
    # We also chunk each script into Scene nodes connected via HAS_SCENE.
    # Scripts give you the "full document" pattern; Scenes give you the
    # "retrieve a chunk" pattern. NB4 uses both.
    if SCRIPTS_DIR.exists():
        for title, _year, slug in MOVIES:
            script_path = SCRIPTS_DIR / f"{slug}.txt"
            if not script_path.exists():
                continue
            content = script_path.read_text(encoding="utf-8", errors="replace")

            # Script node + HAS_SCRIPT edge
            conn.execute(
                """
                CREATE (:Script {
                    movie_title: $title,
                    source_url: $url,
                    content: $content
                })
                """,
                {
                    "title": title,
                    "url": _movie_slug_to_url(slug),
                    "content": content,
                },
            )
            conn.execute(
                """
                MATCH (m:Movie {title: $title}), (s:Script {movie_title: $title})
                CREATE (m)-[:HAS_SCRIPT]->(s)
                """,
                {"title": title},
            )

            # Scene nodes + HAS_SCENE edges (one per slug line in the script)
            for scene_number, (heading, body) in enumerate(chunk_script_to_scenes(content)):
                scene_id = f"{title}#{scene_number:04d}"
                conn.execute(
                    """
                    CREATE (:Scene {
                        scene_id: $scene_id,
                        movie_title: $title,
                        scene_number: $n,
                        heading: $heading,
                        body: $body
                    })
                    """,
                    {
                        "scene_id": scene_id,
                        "title": title,
                        "n": scene_number,
                        "heading": heading,
                        "body": body,
                    },
                )
                conn.execute(
                    """
                    MATCH (m:Movie {title: $title}),
                          (sc:Scene {scene_id: $scene_id})
                    CREATE (m)-[:HAS_SCENE]->(sc)
                    """,
                    {"title": title, "scene_id": scene_id},
                )

    return DB_PATH


if __name__ == "__main__":
    force = "--force" in sys.argv
    path = build_if_missing(force=force)

    import kuzu

    conn = kuzu.Connection(kuzu.Database(str(path)))

    print(f"Built movie graph at {path}")
    print()
    print(
        f"Nodes: {len(PEOPLE)} people, {len(MOVIES)} movies, {len(STUDIOS)} studios"
    )
    print(
        f"Edges: {len(DIRECTED)} DIRECTED, "
        f"{len(ACTED_IN)} ACTED_IN, {len(PRODUCED_BY)} PRODUCED_BY"
    )
    n_scripts = conn.execute(
        "MATCH (s:Script) RETURN count(s) AS n"
    ).get_as_df().iloc[0]["n"]
    print(f"Scripts loaded: {n_scripts} / {len(MOVIES)}")
    if n_scripts < len(MOVIES):
        print(
            f"  (Run `python scripts/download_scripts.py` "
            f"to fetch the rest into {SCRIPTS_DIR.relative_to(DATA_DIR.parent)})"
        )
    n_scenes = conn.execute(
        "MATCH (sc:Scene) RETURN count(sc) AS n"
    ).get_as_df().iloc[0]["n"]
    print(f"Scenes loaded: {n_scenes}")
    print()
    print("Sanity check: films per director")
    df = conn.execute(
        """
        MATCH (p:Person)-[:DIRECTED]->(m:Movie)
        RETURN p.name AS director, count(m) AS films
        ORDER BY films DESC, director
        """
    ).get_as_df()
    print(df.to_string(index=False))
    if n_scenes:
        print()
        print("Sanity check: scenes per movie")
        df = conn.execute(
            """
            MATCH (m:Movie)-[:HAS_SCENE]->(sc:Scene)
            RETURN m.title AS movie, count(sc) AS scenes
            ORDER BY scenes DESC
            """
        ).get_as_df()
        print(df.to_string(index=False))
