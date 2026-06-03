"""Build the movie graph corpus used by NB1, NB2, NB3.

Ships with a curated, hand-authored seed list: ~11 movies, ~12 people,
~5 studios, ~40 relationships. Small enough to print every row, big
enough that two-hop and three-hop traversals have real answers to find.

The studio attributions are deliberately simplified: real distribution
credits are messier (joint distribution, regional rights, label vs
studio). Treat this as a teaching graph, not a film database.

Output: data/movies.kuzu/ (a Kuzu database directory, gitignored).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "movies.kuzu"


# --- Nodes ---------------------------------------------------------------

# (name, born_year)
PEOPLE: list[tuple[str, int]] = [
    ("Christopher Nolan", 1970),
    ("Denis Villeneuve", 1967),
    ("Greta Gerwig", 1983),
    ("Jordan Peele", 1979),
    ("Bong Joon-ho", 1969),
    ("Cillian Murphy", 1976),
    ("Florence Pugh", 1996),
    ("Robert Pattinson", 1986),
    ("Zendaya", 1996),
    ("Timothée Chalamet", 1995),
    ("Margot Robbie", 1990),
    ("Daniel Kaluuya", 1989),
    ("Lupita Nyong'o", 1983),
    ("Ryan Gosling", 1980),
]

# (title, year)
MOVIES: list[tuple[str, int]] = [
    ("Oppenheimer", 2023),
    ("Tenet", 2020),
    ("Dune: Part Two", 2024),
    ("Blade Runner 2049", 2017),
    ("Barbie", 2023),
    ("Little Women", 2019),
    ("Get Out", 2017),
    ("Nope", 2022),
    ("Us", 2019),
    ("Parasite", 2019),
    ("Mickey 17", 2025),
]

# (name,)
STUDIOS: list[tuple[str]] = [
    ("Warner Bros",),
    ("Universal",),
    ("Lionsgate",),
    ("Paramount",),
    ("A24",),
]


# --- Relationships -------------------------------------------------------

# (director, movie)
DIRECTED: list[tuple[str, str]] = [
    ("Christopher Nolan", "Oppenheimer"),
    ("Christopher Nolan", "Tenet"),
    ("Denis Villeneuve", "Dune: Part Two"),
    ("Denis Villeneuve", "Blade Runner 2049"),
    ("Greta Gerwig", "Barbie"),
    ("Greta Gerwig", "Little Women"),
    ("Jordan Peele", "Get Out"),
    ("Jordan Peele", "Nope"),
    ("Jordan Peele", "Us"),
    ("Bong Joon-ho", "Parasite"),
    ("Bong Joon-ho", "Mickey 17"),
]

# (actor, movie)
ACTED_IN: list[tuple[str, str]] = [
    ("Cillian Murphy", "Oppenheimer"),
    ("Florence Pugh", "Oppenheimer"),
    ("Robert Pattinson", "Tenet"),
    ("Timothée Chalamet", "Dune: Part Two"),
    ("Zendaya", "Dune: Part Two"),
    ("Florence Pugh", "Dune: Part Two"),
    ("Ryan Gosling", "Blade Runner 2049"),
    ("Ryan Gosling", "Barbie"),
    ("Margot Robbie", "Barbie"),
    ("Florence Pugh", "Little Women"),
    ("Timothée Chalamet", "Little Women"),
    ("Daniel Kaluuya", "Get Out"),
    ("Daniel Kaluuya", "Nope"),
    ("Lupita Nyong'o", "Us"),
    ("Robert Pattinson", "Mickey 17"),
]

# (movie, studio)
PRODUCED_BY: list[tuple[str, str]] = [
    ("Oppenheimer", "Universal"),
    ("Tenet", "Warner Bros"),
    ("Dune: Part Two", "Warner Bros"),
    ("Blade Runner 2049", "Warner Bros"),
    ("Barbie", "Warner Bros"),
    ("Little Women", "Lionsgate"),
    ("Get Out", "Universal"),
    ("Nope", "Universal"),
    ("Us", "Universal"),
    ("Parasite", "A24"),
    ("Mickey 17", "Warner Bros"),
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
    "CREATE REL TABLE DIRECTED(FROM Person TO Movie)",
    "CREATE REL TABLE ACTED_IN(FROM Person TO Movie)",
    "CREATE REL TABLE PRODUCED_BY(FROM Movie TO Studio)",
]


def build_if_missing(force: bool = False) -> Path:
    """Idempotently materialize data/movies.kuzu. Returns the path.

    Kuzu 0.11+ stores the database as a single file (despite the name
    suggesting a directory). We use unlink, not rmtree.
    """
    if DB_PATH.exists() and not force:
        return DB_PATH

    if DB_PATH.exists():
        if DB_PATH.is_file():
            DB_PATH.unlink()
        else:
            shutil.rmtree(DB_PATH)  # older Kuzu directory layout
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    import kuzu

    db = kuzu.Database(str(DB_PATH))
    conn = kuzu.Connection(db)

    for stmt in SCHEMA_STATEMENTS:
        conn.execute(stmt)

    # Nodes
    for name, born in PEOPLE:
        conn.execute(
            "CREATE (:Person {name: $name, born_year: $born})",
            {"name": name, "born": born},
        )
    for title, year in MOVIES:
        conn.execute(
            "CREATE (:Movie {title: $title, year: $year})",
            {"title": title, "year": year},
        )
    for (name,) in STUDIOS:
        conn.execute("CREATE (:Studio {name: $name})", {"name": name})

    # Relationships
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

    return DB_PATH


if __name__ == "__main__":
    force = "--force" in sys.argv
    path = build_if_missing(force=force)

    import kuzu

    conn = kuzu.Connection(kuzu.Database(str(path)))

    print(f"Built movie graph at {path}")
    print()
    print(f"Nodes: {len(PEOPLE)} people, {len(MOVIES)} movies, {len(STUDIOS)} studios")
    print(
        f"Edges: {len(DIRECTED)} DIRECTED, "
        f"{len(ACTED_IN)} ACTED_IN, {len(PRODUCED_BY)} PRODUCED_BY"
    )
    print()
    print("Sanity check: directors and how many films each directed")
    df = conn.execute(
        """
        MATCH (p:Person)-[:DIRECTED]->(m:Movie)
        RETURN p.name AS director, count(m) AS films
        ORDER BY films DESC
        """
    ).get_as_df()
    print(df.to_string(index=False))
