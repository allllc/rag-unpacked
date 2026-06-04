"""Build the Python-tips corpus used by all three notebooks in module 03.

Twelve hand-authored tips about the Python standard library. Each tip is
short, focused, and engineered to demonstrate a specific BM25 behavior:

  - tip_03 (bisect), tip_06 (__slots__), tip_09 (lru_cache): rare keywords
    that BM25's IDF reward spikes on
  - tip_01 + tip_02: shared 'sort' / 'key' vocabulary so TF saturation
    is observable
  - tip_10 + tip_11: shared 'path' vocabulary, same pattern
  - tip_04 + tip_05: both about insertion order, a near-synonym setup
  - tip_12: the synonym trap. The doc is about speeding up Python by
    avoiding global lookups, but its vocabulary doesn't overlap with
    'make my code faster' or 'optimize performance'. BM25 misses; vectors
    catch it. NB3 Act III shows hybrid resolving it.

Output: data/corpus.parquet (~6 KB).
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CORPUS_PATH = DATA_DIR / "corpus.parquet"


_SEED_CORPUS: list[dict] = [
    {
        "title": "Sort a list of dicts by a key with operator.itemgetter",
        "text": (
            "To sort a list of dicts by one of their fields, pass "
            "operator.itemgetter to the key argument of sorted. This is "
            "faster than a lambda because itemgetter is implemented in C, "
            "and it reads more clearly when the sort key is a single field. "
            "Example: sorted(rows, key=operator.itemgetter('name')) "
            "returns the rows sorted by their name field. The same trick "
            "works on lists of tuples by index."
        ),
    },
    {
        "title": "Sort with a custom key function",
        "text": (
            "Any callable can be a sort key. Pass a lambda or a function to "
            "the key argument of sorted or list.sort, and Python calls it "
            "on each element to derive the value used for comparison. The "
            "key is computed once per element, so even complex key functions "
            "are efficient on large lists. Use reverse=True to invert the "
            "order. A common pattern is sorting strings case-insensitively "
            "with key=str.lower."
        ),
    },
    {
        "title": "bisect keeps a sorted list sorted on insert",
        "text": (
            "The bisect module finds the position where a value should be "
            "inserted into a sorted list to keep it sorted, without scanning "
            "the whole list. bisect.insort inserts the value in the right "
            "place in one call. This is much faster than appending and "
            "re-sorting on every insert. Use bisect_left and bisect_right "
            "to control how ties are handled."
        ),
    },
    {
        "title": "collections.OrderedDict preserves insertion order",
        "text": (
            "collections.OrderedDict is a dict subclass that remembers the "
            "order keys were added. It supports a move_to_end method for "
            "LRU-cache-like behavior, and its equality comparison considers "
            "order, unlike a regular dict. Use it when you need explicit "
            "ordering semantics in your code. For most cases since Python "
            "3.7 a regular dict is enough."
        ),
    },
    {
        "title": "dict itself preserves insertion order since Python 3.7",
        "text": (
            "Since Python 3.7 the built-in dict preserves insertion order "
            "as a language guarantee, not just a CPython implementation "
            "detail. Iterating a dict yields keys in the order they were "
            "added. This means you can drop collections.OrderedDict in most "
            "code that targets a modern Python. Reach for OrderedDict only "
            "when you need its extra methods like move_to_end or its "
            "order-sensitive equality."
        ),
    },
    {
        "title": "Use __slots__ to shrink instance memory",
        "text": (
            "Adding __slots__ to a class tells Python which attributes "
            "instances are allowed to have, and stops it from allocating a "
            "per-instance __dict__. The memory savings can be large when "
            "you have millions of small objects. The trade-off is no dynamic "
            "attribute assignment and trickier inheritance. Use it on "
            "value-object classes where the attribute set is fixed."
        ),
    },
    {
        "title": "Walrus operator := for expression assignment",
        "text": (
            "The walrus operator := lets you assign to a name as part of "
            "an expression. The classic use is in a while loop reading "
            "lines: while (line := f.readline()): process(line). It also "
            "shines in list comprehensions where you want to compute a "
            "value once and use it twice. Available in Python 3.8 and "
            "later."
        ),
    },
    {
        "title": "dataclasses.dataclass removes __init__ boilerplate",
        "text": (
            "Apply the @dataclass decorator to a class and Python generates "
            "an __init__, __repr__, and __eq__ from the type-annotated "
            "class attributes. This replaces the usual boilerplate of "
            "writing self.x = x for every field. Optional features include "
            "frozen instances, default factories, and post-init hooks. "
            "Use it for plain data containers that don't need custom "
            "behavior."
        ),
    },
    {
        "title": "functools.lru_cache memoizes pure functions",
        "text": (
            "Decorate a function with @functools.lru_cache to cache its "
            "return values keyed by the arguments. Subsequent calls with "
            "the same arguments skip the function body and return the "
            "cached result. The cache has a configurable maxsize and "
            "evicts the least-recently-used entry when full. Works only "
            "with hashable arguments, and only on pure functions where "
            "the result depends solely on the inputs."
        ),
    },
    {
        "title": "pathlib.Path replaces os.path for new code",
        "text": (
            "pathlib.Path is an object-oriented path API that handles "
            "joining, reading, writing, and globbing through method calls "
            "instead of os.path functions. Path('/a') / 'b' joins to "
            "Path('/a/b'). It has read_text and write_text helpers for "
            "small files. Modern Python code should reach for pathlib "
            "first; fall back to os.path only when interfacing with "
            "libraries that expect string paths."
        ),
    },
    {
        "title": "os.path.join builds platform-correct paths",
        "text": (
            "os.path.join concatenates path components using the right "
            "separator for the current operating system. It's the safe "
            "way to construct paths when you're working with string paths "
            "rather than pathlib objects. os.path.exists, os.path.isfile, "
            "and os.path.dirname round out the module's most-used "
            "functions. Most new code should prefer pathlib.Path."
        ),
    },
    {
        "title": "Speed up loops by avoiding global name lookups",
        "text": (
            "Python looks up every name at runtime, and global lookups "
            "are slower than local ones. Inside a hot loop, bind functions "
            "and constants you use repeatedly to local variables before "
            "the loop starts: local_len = len, then call local_len(x) "
            "inside. The interpreter does a fast LOAD_FAST instead of a "
            "LOAD_GLOBAL on each iteration. The savings add up when the "
            "loop runs millions of times."
        ),
    },
]


def _chunk_id(text: str) -> str:
    return "tip_" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def build_seed_corpus() -> "pd.DataFrame":
    import pandas as pd

    rows = []
    for i, entry in enumerate(_SEED_CORPUS, start=1):
        text = _normalize(entry["text"])
        # Stable, readable id: "tip_01", "tip_02", ... so the demo
        # questions in the notebooks can refer to them by name.
        rows.append(
            {
                "id": f"tip_{i:02d}",
                "title": entry["title"],
                "text": text,
            }
        )
    return pd.DataFrame(rows)


def build_if_missing(force: bool = False) -> Path:
    """Idempotently materialize data/corpus.parquet. Returns the path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if CORPUS_PATH.exists() and not force:
        return CORPUS_PATH
    df = build_seed_corpus()
    df.to_parquet(CORPUS_PATH, index=False)
    return CORPUS_PATH


if __name__ == "__main__":
    force = "--force" in sys.argv
    path = build_if_missing(force=force)
    import pandas as pd

    df = pd.read_parquet(path)
    print(f"Wrote {len(df)} tips to {path}")
    print(df[["id", "title"]].to_string(index=False))
