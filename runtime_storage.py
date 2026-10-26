"""Use a persistent hosting disk when configured; keep local paths unchanged."""
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("STOCKSENSE_DATA_DIR") or PROJECT_DIR)
NEWS_CACHE_DIR = DATA_DIR / "news_cache"


def gmp_history_path():
    path = DATA_DIR / "gmp_history.csv"
    if DATA_DIR != PROJECT_DIR:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        seed = PROJECT_DIR / "gmp_history.csv"
        if seed.exists():
            try:
                # Exclusive creation keeps a later restart from overwriting
                # history already collected on the persistent disk.
                with path.open("xb") as output:
                    output.write(seed.read_bytes())
            except FileExistsError:
                pass
    return str(path)
