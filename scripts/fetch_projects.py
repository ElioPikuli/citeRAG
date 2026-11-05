"""Download the portfolio projects that CiteRAG indexes into ./corpus (git-ignored).

Everything comes from public sources, so anyone can rebuild the index:
- project repos are shallow-cloned from GitHub; Dynamic Labyrinths is a ~1 GB Unity repo, so it gets a
  sparse, blob-filtered clone of its README and gameplay scripts only (~3 MB);
- the portfolio site's repo is private, so its pages are downloaded from the live site instead.

Usage:  python scripts/fetch_projects.py
Then:   citerag ingest
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path

GITHUB = "https://github.com/ElioPikuli"
PORTFOLIO_SITE = "https://elio-pikuli.vercel.app"
PORTFOLIO_PAGES = ["index.html", "script.js"]

# local dir name -> (repo name, sparse-checkout patterns or None for the whole repo)
REPOS: dict[str, tuple[str, list[str] | None]] = {
    "ConcreteCalc": ("ConcreteCalc", None),
    "health-monitor-dashboard": ("health-monitor-dashboard", None),
    "devops": ("devops", None),
    "AIVoiceAssistant": ("AIVoiceAssistant", None),
    "DynamicLabyrinths": (
        "Dynamic-Labyrinths---Asymmetric-Hunter-vs.-Builder",
        ["/README.md", "/DynamicLab/Assets/*.cs"],
    ),
}


def git(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True)


def fetch_repo(dest: Path, name: str, repo: str, sparse: list[str] | None) -> None:
    target = dest / name
    url = f"{GITHUB}/{repo}.git"
    if (target / ".git").exists():
        print(f"updating {name}")
        git("pull", "--ff-only", "--quiet", cwd=target)
        return
    print(f"cloning {name}")
    if sparse is None:
        git("clone", "--depth", "1", "--quiet", url, str(target))
        return
    git("clone", "--depth", "1", "--filter=blob:none", "--no-checkout", "--quiet", url, str(target))
    git("sparse-checkout", "set", "--no-cone", *sparse, cwd=target)
    git("checkout", "--quiet", cwd=target)


def fetch_site(dest: Path) -> None:
    target = dest / "Portfolio"
    target.mkdir(parents=True, exist_ok=True)
    print("downloading Portfolio (live site)")
    for page in PORTFOLIO_PAGES:
        req = urllib.request.Request(f"{PORTFOLIO_SITE}/{page}", headers={"User-Agent": "citerag-fetch"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            (target / page).write_bytes(resp.read())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=Path, default=Path(__file__).resolve().parents[1] / "corpus",
                    help="where to put the projects (default: ./corpus)")
    args = ap.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)
    failed = []
    for name, (repo, sparse) in REPOS.items():
        try:
            fetch_repo(args.dest, name, repo, sparse)
        except subprocess.CalledProcessError:
            failed.append(name)
    try:
        fetch_site(args.dest)
    except OSError:
        failed.append("Portfolio")
    if failed:
        print(f"could not fetch: {', '.join(failed)} (ingest will skip them)", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
