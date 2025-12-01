#!/usr/bin/env python3
"""
Corpus builder: fetch Wikipedia pages for a user-specified topic/person.

- Asks user for topic/person (e.g. "Adolf Hitler").
- Discovers titles via categories + backlinks.
- Downloads plaintext extracts using async for speed.
- Writes:
    data/raw/<topic_slug>/pages/*.txt
    data/raw/<topic_slug>/manifest.jsonl
"""

import os
import re
import json
import asyncio
import argparse
from collections import deque
from typing import Dict, Iterable, Set, List, Optional
from pathlib import Path

import aiohttp
from aiolimiter import AsyncLimiter
from tqdm.asyncio import tqdm as async_tqdm

DEFAULT_API = "https://en.wikipedia.org/w/api.php"
DEFAULT_USER_AGENT = "LocalLLMTopicPipeline/0.1 (research; contact: usamaahmedshus@gmail.com)"
DEFAULT_OUTPUT_DIR = "data/raw"
DEFAULT_MAX_PAGES = 500
DEFAULT_CAT_DEPTH = 2
DEFAULT_TIMEOUT = 30.0

# Rate limiting: 5 requests per second (conservative)
RATE_LIMITER = AsyncLimiter(max_rate=5, time_period=1.0)
MAX_CONCURRENT = 10  # Max concurrent requests


def clean_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)[:150]


def slugify(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", s.strip().lower()).strip("_")


class WikipediaClient:
    """Async Wikipedia API client with rate limiting."""

    def __init__(
        self,
        api_url: str = DEFAULT_API,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: float = DEFAULT_TIMEOUT,
        max_concurrent: int = MAX_CONCURRENT,
    ) -> None:
        self.api_url = api_url
        self.user_agent = user_agent
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        self.session: Optional[aiohttp.ClientSession] = None
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def __aenter__(self):
        """Async context manager entry."""
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        self.session = aiohttp.ClientSession(
            headers={"User-Agent": self.user_agent},
            timeout=timeout,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def _api_request(self, params: Dict) -> Dict:
        """Make rate-limited async API request."""
        params = {**params, "format": "json"}

        async with self.semaphore:
            async with RATE_LIMITER:
                try:
                    async with self.session.get(self.api_url, params=params) as resp:
                        resp.raise_for_status()
                        return await resp.json()
                except Exception as e:
                    print(f"API request failed: {e}")
                    return {}

    async def get_category_members(
        self,
        category: str,
        depth: int = 1,
        max_pages: int = 1000,
    ) -> Set[str]:
        """Recursively get article titles from Category:<name>."""
        to_visit = deque([(f"Category:{category}", 0)])
        visited: Set[str] = set()
        pages: Set[str] = set()

        while to_visit and len(pages) < max_pages:
            cat, level = to_visit.popleft()
            if cat in visited:
                continue
            visited.add(cat)

            cont = None
            while len(pages) < max_pages:
                params = {
                    "action": "query",
                    "list": "categorymembers",
                    "cmtitle": cat,
                    "cmnamespace": "0|14",  # Articles and categories
                    "cmlimit": "500",
                }
                if cont:
                    params["cmcontinue"] = cont

                data = await self._api_request(params)
                for cm in data.get("query", {}).get("categorymembers", []):
                    if cm["ns"] == 0:  # Article
                        pages.add(cm["title"])
                    elif cm["ns"] == 14 and level < depth:  # Subcategory
                        to_visit.append((cm["title"], level + 1))

                if "continue" in data:
                    cont = data["continue"].get("cmcontinue")
                else:
                    break

                if len(pages) >= max_pages:
                    break

        return pages

    async def get_backlinks(self, title: str, limit: int = 1000) -> Set[str]:
        """Pages linking to a given title."""
        results: Set[str] = set()
        cont = None

        while len(results) < limit:
            params = {
                "action": "query",
                "list": "backlinks",
                "bltitle": title,
                "blnamespace": 0,
                "bllimit": "500",
            }
            if cont:
                params["blcontinue"] = cont

            data = await self._api_request(params)
            for b in data.get("query", {}).get("backlinks", []):
                results.add(b["title"])

            if "continue" in data and len(results) < limit:
                cont = data["continue"].get("blcontinue")
            else:
                break

        return results

    async def get_page_text(self, title: str) -> str:
        """Return plaintext extract for a given page."""
        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": "1",
            "titles": title,
        }
        data = await self._api_request(params)
        pages = data.get("query", {}).get("pages", {})
        for _, p in pages.items():
            return p.get("extract", "") or ""
        return ""


class CorpusBuilder:
    """High-level helper to build a Wikipedia corpus for a single topic."""

    def __init__(
        self,
        client: WikipediaClient,
        output_dir: str = DEFAULT_OUTPUT_DIR,
    ) -> None:
        self.client = client
        self.output_dir = output_dir

    async def discover_titles(
        self,
        seed_title: str,
        cat_depth: int,
        max_pages: int,
    ) -> Set[str]:
        """Discover titles through categories and backlinks in parallel."""
        titles: Set[str] = set()

        # Run discovery tasks in parallel, each with its own limit
        cat_task = self.client.get_category_members(
            category=seed_title,
            depth=cat_depth,
            max_pages=max_pages,
        )
        backlinks_task = self.client.get_backlinks(seed_title, limit=max_pages)

        cat_titles, backlink_titles = await asyncio.gather(
            cat_task,
            backlinks_task,
            return_exceptions=True,
        )

        # Merge results
        if isinstance(cat_titles, set):
            titles |= cat_titles
        if isinstance(backlink_titles, set):
            titles |= backlink_titles

        titles.add(seed_title)

        # Enforce global cap
        if len(titles) > max_pages:
            titles = set(list(titles)[:max_pages])

        return titles

    async def download_single_page(
        self,
        title: str,
        topic_slug: str,
        topic_name: str,
        topic_dir: Path,
    ) -> Optional[Dict]:
        """Download a single page and return manifest record."""
        try:
            text = await self.client.get_page_text(title)
            if not text.strip():
                return None

            filename = clean_filename(title) + ".txt"
            path = topic_dir / filename

            # Write file
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)

            # Create manifest record
            record = {
                "topic_slug": topic_slug,
                "topic_name": topic_name,
                "title": title,
                "filename": filename,
                "rel_path": str(Path(topic_slug) / "pages" / filename),
                "source": "wikipedia",
            }
            return record

        except Exception as e:
            print(f"\nError downloading {title}: {e}")
            return None

    async def download_titles(
        self,
        topic_slug: str,
        topic_name: str,
        titles: Iterable[str],
    ) -> None:
        """Download page text in parallel and write .txt files + manifest.jsonl."""
        topic_dir = Path(self.output_dir) / topic_slug / "pages"
        topic_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = Path(self.output_dir) / topic_slug / "manifest.jsonl"

        sorted_titles = sorted(set(titles))

        print(
            f"\nDownloading {len(sorted_titles)} pages with {MAX_CONCURRENT} concurrent requests..."
        )
        print(f"Rate limit: {RATE_LIMITER.max_rate} req/sec\n")

        # Create download tasks
        tasks = [
            self.download_single_page(title, topic_slug, topic_name, topic_dir)
            for title in sorted_titles
        ]

        # Execute with progress bar
        manifest_records = []

        with open(manifest_path, "w", encoding="utf-8") as manifest_f:
            for coro in async_tqdm(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc=f"Downloading {topic_name}",
            ):
                record = await coro
                if record:
                    manifest_records.append(record)
                    manifest_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    manifest_f.flush()

        print(
            f"\n✓ Successfully downloaded {len(manifest_records)}/{len(sorted_titles)} pages"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Wikipedia corpus for a user-specified topic (async version)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Base output directory (default: data/raw).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=DEFAULT_MAX_PAGES,
        help="Max pages discovered from categories/backlinks (global cap).",
    )
    parser.add_argument(
        "--cat-depth",
        type=int,
        default=DEFAULT_CAT_DEPTH,
        help="Category traversal depth.",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        help="Topic or person, e.g. 'Winston Churchill'. If omitted, you will be prompted.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=MAX_CONCURRENT,
        help="Max concurrent requests (default: 10).",
    )
    return parser.parse_args()


async def async_main() -> None:
    """Async main function."""
    args = parse_args()

    topic_name = args.topic or input("Enter topic/person (e.g. WWII): ").strip()
    while not topic_name:
        topic_name = input("Topic cannot be empty. Enter topic/person: ").strip()

    topic_slug = slugify(topic_name)

    async with WikipediaClient(
        api_url=DEFAULT_API,
        user_agent=DEFAULT_USER_AGENT,
        timeout=DEFAULT_TIMEOUT,
        max_concurrent=args.max_concurrent,
    ) as client:
        builder = CorpusBuilder(client=client, output_dir=args.output_dir)

        print(f"\n{'='*60}")
        print(f"Discovering titles for: {topic_name}")
        print(f"Slug: {topic_slug}")
        print(f"{'='*60}\n")

        titles = await builder.discover_titles(
            seed_title=topic_name,
            cat_depth=args.cat_depth,
            max_pages=args.max_pages,
        )
        print(f"✓ Discovered {len(titles)} candidate pages (capped at {args.max_pages})")

        print(f"\n{'='*60}")
        print(f"Downloading pages for: {topic_name}")
        print(f"{'='*60}")

        await builder.download_titles(
            topic_slug=topic_slug,
            topic_name=topic_name,
            titles=titles,
        )

    print(f"\n{'='*60}")
    print("✓ Corpus build complete!")
    print(f"Output directory: {args.output_dir}/{topic_slug}/")
    print(f"{'='*60}\n")


def main() -> None:
    """Entry point that runs async main."""
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise


if __name__ == "__main__":
    main()
