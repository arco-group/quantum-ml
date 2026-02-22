#!/usr/bin/env python3
import argparse
import os
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import List
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

DEFAULT_URL = "https://physionet.org/physiobank/database/bidmc/bidmc_csv/"
USER_AGENT = "Mozilla/5.0 (compatible; data_download/1.0)"


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: List[str] = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.links.append(value)


def fetch_links(url: str) -> List[str]:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    parser = LinkParser()
    parser.feed(html)
    return parser.links


def should_download(filename: str) -> bool:
    lower = filename.lower()
    return lower.endswith(".csv") or lower.endswith(".csv.gz")


def download_file(url: str, dest_dir: Path, overwrite: bool) -> None:
    filename = os.path.basename(urlparse(url).path)
    if not filename:
        return
    if not should_download(filename):
        return

    dest_path = dest_dir / filename
    if dest_path.exists() and not overwrite:
        print(f"skip  {filename} (exists)")
        return

    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req) as resp:
        tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
        with open(tmp_path, "wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        tmp_path.replace(dest_path)

    print(f"saved {filename}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download BIDMC CSV data into Data folder.")
    parser.add_argument("--url", default=DEFAULT_URL, help="Base URL to download from")
    parser.add_argument("--dest", default="Data", help="Destination folder (relative to script)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    base_url = args.url
    if not base_url.endswith("/"):
        base_url += "/"

    root = Path(__file__).resolve().parent
    dest_dir = root / args.dest
    dest_dir.mkdir(parents=True, exist_ok=True)

    try:
        links = fetch_links(base_url)
    except Exception as exc:
        print(f"error: failed to fetch index: {exc}", file=sys.stderr)
        return 1

    file_links = [urljoin(base_url, href) for href in links]
    file_links = [link for link in file_links if should_download(os.path.basename(urlparse(link).path))]

    if not file_links:
        print("warning: no CSV links found")
        return 0

    for link in sorted(set(file_links)):
        try:
            download_file(link, dest_dir, args.overwrite)
        except Exception as exc:
            print(f"error: failed to download {link}: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
