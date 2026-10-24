"""Download and prepare Amazon 2014 data without executing dataset contents.

Raw metadata is catalog information captured at crawl time, not a historical
snapshot. Temporal splits prevent future review leakage but cannot establish
when each product description became available.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import html
import json
import math
from pathlib import Path
import random
import re
import shutil
import sys
import time
from typing import Any, Iterator
import urllib.error
import urllib.parse
import urllib.request

import numpy as np


SOURCE_PAGE = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon/datasets.html"
BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon/categoryFiles/"
DOMAINS = {
    "baby": "Baby",
    "sports": "Sports_and_Outdoors",
    "clothing": "Clothing_Shoes_and_Jewelry",
}
USER_AGENT = "ShopLens-coursework/0.1 (academic dataset preparation)"


class DataError(ValueError):
    """A source file or preparation request cannot be used as specified."""


def source_urls(domain: str) -> dict[str, str]:
    stem = DOMAINS[domain]
    return {"reviews": f"{BASE_URL}reviews_{stem}_5.json.gz",
            "metadata": f"{BASE_URL}meta_{stem}.json.gz"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, records: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def iter_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Stream strict JSON or the Python-literal dictionaries in Amazon 2014."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    record = ast.literal_eval(line)
                if not isinstance(record, dict):
                    raise ValueError("record must be a dictionary")
            except (ValueError, SyntaxError) as exc:
                raise DataError(f"{path}:{number}: invalid record: {exc}") from exc
            yield record


def clean_text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return " ".join(filter(None, (clean_text(part) for part in value)))
    if not isinstance(value, str):
        return ""
    return " ".join(re.sub(r"<[^>]+>", " ", html.unescape(value)).split())


def image_url(record: dict[str, Any]) -> str:
    candidates = [record.get("imUrl"), record.get("imageURLHighRes"),
                  record.get("imageURL"), record.get("image_url")]
    for candidate in candidates:
        values = candidate if isinstance(candidate, list) else [candidate]
        for value in values:
            if isinstance(value, str) and urllib.parse.urlparse(value).scheme in ("http", "https"):
                return value
    return ""


def download_file(url: str, destination: str | Path, *, timeout: float = 30,
                  retries: int = 3, max_bytes: int | None = None,
                  validate_gzip: bool = False) -> dict[str, Any]:
    """Resume .part files using HTTP Range; only rename complete downloads.

    An interrupted response leaves the .part file for the next invocation.
    Existing completed files are reused, never overwritten implicitly.
    """
    if urllib.parse.urlparse(url).scheme not in ("https", "http"):
        raise DataError("Download URL must use http or https")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size:
        if validate_gzip:
            with gzip.open(destination, "rb") as stream:
                shutil.copyfileobj(stream, _DiscardWriter())
        return {"url": url, "path": str(destination), "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination), "cached": True}
    partial = destination.with_name(destination.name + ".part")
    last_error: Exception | None = None
    for attempt in range(retries):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                resumed = response.status == 206
                if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise DataError("Server returned an unexpected Content-Range")
                start = offset if resumed else 0
                length = response.headers.get("Content-Length")
                expected = start + int(length) if length else None
                if max_bytes is not None and expected is not None and expected > max_bytes:
                    raise DataError(f"Download exceeds {max_bytes} byte limit")
                content_type = response.headers.get("Content-Type", "").lower()
                if "text/html" in content_type:
                    raise DataError("Server returned HTML instead of the requested data")
                with partial.open("ab" if resumed else "wb") as stream:
                    for block in iter(lambda: response.read(1024 * 1024), b""):
                        if max_bytes is not None and stream.tell() + len(block) > max_bytes:
                            raise DataError(f"Download exceeds {max_bytes} byte limit")
                        stream.write(block)
                size = partial.stat().st_size
                if expected is not None and size != expected:
                    raise DataError(f"Incomplete download: expected {expected} bytes, got {size}")
                if not size:
                    raise DataError("Server returned an empty file")
                if validate_gzip:
                    with gzip.open(partial, "rb") as stream:
                        for _ in iter(lambda: stream.read(1024 * 1024), b""):
                            pass
                partial.replace(destination)
                return {"url": url, "path": str(destination), "bytes": size,
                        "sha256": sha256_file(destination), "cached": False}
        except urllib.error.HTTPError as exc:
            # A final response may have arrived completely before an interrupted
            # caller could rename it. The server's length must agree before reuse.
            total = exc.headers.get("Content-Range", "")
            if exc.code == 416 and offset and total == f"bytes */{offset}":
                if validate_gzip:
                    with gzip.open(partial, "rb") as stream:
                        shutil.copyfileobj(stream, _DiscardWriter())
                partial.replace(destination)
                return {"url": url, "path": str(destination), "bytes": offset,
                        "sha256": sha256_file(destination), "cached": False}
            last_error = exc
            if exc.code in (400, 401, 403, 404):
                break
        except (OSError, ValueError) as exc:
            last_error = exc
        if attempt + 1 < retries:
            time.sleep(min(2 ** attempt, 4))
    raise DataError(f"Download failed for {url}: {last_error}. Partial file: {partial}")


class _DiscardWriter:
    def write(self, value: bytes) -> int:
        return len(value)


def download_domain(domain: str, raw_dir: str | Path, *, kind: str = "all",
                    reviews_url: str | None = None, metadata_url: str | None = None,
                    timeout: float = 30) -> dict[str, Any]:
    raw_dir = Path(raw_dir)
    urls = source_urls(domain)
    overrides = {"reviews": reviews_url, "metadata": metadata_url}
    results = {}
    for name in ("reviews", "metadata") if kind == "all" else (kind,):
        url = overrides[name] or urls[name]
        # Use canonical names even when an override is served by a generic endpoint.
        destination = raw_dir / Path(urllib.parse.urlparse(urls[name]).path).name
        print(f"Downloading {domain}/{name}: {url}", file=sys.stderr, flush=True)
        results[name] = download_file(url, destination, timeout=timeout, validate_gzip=True)
        write_json(raw_dir / f"{name}.source.json", {**results[name], "source_page": SOURCE_PAGE,
                                                   "domain": domain})
    return results


def _normalize_event(record: dict[str, Any]) -> dict[str, Any] | None:
    user_id, item_id = record.get("reviewerID"), record.get("asin")
    try:
        rating, timestamp = float(record["overall"]), int(record["unixReviewTime"])
    except (KeyError, ValueError, TypeError, OverflowError):
        return None
    if not user_id or not item_id or not math.isfinite(rating) or not 1 <= rating <= 5 or timestamp < 0:
        return None
    return {"user_id": str(user_id), "item_id": str(item_id), "rating": rating,
            "timestamp": timestamp}


def _image_filename(item_id: str) -> str:
    stem = item_id if re.fullmatch(r"[A-Za-z0-9_-]+", item_id) else hashlib.sha256(item_id.encode()).hexdigest()
    return stem + ".jpg"


def prepare(domain: str, reviews_file: str | Path, metadata_file: str | Path,
            processed_dir: str | Path, *, max_users: int | None = None,
            max_reviews: int | None = None, seed: int = 42,
            require_text: bool = False, require_image: bool = False,
            images_dir: str | Path | None = None, require_images: bool = False,
            min_user_events: int = 3, max_items: int | None = None) -> dict[str, Any]:
    """Prepare one catalog using global .8/.9 timestamp cutoffs.

    max_users selects a reproducible random user subset and retains their full
    histories (unless max_reviews was also requested). A capped prefix is always
    recorded as a subset, never described as the complete 5-core benchmark.
    """
    if (max_users is not None and max_users < 1 or max_reviews is not None and max_reviews < 1
            or max_items is not None and max_items < 1 or min_user_events < 1):
        raise DataError("Subset limits must be positive")
    reviews_file, metadata_file = Path(reviews_file), Path(metadata_file)
    processed_dir = Path(processed_dir)
    images_dir = Path(images_dir) if images_dir else processed_dir / "images"
    events, seen = [], set()
    counts = Counter()
    for record in iter_records(reviews_file):
        if max_reviews is not None and counts["review_records_scanned"] >= max_reviews:
            break
        counts["review_records_scanned"] += 1
        event = _normalize_event(record)
        if event is None:
            counts["invalid_reviews"] += 1
            continue
        key = (event["user_id"], event["item_id"], event["timestamp"])
        if key in seen:
            counts["duplicate_reviews"] += 1
            continue
        seen.add(key)
        events.append(event)
    all_users = sorted({event["user_id"] for event in events})
    if max_users is not None and len(all_users) > max_users:
        selected = set(random.Random(seed).sample(all_users, max_users))
        events = [event for event in events if event["user_id"] in selected]
    if max_items is not None:
        catalog = sorted({event["item_id"] for event in events})
        selected_items = set(random.Random(seed).sample(catalog, min(max_items, len(catalog))))
        events = [event for event in events if event["item_id"] in selected_items]
    needed = {event["item_id"] for event in events}
    items = {}
    for record in iter_records(metadata_file):
        counts["metadata_records_scanned"] += 1
        item_id = str(record.get("asin", ""))
        if item_id not in needed or item_id in items:
            continue
        title, description = clean_text(record.get("title")), clean_text(record.get("description"))
        path = images_dir / _image_filename(item_id)
        items[item_id] = {"item_id": item_id, "title": title, "description": description,
                          "text": " ".join(filter(None, (title, description))),
                          "image_url": image_url(record),
                          "image_path": str(path.resolve()) if path.exists() else "",
                          "metadata": {name: record[name] for name in ("brand", "categories", "price") if name in record}}
        if len(items) == len(needed):
            break
    counts["missing_metadata_before_filter"] = len(needed - items.keys())
    for item_id in needed - items.keys():
        items[item_id] = {"item_id": item_id, "title": "", "description": "", "text": "",
                          "image_url": "", "image_path": "", "metadata": {}}
    counts["missing_text_before_filter"] = sum(not item["text"] for item in items.values())
    counts["missing_image_url_before_filter"] = sum(not item["image_url"] for item in items.values())
    if require_images:
        for item in items.values():
            try:
                if not item["image_path"]:
                    raise DataError("No downloaded image")
                _validate_image(Path(item["image_path"]))
            except (OSError, ValueError):
                item["image_path"] = ""
    items = {item_id: item for item_id, item in items.items()
             if (not require_text or item["text"]) and (not require_image or item["image_url"])
             and (not require_images or item["image_path"])}
    events = [event for event in events if event["item_id"] in items]
    user_counts = Counter(event["user_id"] for event in events if event["rating"] >= 4)
    events = [event for event in events if user_counts[event["user_id"]] >= min_user_events]
    remaining_items = {event["item_id"] for event in events}
    items = {item_id: item for item_id, item in items.items() if item_id in remaining_items}
    if not events:
        raise DataError("No valid interactions remain after selection and metadata filtering")
    events.sort(key=lambda event: (event["timestamp"], event["user_id"], event["item_id"]))
    timestamps = np.array([event["timestamp"] for event in events], dtype=np.int64)
    train_end, validation_end = (int(value) for value in np.quantile(timestamps, [.8, .9], method="lower"))
    for event in events:
        event["split"] = ("train" if event["timestamp"] <= train_end else
                          "validation" if event["timestamp"] <= validation_end else "test")
    event_by_key = {(event["user_id"], event["item_id"], event["timestamp"]): event for event in events}
    review_count = 0

    def train_reviews() -> Iterator[dict[str, Any]]:
        nonlocal review_count
        exported = set()
        for number, record in enumerate(iter_records(reviews_file)):
            if max_reviews is not None and number >= max_reviews:
                break
            event = _normalize_event(record)
            if event is None:
                continue
            key = (event["user_id"], event["item_id"], event["timestamp"])
            normalized = event_by_key.get(key)
            if normalized is None or normalized["split"] != "train" or key in exported:
                continue
            exported.add(key)
            text = " ".join(filter(None, (clean_text(record.get("summary")), clean_text(record.get("reviewText")))))
            if text:
                review_count += 1
                yield {**event, "text": text}

    write_jsonl(processed_dir / "reviews_train.jsonl", train_reviews())
    write_jsonl(processed_dir / "items.jsonl", (items[key] for key in sorted(items)))
    write_jsonl(processed_dir / "events.jsonl", events)
    split_counts = Counter(event["split"] for event in events)
    positive_counts = Counter(event["split"] for event in events if event["rating"] >= 4)
    eligible = Counter()
    histories: dict[str, set[str]] = defaultdict(set)
    # Process a timestamp as a group: tied reviews are never historical evidence
    # for one another. The model/evaluation code must use the same strict rule.
    start = 0
    while start < len(events):
        end = start + 1
        while end < len(events) and events[end]["timestamp"] == events[start]["timestamp"]:
            end += 1
        for event in events[start:end]:
            if event["rating"] >= 4 and histories[event["user_id"]] - {event["item_id"]}:
                eligible[event["split"]] += 1
        for event in events[start:end]:
            if event["rating"] >= 4:
                histories[event["user_id"]].add(event["item_id"])
        start = end
    audit = {
        "source": "Amazon Reviews 2014 (real data)", "source_page": SOURCE_PAGE, "domain": domain,
        "synthetic": False, "scope": "subset" if any(value is not None for value in (max_users, max_reviews, max_items)) else "full-input",
        "subset": {"max_users": max_users, "max_reviews": max_reviews, "max_items": max_items, "seed": seed},
        "input_files": {"reviews": {"path": str(reviews_file.resolve()), "sha256": sha256_file(reviews_file)},
                        "metadata": {"path": str(metadata_file.resolve()), "sha256": sha256_file(metadata_file)}},
        "counts": {**dict(counts), "items": len(items), "users": len({event["user_id"] for event in events}),
                   "events": len(events), "splits": dict(split_counts), "positives": dict(positive_counts),
                   "positive_targets_with_prior_history": dict(eligible), "train_reviews": review_count,
                   "missing_text": sum(not item["text"] for item in items.values()),
                   "missing_image_url": sum(not item["image_url"] for item in items.values()),
                   "local_images": sum(bool(item["image_path"]) for item in items.values())},
        "cutoffs": {"train_end": train_end, "validation_end": validation_end,
                    "quantiles": [.8, .9], "quantile_method": "lower", "ties": "earlier split"},
        "filters": {"require_text": require_text, "require_image_url": require_image,
                    "require_decodable_local_images": require_images,
                    "min_user_positive_events": min_user_events},
        "protocol_notes": [
            "Global temporal split; train timestamp <= train_end, validation <= validation_end, test later.",
            "Positive feedback is rating >= 4. Unobserved items are not explicit dislikes.",
            "Histories require strictly earlier positive timestamps and exclude the target item.",
            "reviews_train.jsonl contains training-period reviews only; item text is title + description only.",
            "Catalog metadata is a static crawl snapshot; historical availability of descriptions is unknown.",
            "The three domains share the Amazon source and are not three independently sourced datasets.",
            "Random user/item subsampling uses the supplied seed; a review cap takes a source-order prefix.",
            "Filtering/subsetting can destroy the original 5-core property; no re-core guarantee is claimed.",
            "Image URL presence does not establish image download or decoding success; audit-images checks this.",
        ],
    }
    write_json(processed_dir / "audit.json", audit)
    return audit


def _validate_image(path: Path) -> None:
    try:
        from PIL import Image
    except ImportError:
        raise DataError("Image verification requires Pillow; install project dependencies") from None
    with Image.open(path) as image:
        if image.width < 1 or image.height < 1:
            raise DataError("Invalid image dimensions")
        image.verify()


def audit_images(processed_dir: str | Path, *, images_dir: str | Path | None = None,
                 limit: int | None = 50, seed: int = 42, download: bool = False,
                 workers: int = 4, timeout: float = 15) -> dict[str, Any]:
    processed_dir = Path(processed_dir)
    images_dir = Path(images_dir) if images_dir else processed_dir / "images"
    items = list(iter_records(processed_dir / "items.jsonl"))
    if limit is not None and limit < 1:
        raise DataError("Image limit must be positive, or use --all")
    selected = random.Random(seed).sample(items, min(limit, len(items))) if limit is not None else items

    def inspect(item: dict[str, Any]) -> dict[str, Any]:
        result = {"item_id": item["item_id"], "url": item.get("image_url", "")}
        existing = Path(item["image_path"]) if item.get("image_path") else None
        path = existing if existing is not None and existing.exists() else images_dir / _image_filename(item["item_id"])
        try:
            if not path.exists():
                if not item.get("image_url"):
                    return {**result, "status": "missing_url"}
                if not download:
                    return {**result, "status": "not_downloaded"}
                download_file(item["image_url"], path, timeout=timeout, retries=2, max_bytes=20 * 1024 * 1024)
            _validate_image(path)
            return {**result, "status": "ok", "image_path": str(path.resolve()),
                    "bytes": path.stat().st_size}
        except Exception as exc:
            return {**result, "status": "error", "error": str(exc)}

    with ThreadPoolExecutor(max_workers=max(1, min(workers, 16))) as pool:
        results = list(pool.map(inspect, selected))
    verified = {entry["item_id"]: entry["image_path"] for entry in results if entry["status"] == "ok"}
    failures = {entry["item_id"] for entry in results if entry["status"] == "error"}
    for item in items:
        if item["item_id"] in verified:
            item["image_path"] = verified[item["item_id"]]
        elif item["item_id"] in failures:
            item["image_path"] = ""
    write_jsonl(processed_dir / "items.jsonl", items)
    report = {"seed": seed, "catalog_items": len(items), "sample_items": len(results),
              "all_items_checked": len(results) == len(items), "download_requested": download,
              "counts": dict(Counter(entry["status"] for entry in results)), "results": results,
              "note": "Sample success rate describes sampled catalog items, not only items with URLs."}
    write_json(processed_dir / "images_audit.json", report)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("download", help="Resume real Amazon 2014 downloads")
    command.add_argument("--domain", choices=DOMAINS, required=True)
    command.add_argument("--raw-dir", type=Path, required=True)
    command.add_argument("--kind", choices=("all", "reviews", "metadata"), default="all")
    command.add_argument("--reviews-url")
    command.add_argument("--metadata-url", help="Override if the original metadata mirror is unavailable")
    command.add_argument("--timeout", type=float, default=30)
    command = commands.add_parser("prepare", help="Map catalog by ASIN and split chronologically")
    command.add_argument("--domain", choices=DOMAINS, required=True)
    command.add_argument("--raw-dir", type=Path)
    command.add_argument("--reviews-file", type=Path)
    command.add_argument("--metadata-file", type=Path)
    command.add_argument("--processed-dir", type=Path, required=True)
    command.add_argument("--images-dir", type=Path)
    command.add_argument("--max-users", type=int)
    command.add_argument("--max-reviews", type=int)
    command.add_argument("--max-items", type=int)
    command.add_argument("--min-user-events", type=int, default=3, help="Minimum positive events after item filtering")
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--require-text", action="store_true")
    command.add_argument("--require-image", action="store_true", help="Require image URL; does not verify decoding")
    command.add_argument("--require-images", action="store_true", help="Require downloaded and decodable local images")
    command = commands.add_parser("audit-images", help="Sample or download and decode catalog images")
    command.add_argument("--processed-dir", type=Path, required=True)
    command.add_argument("--images-dir", type=Path)
    command.add_argument("--limit", type=int, default=50)
    command.add_argument("--all", action="store_true", help="Check every item (can trigger a large download)")
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--download", action="store_true")
    command.add_argument("--workers", type=int, default=4)
    command.add_argument("--timeout", type=float, default=15)
    args = parser.parse_args(argv)
    try:
        if args.command == "download":
            result = download_domain(args.domain, args.raw_dir, kind=args.kind,
                                     reviews_url=args.reviews_url, metadata_url=args.metadata_url, timeout=args.timeout)
        elif args.command == "prepare":
            urls = source_urls(args.domain)
            reviews = args.reviews_file or (args.raw_dir / Path(urls["reviews"]).name if args.raw_dir else None)
            metadata = args.metadata_file or (args.raw_dir / Path(urls["metadata"]).name if args.raw_dir else None)
            if reviews is None or metadata is None:
                parser.error("prepare requires --raw-dir or both --reviews-file and --metadata-file")
            result = prepare(args.domain, reviews, metadata, args.processed_dir, max_users=args.max_users,
                             max_reviews=args.max_reviews, seed=args.seed, require_text=args.require_text,
                             require_image=args.require_image, images_dir=args.images_dir,
                             require_images=args.require_images, min_user_events=args.min_user_events,
                             max_items=args.max_items)
        else:
            result = audit_images(args.processed_dir, images_dir=args.images_dir,
                                  limit=None if args.all else args.limit, seed=args.seed,
                                  download=args.download, workers=args.workers, timeout=args.timeout)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (DataError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
