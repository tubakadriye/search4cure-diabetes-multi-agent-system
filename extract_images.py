"""Extract referenced raster images with document-version and page provenance."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import re
import tempfile

import pymupdf

from download_pdf import CHUNK_BYTES, save_metadata

DEFAULT_MAX_PIXELS = 25_000_000


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(CHUNK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _save_png(folder, data):
    """Publish a content-addressed image atomically; preserve valid existing assets."""
    checksum = hashlib.sha256(data).hexdigest()
    destination = folder / (checksum + ".png")
    if destination.exists():
        if _file_hash(destination) != checksum:
            raise ValueError("An existing PNG does not match its content-addressed filename.")
        return destination, checksum
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=folder, suffix=".part", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination, checksum


def _extract_png(pdf, xref, folder, max_pixels):
    """Decode one image, apply a soft mask if present, and store a PNG derivative."""
    image = pdf.extract_image(xref)
    if not image or not image.get("image"):
        raise ValueError("The PDF image object could not be extracted.")
    if image["width"] * image["height"] > max_pixels:
        raise ValueError("The extracted image exceeds the configured pixel limit.")
    pixmap = pymupdf.Pixmap(image["image"])
    if pixmap.colorspace is None:
        raise ValueError("Images without a standalone color space are not supported.")
    converted = pixmap.colorspace.n not in (1, 3)
    if converted:
        pixmap = pymupdf.Pixmap(pymupdf.csRGB, pixmap)
    mask_xref = image.get("smask", 0)
    if mask_xref:
        mask = pymupdf.Pixmap(pdf, mask_xref)
        if mask.width != pixmap.width or mask.height != pixmap.height:
            raise ValueError("Image and transparency mask dimensions differ.")
        pixmap = pymupdf.Pixmap(pixmap, mask)
    data = pixmap.tobytes("png")
    path, checksum = _save_png(folder, data)
    return {
        "local_path": str(path), "sha256": checksum, "size_bytes": len(data),
        "width": pixmap.width, "height": pixmap.height,
        "alpha": bool(pixmap.alpha), "converted_to_rgb": converted,
        "source_image_sha256": hashlib.sha256(image["image"]).hexdigest(),
        "source_format": image["ext"], "soft_mask_xref": mask_xref,
    }


def extract_images(metadata_path, output_dir=None, max_pixels=DEFAULT_MAX_PIXELS):
    """Verify a downloaded PDF and record successful, failed, and unsupported occurrences."""
    if type(max_pixels) is not int or max_pixels < 1:
        raise ValueError("max_pixels must be a positive integer.")
    metadata_path = Path(metadata_path)
    download = json.loads(metadata_path.read_text(encoding="utf-8"))
    try:
        document = download["search"]["document"]
        identifier, version = document["arxiv_id"], document["version"]
        pdf_path, expected_hash = Path(download["local_path"]), download["sha256"]
    except (KeyError, TypeError) as error:
        raise ValueError("The download record is missing required metadata.") from error
    if download.get("status") != "downloaded" or not isinstance(identifier, str) or not identifier:
        raise ValueError("A successful download record with an arXiv identifier is required.")
    if type(version) is not int or version < 1:
        raise ValueError("An explicit document version is required.")
    if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise ValueError("A valid SHA-256 is required in the download record.")
    folder = Path(output_dir) if output_dir is not None else pdf_path.parent / "images"
    assets = folder / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    parser = {
        "name": "PyMuPDF", "version": pymupdf.VersionBind,
        "method": "get_image_info(xrefs=True) + extract_image + Pixmap PNG",
        "output_format": "png", "apply_soft_masks": True, "max_pixels": max_pixels,
        "page_numbering": "physical PDF pages, starting at 1",
        "coordinates": "unrotated PyMuPDF page coordinates, in points",
        "inline_or_unresolved_xrefs": "record as unsupported; no fallback rendering",
        "page_rendering": False, "figure_detection": False, "ocr": False,
    }
    result = {
        "source_metadata_path": str(metadata_path), "download": download, "parser": parser,
        "status": "failed", "error": None, "page_count": None, "pages": [], "images": [],
    }
    failure = None
    try:
        logging.info("Verifying PDF hash for image extraction: %s", pdf_path)
        if _file_hash(pdf_path) != expected_hash:
            raise ValueError("The saved PDF does not match the recorded SHA-256.")
        cache = {}
        with pymupdf.open(pdf_path) as pdf:
            if not pdf.is_pdf or pdf.needs_pass:
                raise ValueError("An unencrypted PDF is required.")
            result["parser_repaired_pdf"] = pdf.is_repaired
            result["page_count"] = pdf.page_count
            if pdf.is_repaired:
                logging.warning("The parser repaired the PDF while opening it")
            for page_number, page in enumerate(pdf, 1):
                infos = page.get_image_info(xrefs=True)
                page_items = []
                for number, info in enumerate(infos, 1):
                    item = {
                        "arxiv_id": identifier, "version": version,
                        "source_pdf_sha256": expected_hash, "page_number": page_number,
                        "image_number": number, "xref": info.get("xref", 0),
                        "bbox": list(info["bbox"]), "transform": list(info["transform"]),
                        "status": "failed", "error": None,
                    }
                    identity = {**item, "parser": parser}
                    item["image_id"] = hashlib.sha256(
                        json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
                    try:
                        if not item["xref"]:
                            item.update(status="unsupported", error="No extractable image xref; inline or unresolved image.")
                        else:
                            if info["width"] * info["height"] > max_pixels:
                                raise ValueError("Image exceeds the configured pixel limit.")
                            if item["xref"] not in cache:
                                cache[item["xref"]] = _extract_png(pdf, item["xref"], assets, max_pixels)
                            item.update(cache[item["xref"]], status="extracted")
                    except Exception as error:
                        # Preserve other successful images and record this occurrence's failure.
                        item["error"] = f"{type(error).__name__}: {error}"
                    if item["status"] != "extracted":
                        logging.warning("Page %s image %s: %s: %s", page_number, number,
                                        item["status"], item["error"])
                    page_items.append(item)
                result["images"].extend(page_items)
                extracted = sum(item["status"] == "extracted" for item in page_items)
                result["pages"].append({"page_number": page_number, "image_count": len(page_items),
                                        "extracted_count": extracted})
                logging.info("Page %s/%s: %s raster occurrences; %s extracted", page_number,
                             pdf.page_count, len(page_items), extracted)
        if any(item["status"] != "extracted" for item in result["images"]):
            result["status"] = "partial"
        else:
            result["status"] = "extracted" if result["images"] else "no_images"
    except Exception as error:
        failure = error
        result["error"] = f"{type(error).__name__}: {error}"
    result["processed_at"] = datetime.now(timezone.utc).isoformat()
    result["extracted_count"] = sum(item["status"] == "extracted" for item in result["images"])
    result["failed_count"] = sum(item["status"] == "failed" for item in result["images"])
    result["unsupported_count"] = sum(item["status"] == "unsupported" for item in result["images"])
    result["unique_asset_count"] = len({item["sha256"] for item in result["images"] if item["status"] == "extracted"})
    path = save_metadata(folder, result)
    logging.info("Image extraction record saved: %s", path)
    if failure is not None:
        raise ValueError(f"Image extraction failed; details: {path}: {failure}") from failure
    return {**result, "metadata_path": str(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, help="JSON record produced by download_pdf")
    parser.add_argument("--output-dir", help="Defaults to images beside the PDF")
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        result = extract_images(args.metadata, args.output_dir, args.max_pixels)
        summary = {key: value for key, value in result.items() if key not in ("images", "download", "pages")}
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0 if result["status"] in ("extracted", "no_images") else 1
    except (OSError, ValueError) as error:
        parser.exit(1, f"PDF image extraction failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
