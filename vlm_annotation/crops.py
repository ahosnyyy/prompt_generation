"""Extract JPEG crops from full images using normalized bboxes."""

from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image

from src.bbox_annotation import Bbox


def bbox_to_pixel_box(bbox: Bbox, width: int, height: int) -> tuple[int, int, int, int]:
    x0 = int(max(0, bbox.x_min * width))
    y0 = int(max(0, bbox.y_min * height))
    x1 = int(min(width, bbox.x_max * width))
    y1 = int(min(height, bbox.y_max * height))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid pixel box: ({x0},{y0})-({x1},{y1})")
    return x0, y0, x1, y1


def crop_normalized(image: Image.Image, bbox: Bbox) -> Image.Image:
    w, h = image.size
    x0, y0, x1, y1 = bbox_to_pixel_box(bbox, w, h)
    return image.crop((x0, y0, x1, y1))


def crop_to_jpeg_bytes(image: Image.Image, bbox: Bbox, quality: int = 85) -> bytes:
    crop = crop_normalized(image, bbox)
    buf = io.BytesIO()
    crop.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


JPEG_MIME = "image/jpeg"


def encode_jpeg_base64(jpeg_bytes: bytes) -> str:
    return base64.b64encode(jpeg_bytes).decode("ascii")


def save_crop(image: Image.Image, bbox: Bbox, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(crop_to_jpeg_bytes(image, bbox))


def save_jpeg_bytes(jpeg_bytes: bytes, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(jpeg_bytes)


def get_image_mime(path: Path) -> str:
    ext = path.suffix.lstrip(".").lower()
    if ext in ("jpg", "jpeg"):
        return "image/jpeg"
    return f"image/{ext}"
