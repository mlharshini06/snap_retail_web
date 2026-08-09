"""
models/clothing_classifier.py
YOLO-Seg already gives us a class label per garment instance; this
module just normalizes label spelling/synonyms into one canonical
taxonomy so the UI, database, and prompt-building code all agree on
category names (e.g. "tshirt" and "t-shirt" both become "T-Shirt").
"""

from __future__ import annotations

from typing import Optional

from models.segmentation_detector import GarmentInstance

_CANONICAL = {
    "shirt": "Shirt",
    "t-shirt": "T-Shirt",
    "tshirt": "T-Shirt",
    "hoodie": "Hoodie",
    "sweater": "Sweater",
    "sweatshirt": "Sweatshirt",
    "dress": "Dress",
    "coat": "Coat",
    "jacket": "Jacket",
    "blazer": "Blazer",
    "jeans": "Jeans",
    "pants": "Pants",
    "trousers": "Trousers",
    "skirt": "Skirt",
    "shorts": "Shorts",
    "top": "Top",
    "blouse": "Blouse",
    "suit": "Suit",
    "vest": "Vest",
}


def classify(garment: Optional[GarmentInstance]) -> str:
    """Return a human-readable, canonical clothing type, or 'Unknown'
    if nothing was detected."""
    if garment is None:
        return "Unknown"
    key = garment.class_name.strip().lower()
    return _CANONICAL.get(key, garment.class_name.title())


def confidence_label(garment: Optional[GarmentInstance]) -> str:
    if garment is None:
        return ""
    return f"{classify(garment)} ({garment.confidence * 100:.0f}%)"
