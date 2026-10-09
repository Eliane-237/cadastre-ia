"""Lecteurs de documents : PaddleOCR-VL-1.6 (page, tableau) et Qwen3-VL-8B (JSON imposé)."""

from model.readers.base import DocumentReading, FieldReading, PointReading, Reader, SideReading

__all__ = ["DocumentReading", "FieldReading", "PointReading", "Reader", "SideReading"]
