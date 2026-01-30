"""Abstract interfaces for external dependencies."""

from financial_model.interfaces.volume_model import VolumeModelInterface
from financial_model.interfaces.price_curve import PriceCurveInterface

__all__ = ["VolumeModelInterface", "PriceCurveInterface"]
