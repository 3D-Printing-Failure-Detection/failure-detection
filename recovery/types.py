from dataclasses import dataclass, field
from typing import Any, Dict

from shapely.geometry.base import BaseGeometry


@dataclass(eq=False)
class Layer:
    """One printed layer: z is the top height, region the deposited material
    (Polygon/MultiPolygon, holes are real holes). Treated as immutable; derived
    quantities are memoised in `cache` (see geometry.features)."""
    z: float
    region: BaseGeometry
    cache: Dict[str, Any] = field(default_factory=dict, repr=False)
    # Retained by cache keys without retaining an entire previous layer. Unlike
    # id(layer), this identity cannot be recycled while a cache entry exists.
    cache_token: object = field(default_factory=object, init=False, repr=False)
