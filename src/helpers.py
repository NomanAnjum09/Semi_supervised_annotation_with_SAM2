# ---------- Saved-object schema ----------
from dataclasses import dataclass


@dataclass
class Click:
    x: int
    y: int
    label: int  # 1 = add, 0 = subtract

@dataclass
class ObjRecord:
    class_name: str
    clicks: list[Click]
    score: float
