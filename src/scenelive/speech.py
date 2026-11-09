"""Words for speaking: directions and distances as a person would say them (so a speech synthesizer reads them correctly)."""
from __future__ import annotations
import re

CLOCK_WORDS = {12: "прямо по курсу", 1: "справа-спереди", 2: "справа-спереди", 3: "справа", 4: "справа-сзади", 5: "справа-сзади", 6: "сзади",
               7: "слева-сзади", 8: "слева-сзади", 9: "слева", 10: "слева-спереди", 11: "слева-спереди"}


def direction_words(clock: int) -> str:
    return CLOCK_WORDS.get(clock, "")


def metres(n: float) -> str:
    """Genitive after 'около / от / до / дальше': 1 метра, 2 метров, 5 метров, 21 метра ..., fractions: метра."""
    if abs(n - round(n)) > 1e-6: return f"{n:g} метра"
    n = int(round(n)); return f"{n} метра" if (n % 10 == 1 and n % 100 != 11) else f"{n} метров"


def range_speech(label: str | None) -> str:
    """'~5 m' -> 'около 5 метров'; '5-10 m' -> 'от 5 до 10 метров'; '15+ m' -> 'дальше 15 метров'; '?' or None -> ''."""
    if not label or label == "?": return ""
    m = re.fullmatch(r"~([\d.]+) m", label)
    if m: return "около " + metres(float(m.group(1)))
    m = re.fullmatch(r"([\d.]+)\+ m", label)
    if m: return "дальше " + metres(float(m.group(1)))
    m = re.fullmatch(r"([\d.]+)-([\d.]+) m", label)
    if m: return f"от {float(m.group(1)):g} до {metres(float(m.group(2)))}"
    return label
