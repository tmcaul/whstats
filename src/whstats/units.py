import dataclasses
import re
import functools

import pandas as pd

from whstats.sim import Dice, Unit, Weapon

UNITS_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQgZm_bSjPn7SCHcXr-4ZE8U8AQBmCdEP1RiBw0HPDPJlItxYlPHPmSejUodL1ClozkB2AsvyiG8VBn/pub?gid=1228841000&single=true&output=csv"
WEAPONS_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQgZm_bSjPn7SCHcXr-4ZE8U8AQBmCdEP1RiBw0HPDPJlItxYlPHPmSejUodL1ClozkB2AsvyiG8VBn/pub?gid=1209466588&single=true&output=csv"
LOADOUTS_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQgZm_bSjPn7SCHcXr-4ZE8U8AQBmCdEP1RiBw0HPDPJlItxYlPHPmSejUodL1ClozkB2AsvyiG8VBn/pub?gid=550785005&single=true&output=csv"

_DICE_RE = re.compile(r"^(\d+)d(\d+)(?:\+(\d+))?$")


def _parse_value(raw):
    """Parse a weapon "attacks"/"damage" cell: either a plain int or a dice string like '1d6+2'."""
    s = str(raw).strip()
    match = _DICE_RE.match(s)
    if match:
        n, sides, plus = match.groups()
        return Dice(int(n), int(sides), int(plus) if plus else 0)
    return int(s)


def _parse_optional_int(raw):
    if pd.isna(raw) or str(raw).strip() == "":
        return None
    return int(raw)


def _parse_bool(raw):
    return str(raw).strip().lower() == "true"


def _weapon_key(unit) -> str | None:
    """None means the weapon is generic (shared across units), not scoped to one unit."""
    if pd.isna(unit) or str(unit).strip() == "":
        return None
    return unit


@functools.lru_cache(maxsize=1)
def load_armies() -> dict[str, list[Unit]]:
    units_df = pd.read_csv(UNITS_URL).set_index("unit")
    weapons_df = pd.read_csv(WEAPONS_URL)
    loadouts_df = pd.read_csv(LOADOUTS_URL)

    weapon_templates: dict[tuple[str | None, str], Weapon] = {}
    for _, row in weapons_df.iterrows():
        weapon = row["weapon"]
        weapon_templates[(_weapon_key(row["unit"]), weapon)] = Weapon(
            name=weapon,
            attacks=_parse_value(row["attacks"]),
            skill=int(row["skill"]),
            strength=int(row["strength"]),
            ap=int(row["ap"]),
            damage=_parse_value(row["damage"]),
            phase=row["phase"],
            torrent=_parse_bool(row["torrent"]),
            wound_bonus=int(row["wound_bonus"]),
            hit_bonus=int(row["hit_bonus"]),
            sustained_hits=_parse_bool(row["sustained_hits"]),
            lethal_hits=_parse_bool(row["lethal_hits"]),
            reroll_hit_ones=_parse_bool(row["reroll_hit_ones"]),
            reroll_all_hits=_parse_bool(row["reroll_all_hits"]),
            reroll_wound_ones=_parse_bool(row["reroll_wound_ones"]),
            reroll_all_wounds=_parse_bool(row["reroll_all_wounds"]),
        )

    armies: dict[str, list[Unit]] = {}
    for display_name, loadout_rows in loadouts_df.groupby("display_name"):
        army = loadout_rows["army"].iloc[0]
        unit_name = loadout_rows["unit"].iloc[0]
        urow = units_df.loc[unit_name]

        weapons = []
        for _, wrow in loadout_rows.iterrows():
            weapon = wrow["weapon"]
            template = weapon_templates.get((unit_name, weapon)) or weapon_templates[(None, weapon)]
            weapons.append(
                dataclasses.replace(template, models_equipped=_parse_optional_int(wrow["count"]))
            )

        unit = Unit(
            name=display_name,
            toughness=int(urow["toughness"]),
            saving_throw=int(urow["saving_throw"]),
            wounds=int(urow["wounds"]),
            models=int(urow["models"]),
            weapons=weapons,
            invuln=_parse_optional_int(urow["invuln"]),
            fnp=_parse_optional_int(urow["fnp"]),
            damage_modifier=int(urow["damage_modifier"]),
        )
        armies.setdefault(army, []).append(unit)

    return armies
