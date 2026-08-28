"""Tests for whstats.units.load_armies, mocking pd.read_csv so no network
access to the published Google Sheet is required."""

import io

import pandas as pd
import pytest

from whstats import units
from whstats.sim import Dice


UNITS_CSV = """unit,toughness,saving_throw,wounds,models,invuln,fnp,damage_modifier,hit_bonus,wound_bonus
Grunt Squad,4,3,1,5,,,0,-1,0
Heavy Squad,7,2,3,3,4,5,-1,0,1
"""

WEAPONS_CSV = """unit,weapon,attacks,skill,strength,ap,damage,phase,torrent,wound_bonus,hit_bonus,sustained_hits,lethal_hits,devastating_wounds,reroll_hit_ones,reroll_all_hits,reroll_wound_ones,reroll_all_wounds
,Lasgun,1,3,3,0,1,ranged,false,0,0,false,false,false,false,false,false,false
,Combat Knife,2,3,3,0,1,melee,false,0,0,true,true,false,false,false,false,false
Heavy Squad,Plasma Cannon,1d3,4,8,-3,2,ranged,false,0,0,false,false,true,false,false,false,false
"""

LOADOUTS_CSV = """display_name,army,unit,weapon,count
Grunt Squad A,Imperium,Grunt Squad,Lasgun,5
Grunt Squad A,Imperium,Grunt Squad,Combat Knife,5
Heavy Squad A,Imperium,Heavy Squad,Plasma Cannon,3
"""


@pytest.fixture(autouse=True)
def mock_sheets(monkeypatch):
    """Route the three published-CSV URLs to small in-memory fixtures.

    Parses the fixture CSVs with the real pd.read_csv *before* patching, so
    the patched version below doesn't recurse into itself (units.pd is the
    same module object as this file's `pd`).
    """
    frames = {
        units.UNITS_URL: pd.read_csv(io.StringIO(UNITS_CSV)),
        units.WEAPONS_URL: pd.read_csv(io.StringIO(WEAPONS_CSV)),
        units.LOADOUTS_URL: pd.read_csv(io.StringIO(LOADOUTS_CSV)),
    }

    def fake_read_csv(url, *args, **kwargs):
        return frames[url].copy()

    monkeypatch.setattr(units.pd, "read_csv", fake_read_csv)
    units.load_armies.cache_clear()
    yield
    units.load_armies.cache_clear()


def test_load_armies_groups_units_by_army():
    armies = units.load_armies()

    assert list(armies.keys()) == ["Imperium"]
    assert {u.name for u in armies["Imperium"]} == {"Grunt Squad A", "Heavy Squad A"}


def test_unit_level_stats_are_parsed_from_the_units_sheet():
    armies = units.load_armies()
    heavy = next(u for u in armies["Imperium"] if u.name == "Heavy Squad A")

    assert heavy.toughness == 7
    assert heavy.saving_throw == 2
    assert heavy.wounds == 3
    assert heavy.models == 3
    assert heavy.invuln == 4
    assert heavy.fnp == 5
    assert heavy.damage_modifier == -1
    assert heavy.hit_bonus == 0
    assert heavy.wound_bonus == 1


def test_unit_with_blank_invuln_and_fnp_parses_to_none():
    armies = units.load_armies()
    grunt = next(u for u in armies["Imperium"] if u.name == "Grunt Squad A")

    assert grunt.invuln is None
    assert grunt.fnp is None
    assert grunt.hit_bonus == -1
    assert grunt.wound_bonus == 0


def test_generic_weapon_is_shared_across_units():
    armies = units.load_armies()
    grunt = next(u for u in armies["Imperium"] if u.name == "Grunt Squad A")
    weapon_names = {w.name for w in grunt.weapons}

    assert weapon_names == {"Lasgun", "Combat Knife"}


def test_unit_specific_weapon_overrides_generic_lookup():
    armies = units.load_armies()
    heavy = next(u for u in armies["Imperium"] if u.name == "Heavy Squad A")

    assert [w.name for w in heavy.weapons] == ["Plasma Cannon"]
    plasma = heavy.weapons[0]
    assert plasma.attacks == Dice(1, 3, 0)
    assert plasma.strength == 8
    assert plasma.ap == -3
    assert plasma.devastating_wounds is True
    assert plasma.models_equipped == 3


def test_weapon_boolean_flags_are_parsed_correctly():
    armies = units.load_armies()
    grunt = next(u for u in armies["Imperium"] if u.name == "Grunt Squad A")
    knife = next(w for w in grunt.weapons if w.name == "Combat Knife")
    lasgun = next(w for w in grunt.weapons if w.name == "Lasgun")

    assert knife.sustained_hits is True
    assert knife.lethal_hits is True
    assert knife.devastating_wounds is False
    assert lasgun.sustained_hits is False
    assert lasgun.devastating_wounds is False


def test_models_equipped_comes_from_the_loadout_count():
    armies = units.load_armies()
    grunt = next(u for u in armies["Imperium"] if u.name == "Grunt Squad A")
    weapons_by_name = {w.name: w for w in grunt.weapons}

    assert weapons_by_name["Lasgun"].models_equipped == 5
    assert weapons_by_name["Combat Knife"].models_equipped == 5


def test_load_armies_is_cached_until_cleared():
    first = units.load_armies()
    second = units.load_armies()

    assert first is second


# --- Small parsing helpers (pure functions, no mocking needed) -------------


def test_parse_value_plain_int():
    assert units._parse_value("3") == 3


def test_parse_value_dice_string():
    assert units._parse_value("2d6+1") == Dice(2, 6, 1)


def test_parse_value_dice_string_without_plus():
    assert units._parse_value("1d3") == Dice(1, 3, 0)


def test_parse_optional_int_blank_is_none():
    assert units._parse_optional_int(float("nan")) is None
    assert units._parse_optional_int("") is None


def test_parse_optional_int_present_value():
    assert units._parse_optional_int("4") == 4


def test_parse_bool_is_case_insensitive():
    assert units._parse_bool("TRUE") is True
    assert units._parse_bool("false") is False
    assert units._parse_bool("") is False


def test_parse_int_or_default_falls_back_when_blank():
    assert units._parse_int_or_default(float("nan")) == 0
    assert units._parse_int_or_default("", default=2) == 2
    assert units._parse_int_or_default("5") == 5
