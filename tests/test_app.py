"""Tests for whstats.app.

Two layers:
  - Direct unit tests of the plain helper functions (no Streamlit runtime).
  - Headless UI smoke tests via Streamlit's `AppTest`, which executes a page
    function exactly as the real app would (widgets, session state,
    fragments) without a browser. Each page is wrapped in a tiny inline
    script for `AppTest.from_string`, since `AppTest.from_function` requires
    a fully self-contained function body and our page functions rely on
    whstats.app's module-level imports.
"""

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from streamlit.testing.v1 import AppTest

from whstats import units
from whstats.app import (
    _build_heatmap_figure,
    _fmt_dice_or_int,
    _weapons_to_editor_df,
    apply_modifiers,
    create_unit_heatmap,
)
from whstats.sim import Dice, Unit, Weapon


UNITS_CSV = """unit,toughness,saving_throw,wounds,models,invuln,fnp,damage_modifier,hit_bonus,wound_bonus
Grunt Squad,4,3,1,5,,,0,-1,0
Ork Boyz,5,6,1,10,,,0,0,0
"""

WEAPONS_CSV = """unit,weapon,attacks,skill,strength,ap,damage,phase,torrent,wound_bonus,hit_bonus,sustained_hits,lethal_hits,devastating_wounds,reroll_hit_ones,reroll_all_hits,reroll_wound_ones,reroll_all_wounds
,Lasgun,1,3,3,0,1,ranged,false,0,0,false,false,false,false,false,false,false
,Choppa,2,3,4,0,1,melee,false,0,0,true,true,false,false,false,false,false
"""

LOADOUTS_CSV = """display_name,army,unit,weapon,count
Grunt Squad A,Imperium,Grunt Squad,Lasgun,5
Ork Boyz A,Orks,Ork Boyz,Choppa,10
"""


@pytest.fixture(autouse=True)
def mock_sheets(monkeypatch):
    """Two tiny armies (Imperium / Orks) served in place of the Google Sheet."""
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


def run_page(page_func_name: str) -> AppTest:
    script = f"from whstats.app import {page_func_name}\n{page_func_name}()\n"
    at = AppTest.from_string(script, default_timeout=30)
    at.run()
    return at


# --- Plain helper functions --------------------------------------------------


def test_fmt_dice_or_int_plain_int():
    assert _fmt_dice_or_int(3) == "3"


def test_fmt_dice_or_int_dice_with_plus():
    assert _fmt_dice_or_int(Dice(2, 6, 1)) == "2d6+1"


def test_fmt_dice_or_int_dice_without_plus():
    assert _fmt_dice_or_int(Dice(1, 3)) == "1d3"


def test_weapons_to_editor_df_includes_all_modifier_columns():
    weapon = Weapon(
        name="Choppa",
        attacks=2,
        skill=3,
        strength=4,
        ap=0,
        damage=1,
        sustained_hits=True,
        lethal_hits=True,
        devastating_wounds=True,
        models_equipped=7,
    )
    unit = Unit(name="Orks", toughness=5, saving_throw=6, wounds=1, models=10, weapons=[weapon])

    df = _weapons_to_editor_df(unit)

    row = df.iloc[0]
    assert row["Weapon"] == "Choppa"
    assert row["Count"] == 7
    assert bool(row["Sustained Hits"]) is True
    assert bool(row["Lethal Hits"]) is True
    assert bool(row["Devastating Wounds"]) is True
    assert bool(row["Torrent"]) is False


def test_weapons_to_editor_df_defaults_count_to_unit_models_when_unequipped():
    weapon = Weapon(name="Lasgun", attacks=1, skill=3, strength=3, ap=0, damage=1)
    unit = Unit(name="Grunts", toughness=4, saving_throw=3, wounds=1, models=5, weapons=[weapon])

    df = _weapons_to_editor_df(unit)

    assert df.iloc[0]["Count"] == 5


def test_apply_modifiers_clamps_hit_and_wound_bonus_to_plus_minus_one():
    weapon = Weapon(name="W", attacks=1, skill=3, strength=3, ap=0, damage=1, hit_bonus=1)
    unit = Unit(name="U", toughness=4, saving_throw=3, wounds=1, models=10, weapons=[weapon])

    modified = apply_modifiers([unit], hit_mod=1, wound_mod=-1, ap_mod=0)

    assert modified[0].weapons[0].hit_bonus == 1  # 1 + 1 clamped to 1
    assert modified[0].weapons[0].wound_bonus == -1  # 0 - 1 clamped to -1


def test_apply_modifiers_never_pushes_ap_positive():
    weapon = Weapon(name="W", attacks=1, skill=3, strength=3, ap=-1, damage=1)
    unit = Unit(name="U", toughness=4, saving_throw=3, wounds=1, models=10, weapons=[weapon])

    modified = apply_modifiers([unit], hit_mod=0, wound_mod=0, ap_mod=5)

    assert modified[0].weapons[0].ap == 0


def test_apply_modifiers_scales_models_and_weapon_counts_together():
    weapon = Weapon(name="W", attacks=1, skill=3, strength=3, ap=0, damage=1, models_equipped=10)
    unit = Unit(name="U", toughness=4, saving_throw=3, wounds=1, models=10, weapons=[weapon])

    modified = apply_modifiers([unit], hit_mod=0, wound_mod=0, ap_mod=0, model_pct=0.5)

    assert modified[0].models == 5
    assert modified[0].weapons[0].models_equipped == 5


def test_apply_modifiers_preserves_unit_level_hit_and_wound_bonus():
    unit = Unit(
        name="U", toughness=4, saving_throw=3, wounds=1, models=10, hit_bonus=-1, wound_bonus=1
    )

    modified = apply_modifiers([unit], hit_mod=0, wound_mod=0, ap_mod=0)

    assert modified[0].hit_bonus == -1
    assert modified[0].wound_bonus == 1


def test_build_heatmap_figure_damage_metric():
    df = pd.DataFrame({"TargetA": [1, 2]}, index=["UnitA", "UnitB"])
    fig = _build_heatmap_figure(df, df.astype(str), metric="damage", row_axis_title="Unit")

    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Total Wounds Dealt"


def test_build_heatmap_figure_pct_kills_metric():
    df = pd.DataFrame({"TargetA": [50, 75]}, index=["UnitA", "UnitB"])
    fig = _build_heatmap_figure(df, df.astype(str), metric="pct_kills", row_axis_title="Unit")

    assert fig.layout.title.text == "% Models Killed"
    assert fig.data[0].zmax == 100


def test_create_unit_heatmap_uses_injected_simulate_fn():
    weapon = Weapon(name="W", attacks=1, skill=3, strength=3, ap=0, damage=1, phase="ranged")
    attacker = Unit(name="Att", toughness=4, saving_throw=3, wounds=1, models=1, weapons=[weapon])
    target = Unit(name="Tgt", toughness=4, saving_throw=3, wounds=5, models=3)

    def fake_simulate(target, weapon, attacker, trials):
        return np.full((trials, 2), 4)

    fig = create_unit_heatmap(
        [attacker], [target], phase="ranged", simulate_fn=fake_simulate, metric="damage", trials=10
    )

    assert fig.data[0].z[0][0] == 8  # 2 attacks * 4 damage each, deterministic


def test_create_unit_heatmap_handles_no_weapons_in_phase():
    attacker = Unit(name="Att", toughness=4, saving_throw=3, wounds=1, models=1, weapons=[])
    target = Unit(name="Tgt", toughness=4, saving_throw=3, wounds=5, models=3)

    fig = create_unit_heatmap(
        [attacker], [target], phase="melee", simulate_fn=None, metric="damage", trials=10
    )

    assert fig.data[0].z[0][0] == 0


# --- Drilldown page (headless UI smoke tests) -------------------------------


def test_drilldown_page_renders_without_error():
    at = run_page("render_unit_vs_unit_page")

    assert not at.exception
    assert at.title[0].value == "Drilldown"


def test_drilldown_page_defender_stats_table_reflects_unit_defensive_bonus():
    at = run_page("render_unit_vs_unit_page")

    # Defender defaults to "Grunt Squad A" -> hit_bonus=-1, wound_bonus=0.
    def_row = at.dataframe[0].value.iloc[0]
    assert def_row["Hit Bonus"] == -1
    assert def_row["Wound Bonus"] == 0
    assert def_row["Models"] == 5
    assert def_row["Toughness"] == 4


def test_drilldown_page_weapon_editor_has_all_modifier_checkboxes():
    at = run_page("render_unit_vs_unit_page")

    df = at.dataframe[1].value
    for column in [
        "Torrent",
        "Sustained Hits",
        "Lethal Hits",
        "Devastating Wounds",
        "Reroll Hit Ones",
        "Reroll All Hits",
        "Reroll Wound Ones",
        "Reroll All Wounds",
    ]:
        assert column in df.columns


def test_drilldown_page_weapon_editor_reflects_weapon_flags():
    at = run_page("render_unit_vs_unit_page")
    at.selectbox(key="atk_army_uvu").set_value("Orks")
    at.run()

    assert not at.exception
    row = at.dataframe[1].value.iloc[0]
    assert row["Weapon"] == "Choppa"
    assert bool(row["Sustained Hits"]) is True
    assert bool(row["Lethal Hits"]) is True


def test_drilldown_page_switching_attacker_unit_does_not_error():
    at = run_page("render_unit_vs_unit_page")
    at.selectbox(key="atk_army_uvu").set_value("Orks")
    at.run()
    at.selectbox(key="atk_unit_uvu").set_value("Ork Boyz A")
    at.run()

    assert not at.exception


def test_drilldown_page_changing_trial_count_does_not_error():
    at = run_page("render_unit_vs_unit_page")
    at.number_input(key="trials_uvu").set_value(250)
    at.run()

    assert not at.exception


# --- Army summary page (headless UI smoke tests) ----------------------------


def test_army_page_renders_without_error():
    at = run_page("render_army_unit_summary_page")

    assert not at.exception
    assert at.title[0].value == "Army"


def test_army_page_flip_button_swaps_attacker_and_defender():
    at = run_page("render_army_unit_summary_page")
    at.sidebar.selectbox(key="def_army_summary").set_value("Orks")
    at.run()

    before_attacker = at.sidebar.selectbox(key="atk_army_summary").value
    before_defender = at.sidebar.selectbox(key="def_army_summary").value

    at.button[0].click()
    at.run()

    assert not at.exception
    assert at.sidebar.selectbox(key="atk_army_summary").value == before_defender
    assert at.sidebar.selectbox(key="def_army_summary").value == before_attacker


def test_army_page_hit_wound_ap_sliders_do_not_error():
    at = run_page("render_army_unit_summary_page")
    at.sidebar.slider(key="summary_hit").set_value(1)
    at.sidebar.slider(key="summary_wound").set_value(-1)
    at.sidebar.slider(key="summary_ap").set_value(-2)
    at.run()

    assert not at.exception
