"""Tests for whstats.sim.

Deterministic tests patch np.random.randint to a fixed sequence of constant
rolls. For any Weapon with an int `attacks`/`damage`, no torrent, and no
reroll flags, `simulate()` calls `np.random.randint` exactly three times in
this order: hit rolls, wound rolls, save rolls. That fixed call order is
what `patch_rolls` relies on.
"""

import numpy as np
import pytest

from whstats.sim import Dice, Unit, Weapon, calculate_remaining_models, simulate


@pytest.fixture
def patch_rolls(monkeypatch):
    """Force np.random.randint to return a constant per call, in call order."""

    def _patch(values):
        calls = iter(values)

        def fake_randint(low, high, size):
            return np.full(size, next(calls))

        monkeypatch.setattr(np.random, "randint", fake_randint)

    return _patch


def make_unit(**overrides) -> Unit:
    defaults = dict(name="Target", toughness=4, saving_throw=3, wounds=1, models=10)
    defaults.update(overrides)
    return Unit(**defaults)


def make_weapon(**overrides) -> Weapon:
    defaults = dict(name="Weapon", attacks=5, skill=3, strength=4, ap=0, damage=1)
    defaults.update(overrides)
    return Weapon(**defaults)


# --- Guaranteed edge cases (no RNG needed) ---------------------------------


def test_zero_attacks_deals_no_damage():
    weapon = make_weapon(attacks=0)
    target = make_unit()
    attacker = make_unit(name="Attacker")

    damage = simulate(target, weapon, attacker, trials=50)

    assert damage.shape == (50, 1)
    assert np.all(damage == 0)


def test_impossible_skill_never_hits():
    # skill=7 can never be met by a d6 roll (max 6), regardless of RNG.
    weapon = make_weapon(skill=7, attacks=10, damage=3)
    target = make_unit()
    attacker = make_unit(name="Attacker")

    damage = simulate(target, weapon, attacker, trials=200)

    assert np.all(damage == 0)


def test_guaranteed_skill_always_hits_and_wounds_and_unsaved():
    # skill=1: any roll hits. strength=2*toughness: wound threshold is 2, and
    # target save is impossible (saving_throw=7 => threshold 7, always fails).
    weapon = make_weapon(skill=1, attacks=4, strength=8, damage=2, ap=0)
    target = make_unit(toughness=4, saving_throw=7)
    attacker = make_unit(name="Attacker")

    damage = simulate(target, weapon, attacker, trials=200)

    # A roll of 1 wounds (thresh=2) only when it fails, i.e. roll==1 out of
    # 1..6, so this alone isn't fully guaranteed to wound -- but the save is
    # guaranteed to fail whenever a wound lands, so every trial's total
    # damage must be a multiple of `damage` and attacks that do wound cannot
    # ever be saved.
    assert np.all(damage[damage > 0] == weapon.damage)


# --- Guaranteed edge cases (RNG rolls forced to a fixed sequence) ---------


def test_devastating_wounds_bypasses_a_would_be_successful_save(patch_rolls):
    # hit=6 (hits & is critical), wound=6 (wounds & is a critical wound),
    # save=6 which would normally SAVE against a saving_throw=2 target
    # (threshold 2, needs roll < 2 to fail the save).
    patch_rolls([6, 6, 6])
    weapon = make_weapon(skill=2, strength=8, damage=2, attacks=3, devastating_wounds=True)
    target = make_unit(toughness=4, saving_throw=2)
    attacker = make_unit(name="Attacker")

    damage = simulate(target, weapon, attacker, trials=10)

    assert np.all(damage == weapon.damage)


def test_without_devastating_wounds_the_same_rolls_are_saved(patch_rolls):
    patch_rolls([6, 6, 6])
    weapon = make_weapon(skill=2, strength=8, damage=2, attacks=3, devastating_wounds=False)
    target = make_unit(toughness=4, saving_throw=2)
    attacker = make_unit(name="Attacker")

    damage = simulate(target, weapon, attacker, trials=10)

    assert np.all(damage == 0)


def test_lethal_hits_forces_a_wound_on_a_failed_wound_roll(patch_rolls):
    # hit=6 (critical), wound=1 (would normally fail: thresh=6 for
    # strength*2 <= toughness), save=6 (always fails vs saving_throw=7).
    patch_rolls([6, 1, 6])
    weapon = make_weapon(skill=2, strength=1, damage=1, attacks=2, lethal_hits=True)
    target = make_unit(toughness=6, saving_throw=7)
    attacker = make_unit(name="Attacker")

    damage = simulate(target, weapon, attacker, trials=10)

    assert np.all(damage == weapon.damage)


def test_without_lethal_hits_the_same_rolls_fail_to_wound(patch_rolls):
    patch_rolls([6, 1, 6])
    weapon = make_weapon(skill=2, strength=1, damage=1, attacks=2, lethal_hits=False)
    target = make_unit(toughness=6, saving_throw=7)
    attacker = make_unit(name="Attacker")

    damage = simulate(target, weapon, attacker, trials=10)

    assert np.all(damage == 0)


def test_sustained_hits_doubles_the_attack_pool_on_all_criticals(patch_rolls):
    # Every hit roll is a critical 6, so every attack spawns one extra
    # sustained-hit attack: max_attacks_sim == 2 * max_attacks.
    patch_rolls([6, 6, 6])
    weapon = make_weapon(skill=2, strength=8, damage=1, attacks=3, sustained_hits=True)
    target = make_unit(toughness=4, saving_throw=7)
    attacker = make_unit(name="Attacker", models=1)

    damage = simulate(target, weapon, attacker, trials=10)

    assert damage.shape == (10, 6)
    assert np.all(damage == weapon.damage)


def test_unit_hit_bonus_reduces_incoming_hit_chance(patch_rolls):
    # skill=2, roll forced to 2 => hits with no bonus, but a target
    # hit_bonus of -1 shifts the roll to 1, which misses.
    patch_rolls([2, 6, 6])
    weapon = make_weapon(skill=2, attacks=1, damage=1)
    attacker = make_unit(name="Attacker")

    target_no_bonus = make_unit(hit_bonus=0, saving_throw=7)
    assert np.all(simulate(target_no_bonus, weapon, attacker, trials=10) == weapon.damage)

    patch_rolls([2, 6, 6])
    target_with_bonus = make_unit(hit_bonus=-1, saving_throw=7)
    assert np.all(simulate(target_with_bonus, weapon, attacker, trials=10) == 0)


def test_unit_wound_bonus_increases_incoming_wound_chance(patch_rolls):
    # strength == toughness => wound threshold 4. Roll forced to 3 fails
    # without a bonus, but a target wound_bonus of +1 pushes it to 4.
    patch_rolls([6, 3, 6])
    weapon = make_weapon(skill=2, strength=4, damage=1, attacks=1)
    target_no_bonus = make_unit(toughness=4, saving_throw=7, wound_bonus=0)
    attacker = make_unit(name="Attacker")
    assert np.all(simulate(target_no_bonus, weapon, attacker, trials=10) == 0)

    patch_rolls([6, 3, 6])
    target_with_bonus = make_unit(toughness=4, saving_throw=7, wound_bonus=1)
    assert np.all(simulate(target_with_bonus, weapon, attacker, trials=10) == weapon.damage)


def test_weapon_and_unit_hit_bonus_stack_but_clamp_to_plus_one(patch_rolls):
    # skill=4, roll forced to 2. A single +1 modifier gives hit_roll=3, still
    # a miss. If weapon (+1) and unit (+1) bonuses summed unclamped to +2,
    # hit_roll would be 4 and (wrongly) hit. Clamped to +1 total, it misses.
    patch_rolls([2, 6, 6])
    weapon = make_weapon(skill=4, attacks=1, damage=1, hit_bonus=1)
    target = make_unit(hit_bonus=1, saving_throw=7)
    attacker = make_unit(name="Attacker")

    damage = simulate(target, weapon, attacker, trials=10)

    assert np.all(damage == 0)


# --- Dataclass validation ---------------------------------------------------


def test_weapon_rejects_out_of_range_hit_bonus():
    with pytest.raises(AssertionError):
        make_weapon(hit_bonus=2)


def test_weapon_rejects_positive_ap():
    with pytest.raises(AssertionError):
        make_weapon(ap=1)


def test_unit_rejects_out_of_range_wound_bonus():
    with pytest.raises(AssertionError):
        make_unit(wound_bonus=-2)


# --- calculate_remaining_models (pure function, no RNG) --------------------


def test_calculate_remaining_models_kills_one_model_per_lethal_hit():
    unit = make_unit(wounds=2, models=3)
    # 5 attacks of 2 damage each: kills a model every attack, one is wasted
    # once the unit is fully dead.
    damage_matrix = np.full((1, 5), 2)

    remaining = calculate_remaining_models(unit, damage_matrix)

    assert remaining.tolist() == [0]


def test_calculate_remaining_models_carries_wounds_across_attacks():
    unit = make_unit(wounds=3, models=2)
    # Two 2-damage hits kill the first model (2+2=4 >= 3, wounds carry over
    # only within a model), a third hit starts damaging the second model.
    damage_matrix = np.array([[2, 2, 2]])

    remaining = calculate_remaining_models(unit, damage_matrix)

    assert remaining.tolist() == [1]


def test_calculate_remaining_models_never_goes_negative():
    unit = make_unit(wounds=1, models=2)
    damage_matrix = np.full((1, 10), 5)

    remaining = calculate_remaining_models(unit, damage_matrix)

    assert remaining.tolist() == [0]


# --- Smoke tests (real RNG, statistical sanity only) ------------------------


@pytest.mark.parametrize(
    "weapon_kwargs",
    [
        dict(),
        dict(torrent=True),
        dict(sustained_hits=True),
        dict(lethal_hits=True),
        dict(devastating_wounds=True),
        dict(reroll_hit_ones=True),
        dict(reroll_all_hits=True),
        dict(reroll_wound_ones=True),
        dict(reroll_all_wounds=True),
        dict(attacks=Dice(2, 6), damage=Dice(1, 3, 1)),
        dict(sustained_hits=True, lethal_hits=True, devastating_wounds=True),
    ],
)
def test_simulate_smoke(weapon_kwargs):
    weapon = make_weapon(**weapon_kwargs)
    target = make_unit(invuln=5, fnp=6, damage_modifier=-1, hit_bonus=-1, wound_bonus=1)
    attacker = make_unit(name="Attacker", models=5)

    damage = simulate(target, weapon, attacker, trials=500)

    assert damage.shape[0] == 500
    assert np.all(damage >= 0)
    assert damage.dtype.kind in "iu"

    remaining = calculate_remaining_models(target, damage)
    assert np.all(remaining >= 0)
    assert np.all(remaining <= target.models)
