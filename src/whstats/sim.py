import numpy as np
import dataclasses
import seaborn as sns
import matplotlib.pyplot as plt


@dataclasses.dataclass(frozen=True)
class Unit:
    toughness: int
    saving_throw: int
    wounds: int
    models: int
    invuln: int | None = None
    fnp: int | None = None  # Feel No Pain threshold (e.g., 5 for a 5+ FNP)


@dataclasses.dataclass(frozen=True)
class Dice:
    n: int
    sides: int
    plus: int = 0

    def roll_trials(self, trials: int) -> np.ndarray:
        """Rolls n dice per trial and sums them. Shape: (trials,)"""
        rolls = np.random.randint(low=1, high=self.sides + 1, size=(trials, self.n))
        return rolls.sum(axis=1) + self.plus

    def roll_matrix(self, shape: tuple) -> np.ndarray:
        """Rolls dice for a 2D matrix of successful attacks."""
        rolls = np.random.randint(low=1, high=self.sides + 1, size=(*shape, self.n))
        return rolls.sum(axis=-1) + self.plus

    def __post_init__(self):
        assert self.sides > 0, f"invalid number of sides: {self.sides}"


@dataclasses.dataclass(frozen=True)
class Weapon:
    attacks: int | Dice
    skill: int
    strength: int
    ap: int
    damage: int | Dice
    torrent: bool = False
    wound_bonus: int = 0
    hit_bonus: int = 0
    sustained_hits: bool = False  # Adds 1 extra hit on an unmodified 6
    lethal_hits: bool = False
    reroll_hit_ones: bool = False
    reroll_all_hits: bool = False
    reroll_wound_ones: bool = False
    reroll_all_wounds: bool = False

    def __post_init__(self):
        assert self.ap <= 0, "expect ap <= 0"
        assert -1 <= self.wound_bonus <= 1, "expect wound bonus -1 -> 1"
        assert -1 <= self.hit_bonus <= 1, "expect hit bonus -1 -> 1"


def _to_wound(unit: Unit, weapon: Weapon) -> int:
    if weapon.strength >= 2 * unit.toughness:
        return 2
    if weapon.strength > unit.toughness:
        return 3
    if weapon.strength == unit.toughness:
        return 4
    if weapon.strength * 2 <= unit.toughness:
        return 6
    return 5


def _to_save(unit: Unit, weapon: Weapon) -> int:
    baseline = unit.saving_throw - weapon.ap
    if unit.invuln is None:
        return baseline
    return min(baseline, unit.invuln)


def simulate(unit: Unit, weapon: Weapon, trials: int = 1000) -> np.ndarray:

    # Determine attacks per trial
    if isinstance(weapon.attacks, int):
        attacks_per_trial = np.full(trials, weapon.attacks)
    else:
        attacks_per_trial = weapon.attacks.roll_trials(trials)

    max_attacks = attacks_per_trial.max()

    if max_attacks == 0:
        return np.zeros((trials, 1), dtype=int)

    attack_mask = np.arange(max_attacks) < attacks_per_trial[:, None]

    if weapon.torrent:
        hits_mask = attack_mask  # Auto-hits
        critical_hits = np.zeros_like(attack_mask)  # Torrents don't roll, so no crits
        max_attacks_sim = max_attacks
    else:
        unmodified_hit_rolls = np.random.randint(
            low=1, high=7, size=(trials, max_attacks)
        )

        if weapon.reroll_hit_ones:
            assert not weapon.reroll_all_hits
            unmodified_hit_rerolls = np.random.randint(
                low=1, high=7, size=(trials, max_attacks)
            )
            rerolled = unmodified_hit_rolls == 1
            unmodified_hit_rolls[rerolled] = unmodified_hit_rerolls[rerolled]

        elif weapon.reroll_all_hits:
            unmodified_hit_rerolls = np.random.randint(
                low=1, high=7, size=(trials, max_attacks)
            )
            unmodified_hit_rolls = np.maximum(
                unmodified_hit_rolls, unmodified_hit_rerolls
            )

        hit_rolls = unmodified_hit_rolls + weapon.hit_bonus
        hits_mask = attack_mask & (hit_rolls >= weapon.skill)
        critical_hits = attack_mask & (unmodified_hit_rolls == 6)

    if weapon.sustained_hits:
        hits_mask = np.concatenate([hits_mask, critical_hits], axis=1)
        critical_hits = np.concatenate(
            [critical_hits, np.zeros_like(critical_hits)], axis=1
        )

        max_attacks_sim = max_attacks * 2
    else:
        max_attacks_sim = max_attacks

    wound_thresh = _to_wound(unit, weapon)
    unmodified_wound_rolls = np.random.randint(
        low=1, high=7, size=(trials, max_attacks_sim)
    )

    if weapon.reroll_wound_ones:
        assert not weapon.reroll_all_wounds
        unmodified_wound_rerolls = np.random.randint(
            low=1, high=7, size=(trials, max_attacks_sim)
        )
        rerolled = unmodified_wound_rolls == 1
        unmodified_wound_rolls[rerolled] = unmodified_wound_rerolls[rerolled]

    elif weapon.reroll_all_wounds:
        unmodified_wound_rerolls = np.random.randint(
            low=1, high=7, size=(trials, max_attacks_sim)
        )
        unmodified_wound_rolls = np.maximum(
            unmodified_wound_rolls, unmodified_wound_rerolls
        )

    wound_rolls = unmodified_wound_rolls + weapon.wound_bonus
    wounds_mask = hits_mask & (wound_rolls >= wound_thresh)

    if weapon.lethal_hits:
        wounds_mask[critical_hits] = True

    save_thresh = _to_save(unit, weapon)
    save_rolls = np.random.randint(low=1, high=7, size=(trials, max_attacks_sim))
    unsaved_mask = wounds_mask & (save_rolls < save_thresh)

    if isinstance(weapon.damage, int):
        damage_matrix = np.where(unsaved_mask, weapon.damage, 0)
    else:
        damage_rolls = weapon.damage.roll_matrix((trials, max_attacks_sim))
        damage_matrix = np.where(unsaved_mask, damage_rolls, 0)

    if unit.fnp is not None:
        p_fail_fnp = (unit.fnp - 1) / 6.0
        damage_matrix = np.random.binomial(damage_matrix, p_fail_fnp)

    return damage_matrix


def calculate_remaining_models(unit: Unit, damage_matrix: np.ndarray) -> np.ndarray:
    """
    Calculates the number of models left in the unit across all trials,
    respecting the 'damage does not spill over' rule.
    """

    trials, max_attacks = damage_matrix.shape
    current_wounds = np.full(trials, unit.wounds)
    dead_models = np.zeros(trials, dtype=int)

    for i in range(max_attacks):
        dmg = damage_matrix[:, i]
        current_wounds -= dmg

        killed_this_step = current_wounds <= 0

        dead_models = np.minimum(
            dead_models + killed_this_step.astype(int), unit.models
        )
        current_wounds = np.where(killed_this_step, unit.wounds, current_wounds)

    return unit.models - dead_models


if __name__ == "__main__":

    my_weapons = {
        "Wraithcannon": Weapon(
            attacks=5,
            skill=4,
            strength=14,
            ap=-4,
            damage=Dice(1, 6, 1),
            hit_bonus=1,
            wound_bonus=1,
        ),
        "D-Scythe": Weapon(
            attacks=Dice(5, 6),
            skill=0,
            strength=7,
            ap=-3,
            damage=1,
            wound_bonus=1,
            torrent=True,
        ),
        "Wraithblades": Weapon(
            attacks=25,
            skill=4,
            strength=5,
            ap=-2,
            damage=2,
            hit_bonus=1,
            wound_bonus=1,
        ),
        "Windriders": Weapon(
            attacks=9, skill=3, strength=6, ap=-1, damage=2, lethal_hits=True
        ),
        "Guardian Shuriken Catapult": Weapon(
            attacks=20,
            skill=3,
            strength=4,
            ap=-1,
            damage=1,
            wound_bonus=1,
        ),
    }

    alf_units = {
        "": Unit(
            toughness=5,
            saving_throw=3,
            wounds=3,
            models=6,
            invuln=5,
        )
    }

    n_weapons = len(weapons)
    fig, axes = plt.subplots(n_weapons, 2, figsize=(8, 2 * n_weapons))

    for i, (name, weapon) in enumerate(weapons.items()):
        damage_matrix = simulate(u, weapon, trials=5000)
        flattened_damage = damage_matrix.sum(axis=1)
        remaining_models = calculate_remaining_models(u, damage_matrix)

        # Plot Damage Distribution (Left Column)
        sns.histplot(
            flattened_damage, ax=axes[i, 0], kde=True, stat="probability", discrete=True
        )
        axes[i, 0].set_xlabel("Total Damage")
        axes[i, 0].set_title(f"{name} - Damage Distribution")

        # Plot Surviving Models (Right Column)
        sns.histplot(
            remaining_models,
            ax=axes[i, 1],
            stat="probability",
            discrete=True,
            color="coral",
        )
        axes[i, 1].set_xlabel("Models Remaining")
        axes[i, 1].set_title(f"{name} - Surviving Models")
        axes[i, 1].set_xticks(range(u.models + 1))

    plt.tight_layout()
    plt.show()
