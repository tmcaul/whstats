import numpy as np
import dataclasses


@dataclasses.dataclass(frozen=True)
class Dice:
    n: int
    sides: int
    plus: int = 0

    def roll_trials(self, trials: int) -> np.ndarray:
        rolls = np.random.randint(low=1, high=self.sides + 1, size=(trials, self.n))
        return rolls.sum(axis=1) + self.plus

    def roll_matrix(self, shape: tuple) -> np.ndarray:
        rolls = np.random.randint(low=1, high=self.sides + 1, size=(*shape, self.n))
        return rolls.sum(axis=-1) + self.plus

    def __post_init__(self):
        assert self.sides > 0, f"invalid number of sides: {self.sides}"


@dataclasses.dataclass(frozen=True)
class Weapon:
    name: str
    attacks: int | Dice
    skill: int
    strength: int
    ap: int
    damage: int | Dice
    torrent: bool = False
    wound_bonus: int = 0
    hit_bonus: int = 0
    sustained_hits: bool = False
    lethal_hits: bool = False
    reroll_hit_ones: bool = False
    reroll_all_hits: bool = False
    reroll_wound_ones: bool = False
    reroll_all_wounds: bool = False
    models_equipped: int | None = None
    phase: str = "ranged"

    def __post_init__(self):
        assert self.ap <= 0, "expect ap <= 0"
        assert -1 <= self.wound_bonus <= 1, "expect wound bonus -1 -> 1"
        assert -1 <= self.hit_bonus <= 1, "expect hit bonus -1 -> 1"
        assert self.phase in ("melee", "ranged"), "expect phase to be 'melee' or 'ranged'"


@dataclasses.dataclass(frozen=True)
class Unit:
    name: str
    toughness: int
    saving_throw: int
    wounds: int
    models: int
    weapons: list[Weapon] = dataclasses.field(default_factory=list)
    invuln: int | None = None
    fnp: int | None = None
    damage_modifier: int = 0


def _to_wound(target: Unit, weapon: Weapon) -> int:
    if weapon.strength >= 2 * target.toughness:
        return 2
    if weapon.strength > target.toughness:
        return 3
    if weapon.strength == target.toughness:
        return 4
    if weapon.strength * 2 <= target.toughness:
        return 6
    return 5


def _to_save(target: Unit, weapon: Weapon) -> int:
    baseline = target.saving_throw - weapon.ap
    if target.invuln is None:
        return baseline
    return min(baseline, target.invuln)


def simulate(target: Unit, weapon: Weapon, attacker: Unit, trials: int = 1000) -> np.ndarray:
    equipped_count = (
        weapon.models_equipped if weapon.models_equipped is not None else attacker.models
    )

    if isinstance(weapon.attacks, int):
        attacks_per_trial = np.full(trials, weapon.attacks * equipped_count)
    else:
        scaled_dice = Dice(
            weapon.attacks.n * equipped_count,
            weapon.attacks.sides,
            weapon.attacks.plus * equipped_count,
        )
        attacks_per_trial = scaled_dice.roll_trials(trials)

    max_attacks = attacks_per_trial.max()

    if max_attacks == 0:
        return np.zeros((trials, 1), dtype=int)

    attack_mask = np.arange(max_attacks) < attacks_per_trial[:, None]

    if weapon.torrent:
        hits_mask = attack_mask
        critical_hits = np.zeros_like(attack_mask)
        max_attacks_sim = max_attacks
    else:
        unmodified_hit_rolls = np.random.randint(low=1, high=7, size=(trials, max_attacks))

        if weapon.reroll_hit_ones:
            unmodified_hit_rerolls = np.random.randint(low=1, high=7, size=(trials, max_attacks))
            rerolled = unmodified_hit_rolls == 1
            unmodified_hit_rolls[rerolled] = unmodified_hit_rerolls[rerolled]

        elif weapon.reroll_all_hits:
            unmodified_hit_rerolls = np.random.randint(low=1, high=7, size=(trials, max_attacks))
            unmodified_hit_rolls = np.maximum(unmodified_hit_rolls, unmodified_hit_rerolls)

        hit_rolls = unmodified_hit_rolls + weapon.hit_bonus
        hits_mask = attack_mask & (hit_rolls >= weapon.skill)
        critical_hits = attack_mask & (unmodified_hit_rolls == 6)

    if weapon.sustained_hits:
        hits_mask = np.concatenate([hits_mask, critical_hits], axis=1)
        critical_hits = np.concatenate([critical_hits, np.zeros_like(critical_hits)], axis=1)
        max_attacks_sim = max_attacks * 2
    else:
        max_attacks_sim = max_attacks

    wound_thresh = _to_wound(target, weapon)
    unmodified_wound_rolls = np.random.randint(low=1, high=7, size=(trials, max_attacks_sim))

    if weapon.reroll_wound_ones:
        unmodified_wound_rerolls = np.random.randint(low=1, high=7, size=(trials, max_attacks_sim))
        rerolled = unmodified_wound_rolls == 1
        unmodified_wound_rolls[rerolled] = unmodified_wound_rerolls[rerolled]

    elif weapon.reroll_all_wounds:
        unmodified_wound_rerolls = np.random.randint(low=1, high=7, size=(trials, max_attacks_sim))
        unmodified_wound_rolls = np.maximum(unmodified_wound_rolls, unmodified_wound_rerolls)

    wound_rolls = unmodified_wound_rolls + weapon.wound_bonus
    wounds_mask = hits_mask & (wound_rolls >= wound_thresh)

    if weapon.lethal_hits:
        wounds_mask[critical_hits] = True

    save_thresh = _to_save(target, weapon)
    save_rolls = np.random.randint(low=1, high=7, size=(trials, max_attacks_sim))
    unsaved_mask = wounds_mask & (save_rolls < save_thresh)

    if isinstance(weapon.damage, int):
        damage_matrix = np.where(unsaved_mask, weapon.damage, 0)
    else:
        damage_rolls = weapon.damage.roll_matrix((trials, max_attacks_sim))
        damage_matrix = np.where(unsaved_mask, damage_rolls, 0)

    if target.fnp is not None:
        p_fail_fnp = (target.fnp - 1) / 6.0
        damage_matrix = np.random.binomial(damage_matrix, p_fail_fnp)

    if target.damage_modifier != 0:
        damage_matrix = np.where(
            damage_matrix > 0, np.maximum(1, damage_matrix + target.damage_modifier), 0
        )

    return damage_matrix.astype(int)


def calculate_remaining_models(unit: Unit, damage_matrix: np.ndarray) -> np.ndarray:
    trials, max_attacks = damage_matrix.shape
    current_wounds = np.full(trials, unit.wounds)
    dead_models = np.zeros(trials, dtype=int)

    for i in range(max_attacks):
        dmg = damage_matrix[:, i]
        current_wounds -= dmg
        killed_this_step = current_wounds <= 0
        dead_models = np.minimum(dead_models + killed_this_step.astype(int), unit.models)
        current_wounds = np.where(killed_this_step, unit.wounds, current_wounds)

    return unit.models - dead_models
