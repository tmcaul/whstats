import numpy as np
import dataclasses
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


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
    name: str
    attacks: int | Dice  # Attacks PER MODEL
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
    models_equipped: int | None = None  # If None, assumes all models in the unit are equipped

    def __post_init__(self):
        assert self.ap <= 0, "expect ap <= 0"
        assert -1 <= self.wound_bonus <= 1, "expect wound bonus -1 -> 1"
        assert -1 <= self.hit_bonus <= 1, "expect hit bonus -1 -> 1"


@dataclasses.dataclass(frozen=True)
class Unit:
    name: str
    toughness: int
    saving_throw: int
    wounds: int
    models: int
    weapons: list[Weapon] = dataclasses.field(default_factory=list)
    invuln: int | None = None
    fnp: int | None = None  # Feel No Pain threshold (e.g., 5 for a 5+ FNP)


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

    # Determine attacks per trial by scaling per-model attacks by equipped_count
    if isinstance(weapon.attacks, int):
        attacks_per_trial = np.full(trials, weapon.attacks * equipped_count)
    else:
        # Scale the dice pool by the number of models equipped
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
        hits_mask = attack_mask  # Auto-hits
        critical_hits = np.zeros_like(attack_mask)  # Torrents don't roll, so no crits
        max_attacks_sim = max_attacks
    else:
        unmodified_hit_rolls = np.random.randint(low=1, high=7, size=(trials, max_attacks))

        if weapon.reroll_hit_ones:
            assert not weapon.reroll_all_hits
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
        assert not weapon.reroll_all_wounds
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

        dead_models = np.minimum(dead_models + killed_this_step.astype(int), unit.models)
        current_wounds = np.where(killed_this_step, unit.wounds, current_wounds)

    return unit.models - dead_models


def plot_weapon_vs_unit_heatmap(
    attacker_units: list[Unit], target_units: list[Unit], simulate_fn, trials=2000
):
    """Generates a heatmap comparing damage percentiles of all weapons across all target units."""
    median_data = []
    annot_data = []

    for attacker in attacker_units:
        for weapon in attacker.weapons:
            w_name = f"{attacker.name}: {weapon.name}"
            med_row = {"Weapon": w_name}
            ann_row = {"Weapon": w_name}

            for target in target_units:
                damage_matrix = simulate_fn(
                    target=target, weapon=weapon, attacker=attacker, trials=trials
                )

                # Calculate totals for each trial
                totals = damage_matrix.sum(axis=1)

                # Calculate percentiles
                p25, p50, p75 = np.percentile(totals, [25, 50, 75])

                # Store median for the color scale
                med_row[target.name] = p50

                # Store the formatted string for the text annotation
                ann_row[target.name] = f"{p25:.0f} - {p50:.0f} - {p75:.0f}"

            median_data.append(med_row)
            annot_data.append(ann_row)

    df_median = pd.DataFrame(median_data).set_index("Weapon")
    df_annot = pd.DataFrame(annot_data).set_index("Weapon")

    plt.figure(figsize=(14, max(8, len(df_median) * 0.3)))

    sns.heatmap(
        df_median,
        annot=df_annot,
        fmt="",
        cmap="YlOrRd",
        linewidths=0.5,
        cbar_kws={"label": "Median Damage"},
        vmin=0,
        vmax=25,
    )

    plt.title(
        "Weapon Damage",
        fontsize=12,
        fontweight="bold",
    )

    plt.ylabel("Weapon")
    plt.xlabel("Target Unit")
    plt.tight_layout()
    plt.show()


def plot_weapon_vs_unit_kills_heatmap(
    attacker_units: list[Unit], target_units: list[Unit], simulate_fn, trials=2000
):
    """Generates a heatmap comparing models killed percentiles of all weapons across all target units."""
    median_data = []
    annot_data = []

    for attacker in attacker_units:
        for weapon in attacker.weapons:
            w_name = f"{attacker.name}: {weapon.name}"
            med_row = {"Weapon": w_name}
            ann_row = {"Weapon": w_name}

            for target in target_units:
                damage_matrix = simulate_fn(
                    target=target, weapon=weapon, attacker=attacker, trials=trials
                )

                remaining_models = calculate_remaining_models(target, damage_matrix)
                killed_models = target.models - remaining_models
                p25, p50, p75 = np.percentile(killed_models, [25, 50, 75])

                # 4. Store data
                med_row[target.name] = p50
                ann_row[target.name] = f"{p25:.0f} - {p50:.0f} - {p75:.0f}"

            median_data.append(med_row)
            annot_data.append(ann_row)

    df_median = pd.DataFrame(median_data).set_index("Weapon")
    df_annot = pd.DataFrame(annot_data).set_index("Weapon")

    plt.figure(figsize=(14, max(8, len(df_median) * 0.3)))

    sns.heatmap(
        df_median,
        annot=df_annot,
        fmt="",
        cmap="YlOrRd",
        linewidths=0.5,
        cbar_kws={"label": "Median Models Killed"},
        vmin=0,
    )

    plt.title(
        "Models Killed",
        fontsize=12,
        fontweight="bold",
    )
    plt.ylabel("Weapon")
    plt.xlabel("Target Unit")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    alf_units = [
        Unit(
            name="Lucius the Eternal",
            toughness=5,
            saving_throw=2,
            wounds=6,
            models=1,
            invuln=4,
            weapons=[
                Weapon(name="Blade of the Laer", attacks=6, skill=2, strength=8, ap=-3, damage=3),
                Weapon(name="Lash of Torment", attacks=10, skill=2, strength=4, ap=-1, damage=1),
            ],
        ),
        Unit(
            name="Daemon Prince of Slaanesh with Wings",
            toughness=9,
            saving_throw=2,
            wounds=10,
            models=1,
            invuln=4,
            weapons=[
                Weapon(name="Infernal cannon", attacks=3, skill=2, strength=5, ap=-1, damage=2),
                Weapon(
                    name="Hellforged weapons - strike",
                    attacks=6,
                    skill=2,
                    strength=8,
                    ap=-2,
                    damage=3,
                ),
                Weapon(
                    name="Hellforged weapons - sweep",
                    attacks=14,
                    skill=2,
                    strength=6,
                    ap=0,
                    damage=1,
                ),
            ],
        ),
        Unit(
            name="Lord Exultant",
            toughness=4,
            saving_throw=3,
            wounds=5,
            models=1,
            invuln=4,
            weapons=[
                Weapon(
                    name="Bolt pistol",
                    attacks=1,
                    skill=2,
                    strength=4,
                    ap=0,
                    damage=1,
                    lethal_hits=True,
                ),
                Weapon(
                    name="Phoenix power spear",
                    attacks=5,
                    skill=2,
                    strength=7,
                    ap=-2,
                    damage=2,
                    lethal_hits=True,
                ),
                Weapon(
                    name="Rapture lash",
                    attacks=4,
                    skill=2,
                    strength=4,
                    ap=-1,
                    damage=1,
                    lethal_hits=True,
                ),
                Weapon(
                    name="Close combat weapon",
                    attacks=6,
                    skill=2,
                    strength=4,
                    ap=0,
                    damage=1,
                    lethal_hits=True,
                ),
            ],
        ),
        Unit(
            name="Lord Exultant (Euphoric Crown)",
            toughness=4,
            saving_throw=3,
            wounds=5,
            models=1,
            invuln=4,
            weapons=[
                Weapon(
                    name="Phoenix power spear",
                    attacks=5,
                    skill=2,
                    strength=8,
                    ap=-2,
                    damage=2,
                    lethal_hits=True,
                ),
                Weapon(
                    name="Rapture lash",
                    attacks=4,
                    skill=2,
                    strength=5,
                    ap=-1,
                    damage=1,
                    lethal_hits=True,
                ),
                Weapon(
                    name="Close combat weapon",
                    attacks=6,
                    skill=2,
                    strength=5,
                    ap=0,
                    damage=1,
                    lethal_hits=True,
                ),
            ],
        ),
        Unit(
            name="Lord Kakophonist",
            toughness=5,
            saving_throw=2,
            wounds=6,
            models=1,
            invuln=4,
            weapons=[
                Weapon(name="Screamer pistol", attacks=3, skill=2, strength=5, ap=-1, damage=2),
                Weapon(name="Close combat weapon", attacks=6, skill=2, strength=4, ap=0, damage=1),
            ],
        ),
        Unit(
            name="Infractors",
            toughness=4,
            saving_throw=3,
            wounds=2,
            models=5,
            invuln=None,
            weapons=[
                Weapon(
                    name="Bolt pistol",
                    attacks=1,
                    skill=3,
                    strength=4,
                    ap=0,
                    damage=1,
                    lethal_hits=True,
                ),
                Weapon(
                    name="Plasma pistol - standard",
                    attacks=1,
                    skill=3,
                    strength=7,
                    ap=-2,
                    damage=1,
                    lethal_hits=True,
                ),
                Weapon(
                    name="Plasma pistol - supercharge",
                    attacks=1,
                    skill=3,
                    strength=8,
                    ap=-3,
                    damage=2,
                    lethal_hits=True,
                ),
                Weapon(
                    name="Rapture lash",
                    attacks=6,
                    skill=3,
                    strength=4,
                    ap=-1,
                    damage=1,
                    lethal_hits=True,
                ),
                Weapon(
                    name="Duelling sabre",
                    attacks=4,
                    skill=3,
                    strength=4,
                    ap=-1,
                    damage=1,
                    lethal_hits=True,
                ),
            ],
        ),
        Unit(
            name="Tormentors",
            toughness=4,
            saving_throw=3,
            wounds=2,
            models=5,
            invuln=None,
            weapons=[
                Weapon(name="Boltgun", attacks=2, skill=3, strength=4, ap=0, damage=1),
                Weapon(
                    name="Plasma pistol - standard", attacks=1, skill=3, strength=7, ap=-2, damage=1
                ),
                Weapon(
                    name="Plasma pistol - supercharge",
                    attacks=1,
                    skill=3,
                    strength=8,
                    ap=-3,
                    damage=2,
                ),
                Weapon(name="Meltagun", attacks=1, skill=3, strength=9, ap=-4, damage=Dice(1, 6)),
                Weapon(
                    name="Plasma gun - standard", attacks=1, skill=3, strength=7, ap=-2, damage=1
                ),
                Weapon(
                    name="Plasma gun - supercharge", attacks=1, skill=3, strength=8, ap=-3, damage=2
                ),
                Weapon(name="Power sword", attacks=4, skill=3, strength=5, ap=-2, damage=1),
                Weapon(name="Close combat weapon", attacks=3, skill=3, strength=4, ap=0, damage=1),
            ],
        ),
        Unit(
            name="Flawless Blades",
            toughness=5,
            saving_throw=3,
            wounds=3,
            models=6,
            invuln=5,
            weapons=[
                Weapon(name="Bolt pistol", attacks=1, skill=3, strength=4, ap=0, damage=1),
                Weapon(name="Blissblade", attacks=4, skill=2, strength=6, ap=-3, damage=2),
            ],
        ),
        Unit(
            name="Noise Marines",
            toughness=5,
            saving_throw=3,
            wounds=2,
            models=6,
            invuln=None,
            weapons=[
                Weapon(name="Sonic blaster", attacks=3, skill=3, strength=5, ap=-1, damage=2),
                Weapon(
                    name="Blastmaster - varied frequency",
                    attacks=6,
                    skill=3,
                    strength=6,
                    ap=-2,
                    damage=1,
                ),
                Weapon(
                    name="Blastmaster - single frequency",
                    attacks=3,
                    skill=3,
                    strength=10,
                    ap=-2,
                    damage=3,
                ),
            ],
        ),
    ]

    tom_units = [
        Unit(
            name="Eldrad Ulthran",
            toughness=4,
            saving_throw=6,
            wounds=5,
            models=1,
            invuln=4,
            weapons=[
                Weapon(name="Mind War", attacks=1, skill=2, strength=5, ap=-2, damage=Dice(1, 6)),
                Weapon(name="Shuriken Pistol", attacks=1, skill=2, strength=4, ap=-1, damage=1),
                Weapon(
                    name="Staff of Ulthamar and witchblade",
                    attacks=3,
                    skill=2,
                    strength=5,
                    ap=-1,
                    damage=2,
                ),
            ],
        ),
        Unit(
            name="Spiritseer",
            toughness=3,
            saving_throw=6,
            wounds=3,
            models=1,
            invuln=4,
            weapons=[
                Weapon(name="Witch Staff", attacks=2, skill=2, strength=3, ap=0, damage=Dice(1, 3))
            ],
        ),
        Unit(
            name="Guardian Defenders",
            toughness=3,
            saving_throw=4,
            wounds=1,
            models=11,
            invuln=None,
            weapons=[
                Weapon(
                    name="Shuriken Catapult",
                    attacks=2,
                    skill=3,
                    strength=4,
                    ap=-1,
                    damage=1,
                    models_equipped=10,
                ),
                Weapon(
                    name="Bright Lance",
                    attacks=1,
                    skill=3,
                    strength=12,
                    ap=-3,
                    damage=Dice(1, 6, 2),
                    models_equipped=1,
                ),
                Weapon(
                    name="Close Combat Weapon", attacks=1, skill=3, strength=3, ap=0, damage=1
                ),  # Applies to all 11
            ],
        ),
        Unit(
            name="Wraithblades",
            toughness=6,
            saving_throw=2,
            wounds=3,
            models=5,
            invuln=4,
            weapons=[Weapon(name="Ghostaxe", attacks=3, skill=4, strength=7, ap=-2, damage=2)],
        ),
        Unit(
            name="Wraithguard",
            toughness=6,
            saving_throw=2,
            wounds=3,
            models=5,
            invuln=None,
            weapons=[
                Weapon(
                    name="D-Scythe",
                    attacks=Dice(1, 6),
                    skill=0,
                    strength=7,
                    ap=-3,
                    damage=1,
                    torrent=True,
                ),
                Weapon(
                    name="Wraithcannon",
                    attacks=1,
                    skill=4,
                    strength=14,
                    ap=-4,
                    damage=Dice(1, 6, 1),
                ),
                Weapon(name="Close Combat Weapon", attacks=3, skill=4, strength=5, ap=0, damage=1),
            ],
        ),
        Unit(
            name="Windriders",
            toughness=4,
            saving_throw=4,
            wounds=2,
            models=3,
            invuln=6,
            weapons=[
                Weapon(
                    name="Twin shuriken catapult",
                    attacks=2,
                    skill=3,
                    strength=4,
                    ap=-1,
                    damage=1,
                    reroll_wound_ones=True,
                ),
                Weapon(name="Close combat weapon", attacks=3, skill=3, strength=3, ap=0, damage=1),
            ],
        ),
        Unit(
            name="Wraithknight with Ghostglaive",
            toughness=12,
            saving_throw=2,
            wounds=18,
            models=1,
            invuln=4,
            weapons=[
                Weapon(
                    name="Shuriken Cannon",
                    attacks=3,
                    skill=3,
                    strength=6,
                    ap=-1,
                    damage=2,
                    lethal_hits=True,
                ),
                Weapon(
                    name="Titanic Ghostglaive - Strike",
                    attacks=5,
                    skill=3,
                    strength=16,
                    ap=-3,
                    damage=6,
                ),
                Weapon(
                    name="Titanic Ghostglaive - Sweep",
                    attacks=15,
                    skill=3,
                    strength=8,
                    ap=-2,
                    damage=2,
                ),
            ],
        ),
        Unit(
            name="Wraithlord",
            toughness=10,
            saving_throw=2,
            wounds=10,
            models=1,
            invuln=None,
            weapons=[
                Weapon(
                    name="Flamer",
                    attacks=Dice(1, 6),
                    skill=0,
                    strength=4,
                    ap=0,
                    damage=1,
                    torrent=True,
                ),
                Weapon(
                    name="Bright Lance",
                    attacks=1,
                    skill=4,
                    strength=12,
                    ap=-3,
                    damage=Dice(1, 6, 2),
                ),
                Weapon(
                    name="Ghostglaive Strike",
                    attacks=4,
                    skill=4,
                    strength=10,
                    ap=-3,
                    damage=Dice(1, 6, 1),
                ),
                Weapon(name="Ghostglaive Sweep", attacks=8, skill=4, strength=7, ap=-2, damage=2),
            ],
        ),
    ]

    plot_weapon_vs_unit_heatmap(alf_units, tom_units, simulate)
    plot_weapon_vs_unit_heatmap(tom_units, alf_units, simulate)

    plot_weapon_vs_unit_kills_heatmap(alf_units, tom_units, simulate)
    plot_weapon_vs_unit_kills_heatmap(tom_units, alf_units, simulate)
