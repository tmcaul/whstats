import dataclasses
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from whstats.sim import Unit, simulate, calculate_remaining_models
from whstats.units import load_armies


def _build_heatmap_figure(
    df_median: pd.DataFrame, df_annot: pd.DataFrame, metric: str, row_axis_title: str
) -> go.Figure:
    if metric == "damage":
        title = "Total Wounds Dealt"
        cbar_title = "Median Damage"
        zmax = None
        colorscale = "YlOrRd"
    else:
        title = "% Models Killed"
        cbar_title = "Median % Killed"
        zmax = 100
        colorscale = "Reds"

    fig = go.Figure(
        data=go.Heatmap(
            z=df_median.values,
            x=df_median.columns.tolist(),
            y=df_median.index.tolist(),
            texttemplate="%{z:.0f}",
            textfont={"size": 10},
            colorscale=colorscale,
            zmin=0,
            zmax=zmax,
            colorbar={
                "title": {"text": cbar_title, "side": "top"},
                "orientation": "h",
                "x": 0.5,
                "xanchor": "center",
                "y": 1,
                "yanchor": "bottom",
                "len": 0.8,
                "thickness": 15,
            },
            customdata=df_annot.values,
            hovertemplate=f"<b>{row_axis_title}:</b> %{{y}}<br><b>Target:</b> %{{x}}<br><b>Median:</b> %{{z:.1f}}<br><b>(P25-P50-P75):</b> %{{customdata}}<extra></extra>",
        )
    )

    fig.update_layout(
        title={"text": title, "font": {"size": 16}},
        yaxis={"autorange": "reversed", "tickangle": -30},
        height=max(450, len(df_median) * 30 + 120) + 70,
        margin=dict(l=120, r=40, t=130, b=60),
    )

    return fig


def create_unit_heatmap(
    attackers: list[Unit],
    targets: list[Unit],
    phase: str,
    simulate_fn,
    metric: str = "damage",
    trials: int = 1000,
) -> go.Figure:
    """Rows are whole units (all of that unit's weapons in the given phase, combined)."""
    median_data = []
    annot_data = []

    for attacker in attackers:
        phase_weapons = [w for w in attacker.weapons if w.phase == phase]
        med_row = {"Unit": attacker.name}
        ann_row = {"Unit": attacker.name}

        for target in targets:
            if not phase_weapons:
                med_row[target.name] = 0
                ann_row[target.name] = "—"
                continue

            combined = np.concatenate(
                [
                    simulate_fn(target=target, weapon=weapon, attacker=attacker, trials=trials)
                    for weapon in phase_weapons
                ],
                axis=1,
            )

            if metric == "damage":
                values = combined.sum(axis=1)
                p25, p50, p75 = np.percentile(values, [25, 50, 75])
                med_row[target.name] = p50
                ann_row[target.name] = f"{p25:.0f} - {p50:.0f} - {p75:.0f}"
            else:  # % models killed
                remaining_models = calculate_remaining_models(target, combined)
                models_killed = target.models - remaining_models
                pct_killed = (models_killed / target.models) * 100.0
                p25, p50, p75 = np.percentile(pct_killed, [25, 50, 75])
                med_row[target.name] = p50
                ann_row[target.name] = f"{p25:.0f} {p50:.0f} {p75:.0f}"

        median_data.append(med_row)
        annot_data.append(ann_row)

    df_median = pd.DataFrame(median_data).set_index("Unit")
    df_annot = pd.DataFrame(annot_data).set_index("Unit")

    return _build_heatmap_figure(df_median, df_annot, metric, "Unit")


def apply_modifiers(
    units: list[Unit], hit_mod: int, wound_mod: int, ap_mod: int, model_pct: float = 1.0
) -> list[Unit]:
    """Helper to dynamically inject modifiers and scale starting models"""
    modified_units = []
    for u in units:
        current_models = max(1, int(np.ceil(u.models * model_pct)))
        model_ratio = current_models / u.models if u.models > 0 else 1.0

        mod_weapons = []
        for w in u.weapons:
            new_hit = max(-1, min(1, w.hit_bonus + hit_mod))
            new_wound = max(-1, min(1, w.wound_bonus + wound_mod))
            new_ap = min(0, w.ap + ap_mod)

            if w.models_equipped is not None:
                new_equipped = max(1, int(round(w.models_equipped * model_ratio)))
            else:
                new_equipped = None

            mod_w = dataclasses.replace(
                w, hit_bonus=new_hit, wound_bonus=new_wound, ap=new_ap, models_equipped=new_equipped
            )
            mod_weapons.append(mod_w)

        modified_units.append(dataclasses.replace(u, models=current_models, weapons=mod_weapons))
    return modified_units


def _render_outcome_summary(
    total_wounds: np.ndarray,
    models_killed: np.ndarray,
    def_unit: Unit,
    trials: int,
    key_prefix: str,
) -> None:
    """Renders median/wipeout metrics plus wounds & kills histograms for a damage matrix outcome."""
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Median Wounds", f"{np.median(total_wounds):.1f}")
    metric_col2.metric("Median Models Killed", f"{np.median(models_killed):.1f}")
    metric_col3.metric(
        "Wipeout Probability",
        f"{(np.sum(models_killed == def_unit.models) / trials) * 100:.1f}%",
    )

    plot_col1, plot_col2 = st.columns(2)
    with plot_col1:
        fig_wounds = go.Figure(
            data=[
                go.Histogram(
                    x=total_wounds,
                    xbins=dict(size=1),
                    marker_color="indianred",
                    hovertemplate="Wounds Dealt: %{x}<br>Frequency: %{y}<extra></extra>",
                )
            ]
        )
        fig_wounds.update_layout(
            title="Total Wounds Dealt Distribution",
            xaxis_title="Wounds Dealt",
            yaxis_title="Frequency",
            bargap=0.1,
        )
        st.plotly_chart(fig_wounds, width="stretch", key=f"{key_prefix}_wounds_hist")

    with plot_col2:
        fig_kills_hist = go.Figure(
            data=[
                go.Histogram(
                    x=models_killed,
                    xbins=dict(size=1),
                    marker_color="royalblue",
                    hovertemplate="Models Killed: %{x}<br>Frequency: %{y}<extra></extra>",
                )
            ]
        )
        fig_kills_hist.update_layout(
            title="Models Killed Distribution",
            xaxis_title="Models Killed",
            yaxis_title="Frequency",
            bargap=0.1,
            xaxis=dict(tickmode="linear", tick0=0, dtick=1),
        )
        st.plotly_chart(fig_kills_hist, width="stretch", key=f"{key_prefix}_kills_hist")


def render_unit_vs_unit_page():
    st.title("Drilldown")
    st.markdown(
        "See the overall outcome of a unit's full attack against a single defender, then pick a phase "
        "and weapon below to drill into its exact outcome distribution."
    )
    render_unit_vs_unit_fragment()


def _fmt_dice_or_int(v) -> str:
    if isinstance(v, int):
        return str(v)
    s = f"{v.n}d{v.sides}"
    if v.plus:
        s += f"+{v.plus}"
    return s


def _weapons_to_editor_df(unit: Unit) -> pd.DataFrame:
    rows = []
    for w in unit.weapons:
        default_count = w.models_equipped if w.models_equipped is not None else unit.models
        rows.append(
            {
                "Count": default_count,
                "Weapon": w.name,
                "Phase": w.phase,
                "Attacks": _fmt_dice_or_int(w.attacks),
                "Skill": w.skill,
                "Strength": w.strength,
                "AP": w.ap,
                "Damage": _fmt_dice_or_int(w.damage),
                "Hit Bonus": w.hit_bonus,
                "Wound Bonus": w.wound_bonus,
                "Torrent": w.torrent,
                "Sustained Hits": w.sustained_hits,
                "Lethal Hits": w.lethal_hits,
                "Devastating Wounds": w.devastating_wounds,
                "Reroll Hit Ones": w.reroll_hit_ones,
                "Reroll All Hits": w.reroll_all_hits,
                "Reroll Wound Ones": w.reroll_wound_ones,
                "Reroll All Wounds": w.reroll_all_wounds,
            }
        )
    return pd.DataFrame(rows)


def _defender_to_editor_df(unit: Unit) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Models": unit.models,
                "Toughness": unit.toughness,
                "Wounds": unit.wounds,
                "Save": unit.saving_throw,
                "Invuln (7 = None)": unit.invuln if unit.invuln is not None else 7,
                "FNP (7 = None)": unit.fnp if unit.fnp is not None else 7,
                "Damage Modifier": unit.damage_modifier,
                "Hit Bonus": unit.hit_bonus,
                "Wound Bonus": unit.wound_bonus,
            }
        ]
    )


@st.fragment
def render_unit_vs_unit_fragment():
    """Wrapped in a fragment so modifying inputs does NOT reset scroll page height"""
    army_names = list(load_armies().keys())
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Attacker Setup")
        atk_army = st.selectbox("Attacker Army", army_names, key="atk_army_uvu")

        atk_units = {u.name: u for u in load_armies()[atk_army]}
        if st.session_state.get("atk_unit_uvu") not in atk_units:
            st.session_state.atk_unit_uvu = list(atk_units.keys())[0]

        atk_unit_name = st.selectbox("Attacker Unit", list(atk_units.keys()), key="atk_unit_uvu")
        atk_unit = atk_units[atk_unit_name]

        if "prev_atk_unit_uvu" not in st.session_state:
            st.session_state.prev_atk_unit_uvu = atk_unit_name
            st.session_state.atk_models_uvu = atk_unit.models
        elif st.session_state.prev_atk_unit_uvu != atk_unit_name:
            st.session_state.atk_models_uvu = atk_unit.models
            st.session_state.prev_atk_unit_uvu = atk_unit_name

        atk_models = st.number_input(
            "Starting Models (Attacker)", min_value=1, key="atk_models_uvu"
        )

    with col2:
        st.subheader("Defender Setup")
        def_army = st.selectbox("Defender Army", army_names, key="def_army_uvu")

        def_units = {u.name: u for u in load_armies()[def_army]}
        if st.session_state.get("def_unit_uvu") not in def_units:
            st.session_state.def_unit_uvu = list(def_units.keys())[0]

        def_unit_name = st.selectbox("Defender Unit", list(def_units.keys()), key="def_unit_uvu")
        def_unit = def_units[def_unit_name]

        def_stats_df = _defender_to_editor_df(def_unit)
        edited_def_row = st.data_editor(
            def_stats_df,
            key=f"def_stats_editor_uvu_{def_army}_{def_unit_name}",
            hide_index=True,
            width="stretch",
            num_rows="fixed",
            column_config={
                "Models": st.column_config.NumberColumn(min_value=1, step=1),
                "Toughness": st.column_config.NumberColumn(min_value=1, step=1),
                "Wounds": st.column_config.NumberColumn(min_value=1, step=1),
                "Save": st.column_config.NumberColumn(min_value=2, max_value=7, step=1),
                "Invuln (7 = None)": st.column_config.NumberColumn(min_value=2, max_value=7, step=1),
                "FNP (7 = None)": st.column_config.NumberColumn(min_value=2, max_value=7, step=1),
                "Damage Modifier": st.column_config.NumberColumn(max_value=0, step=1),
                "Hit Bonus": st.column_config.NumberColumn(min_value=-1, max_value=1, step=1),
                "Wound Bonus": st.column_config.NumberColumn(min_value=-1, max_value=1, step=1),
            },
        ).iloc[0]

        def_models = int(edited_def_row["Models"])
        def_t = int(edited_def_row["Toughness"])
        def_w = int(edited_def_row["Wounds"])
        def_sv = int(edited_def_row["Save"])
        def_inv = int(edited_def_row["Invuln (7 = None)"])
        def_fnp = int(edited_def_row["FNP (7 = None)"])
        def_dmg_mod = int(edited_def_row["Damage Modifier"])
        def_hit_bonus = int(edited_def_row["Hit Bonus"])
        def_wound_bonus = int(edited_def_row["Wound Bonus"])

    st.markdown("**Attacker Weapons**")
    weapons_df = _weapons_to_editor_df(atk_unit)
    edited_df = st.data_editor(
        weapons_df,
        key=f"atk_weapons_editor_uvu_{atk_army}_{atk_unit_name}",
        hide_index=True,
        width="stretch",
        num_rows="fixed",
        column_config={
            "Count": st.column_config.NumberColumn(min_value=0, step=1),
            "Weapon": st.column_config.TextColumn(disabled=True),
            "Phase": st.column_config.TextColumn(disabled=True),
            "Attacks": st.column_config.TextColumn(disabled=True),
            "Damage": st.column_config.TextColumn(disabled=True),
            "Skill": st.column_config.NumberColumn(min_value=2, max_value=6, step=1),
            "Strength": st.column_config.NumberColumn(min_value=1, step=1),
            "AP": st.column_config.NumberColumn(max_value=0, step=1),
            "Hit Bonus": st.column_config.NumberColumn(min_value=-1, max_value=1, step=1),
            "Wound Bonus": st.column_config.NumberColumn(min_value=-1, max_value=1, step=1),
            "Torrent": st.column_config.CheckboxColumn(),
            "Sustained Hits": st.column_config.CheckboxColumn(),
            "Lethal Hits": st.column_config.CheckboxColumn(),
            "Devastating Wounds": st.column_config.CheckboxColumn(),
            "Reroll Hit Ones": st.column_config.CheckboxColumn(),
            "Reroll All Hits": st.column_config.CheckboxColumn(),
            "Reroll Wound Ones": st.column_config.CheckboxColumn(),
            "Reroll All Wounds": st.column_config.CheckboxColumn(),
        },
    )

    trials = st.number_input(
        "Simulation Trials", min_value=100, max_value=10000, step=500, value=10000, key="trials_uvu"
    )

    weapons_by_name = {w.name: w for w in atk_unit.weapons}
    edited_weapons = [
        dataclasses.replace(
            weapons_by_name[row["Weapon"]],
            skill=int(row["Skill"]),
            strength=int(row["Strength"]),
            ap=int(row["AP"]),
            hit_bonus=int(row["Hit Bonus"]),
            wound_bonus=int(row["Wound Bonus"]),
            torrent=bool(row["Torrent"]),
            sustained_hits=bool(row["Sustained Hits"]),
            lethal_hits=bool(row["Lethal Hits"]),
            devastating_wounds=bool(row["Devastating Wounds"]),
            reroll_hit_ones=bool(row["Reroll Hit Ones"]),
            reroll_all_hits=bool(row["Reroll All Hits"]),
            reroll_wound_ones=bool(row["Reroll Wound Ones"]),
            reroll_all_wounds=bool(row["Reroll All Wounds"]),
            models_equipped=int(row["Count"]),
        )
        for _, row in edited_df.iterrows()
    ]
    atk_unit_mod = dataclasses.replace(atk_unit, models=atk_models, weapons=edited_weapons)

    parsed_invuln = None if def_inv == 7 else def_inv
    parsed_fnp = None if def_fnp == 7 else def_fnp
    def_unit_mod = dataclasses.replace(
        def_unit,
        models=def_models,
        toughness=def_t,
        saving_throw=def_sv,
        wounds=def_w,
        invuln=parsed_invuln,
        fnp=parsed_fnp,
        damage_modifier=def_dmg_mod,
        hit_bonus=def_hit_bonus,
        wound_bonus=def_wound_bonus,
    )

    st.divider()
    st.subheader("Overall Unit vs Unit")
    overall_phase_label = st.selectbox(
        "Phase", ["Melee", "Shooting", "Both"], index=2, key="overall_phase_uvu"
    )
    overall_phase_key = {"Melee": "melee", "Shooting": "ranged", "Both": None}[overall_phase_label]
    overall_weapons = [
        w for w in atk_unit_mod.weapons if overall_phase_key is None or w.phase == overall_phase_key
    ]

    if not overall_weapons:
        st.caption(f"{atk_unit_name} has no {overall_phase_label.lower()} weapons.")
    else:
        with st.spinner("Rolling the dice..."):
            combined_matrix = np.concatenate(
                [
                    simulate(target=def_unit_mod, weapon=w, attacker=atk_unit_mod, trials=trials)
                    for w in overall_weapons
                ],
                axis=1,
            )
        total_wounds = combined_matrix.sum(axis=1)
        rem_models = calculate_remaining_models(def_unit_mod, combined_matrix)
        models_killed = def_unit_mod.models - rem_models
        _render_outcome_summary(total_wounds, models_killed, def_unit_mod, trials, "uvu_overall")

    st.divider()
    st.subheader("Weapon Detail")

    detail_col1, detail_col2, detail_col3 = st.columns(3)
    with detail_col1:
        phase_label = st.selectbox("Phase", ["Melee", "Shooting"], key="detail_phase_uvu")
    phase_key = "melee" if phase_label == "Melee" else "ranged"
    detail_options = [w.name for w in atk_unit_mod.weapons if w.phase == phase_key]

    if not detail_options:
        st.caption(f"{atk_unit_name} has no {phase_label.lower()} weapons.")
    else:
        if st.session_state.get("detail_wep_uvu") not in detail_options:
            st.session_state.detail_wep_uvu = detail_options[0]
        with detail_col2:
            detail_weapon_name = st.selectbox("Select Weapon", detail_options, key="detail_wep_uvu")
        detail_weapon = next(w for w in atk_unit_mod.weapons if w.name == detail_weapon_name)

        default_detail_count = (
            detail_weapon.models_equipped
            if detail_weapon.models_equipped is not None
            else atk_unit_mod.models
        )
        with detail_col3:
            detail_count = st.number_input(
                "Count",
                min_value=0,
                step=1,
                value=default_detail_count,
                key=f"detail_count_uvu_{detail_weapon_name}",
            )
        detail_weapon = dataclasses.replace(detail_weapon, models_equipped=detail_count)

        with st.spinner("Simulating..."):
            damage_matrix = simulate(
                target=def_unit_mod, weapon=detail_weapon, attacker=atk_unit_mod, trials=trials
            )

        total_wounds = damage_matrix.sum(axis=1)
        rem_models = calculate_remaining_models(def_unit_mod, damage_matrix)
        models_killed = def_unit_mod.models - rem_models
        _render_outcome_summary(total_wounds, models_killed, def_unit_mod, trials, "uvu_detail")


def render_army_unit_summary_page():
    st.title("Army")

    army_names = list(load_armies().keys())

    def flip_armies_summary():
        st.session_state.atk_army_summary, st.session_state.def_army_summary = (
            st.session_state.def_army_summary,
            st.session_state.atk_army_summary,
        )

    st.sidebar.header("Army Selection")
    attacker_label = st.sidebar.selectbox("Attacker Army", army_names, key="atk_army_summary")
    defender_label = st.sidebar.selectbox("Defender Army", army_names, key="def_army_summary")

    st.sidebar.header("Starting Unit Strengths")
    attacker_model_pct = (
        st.sidebar.slider("Attacker Starting Models %", 10, 100, 100, 10, key="atk_pct_summary")
        / 100.0
    )
    defender_model_pct = (
        st.sidebar.slider("Defender Starting Models %", 10, 100, 100, 10, key="def_pct_summary")
        / 100.0
    )

    st.sidebar.header("Attacker Buffs & Dice Setup")
    hit_mod = st.sidebar.slider("To Hit Modifier", -1, 1, 0, key="summary_hit")
    wound_mod = st.sidebar.slider("To Wound Modifier", -1, 1, 0, key="summary_wound")
    ap_mod = st.sidebar.slider("AP Modifier", -3, 1, 0, key="summary_ap")
    trials = st.sidebar.number_input(
        "Simulation Trials", 100, 10000, 10000, 500, key="summary_trials"
    )

    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.subheader(
            f"Scenario: {attacker_label} ({int(attacker_model_pct*100)}% strength) "
            f"attacking {defender_label} ({int(defender_model_pct*100)}% strength)"
        )
    with header_col2:
        st.button(
            "🔄 Flip Attacker / Defender",
            type="secondary",
            width="stretch",
            on_click=flip_armies_summary,
            key="flip_summary",
        )

    attackers_mod = apply_modifiers(
        load_armies()[attacker_label], hit_mod, wound_mod, ap_mod, attacker_model_pct
    )
    defenders_mod = apply_modifiers(load_armies()[defender_label], 0, 0, 0, defender_model_pct)

    for phase, label in [("melee", "Melee Phase"), ("ranged", "Shooting Phase")]:
        st.divider()
        st.subheader(label)
        with st.spinner("Rolling the dice..."):
            fig_damage = create_unit_heatmap(
                attackers_mod, defenders_mod, phase, simulate, metric="damage", trials=trials
            )
            fig_kills = create_unit_heatmap(
                attackers_mod, defenders_mod, phase, simulate, metric="pct_kills", trials=trials
            )
        st.plotly_chart(fig_damage, width="stretch", key=f"summary_{phase}_damage_chart")
        st.plotly_chart(fig_kills, width="stretch", key=f"summary_{phase}_kills_chart")


page_summary = st.Page(render_army_unit_summary_page, title="Army", icon="🗺️")
page_uvu = st.Page(render_unit_vs_unit_page, title="Drilldown", icon="⚔️")

if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Warhammer 40k Sim")

    # Global Session State Init
    army_names = list(load_armies().keys())
    default_attacker = army_names[0] if len(army_names) > 0 else None
    default_defender = (
        army_names[1] if len(army_names) > 1 else (army_names[0] if len(army_names) > 0 else None)
    )

    if "atk_army_summary" not in st.session_state:
        st.session_state.atk_army_summary = default_attacker
    if "def_army_summary" not in st.session_state:
        st.session_state.def_army_summary = default_defender

    if "atk_army_uvu" not in st.session_state:
        st.session_state.atk_army_uvu = default_attacker
    if "def_army_uvu" not in st.session_state:
        st.session_state.def_army_uvu = default_defender

    sidebar_col1, sidebar_col2 = st.sidebar.columns(2)
    sidebar_col1.link_button(
        "Army Data ↗",
        "https://docs.google.com/spreadsheets/d/1uqMFwcVcPISlDX2mWYYnykM0D3Sv8T1eJZzrZsoheWg/edit?gid=0#gid=0",
        width="stretch",
    )
    if sidebar_col2.button("🔄 Reload", width="stretch"):
        load_armies.cache_clear()
        st.rerun()

    pg = st.navigation([page_summary, page_uvu])
    pg.run()
