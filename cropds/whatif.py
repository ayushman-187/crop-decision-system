"""whatif.py - what-if agronomic recommendation engine.

Finds the CHEAPEST combination of controllable changes that lifts a crop's Suitability Index by `target` points
(capped at 100). Levers: N, P, K (can only be ADDED), soil pH (lime up / sulphur down), plus optional
user-supplied alternative-season climates ("what if I sow later?").  Only levers the user actually provided are used.

APPROXIMATE conversions (always shown with a disclaimer):
    dataset nutrient unit ~ 1 kg/ha of N / P2O5 / K2O     (dataset NPK are relative levels, NOT calibrated lab values)
    Urea 46% N | DAP 46% P2O5 | MOP 60% K2O                (standard fertiliser grades)
    Lime ~ 2 t/ha per +1.0 pH unit                         (very rough; depends strongly on soil texture)
"""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd

from .data import FEATURES, CONTROLLABLE, perturb_inputs

DISCLAIMER = ("Approximate guidance only. Dataset N-P-K are relative levels, not lab-calibrated kg/ha; actual doses "
              "depend on soil test, soil texture, crop stage and local extension advice.")
GRADE = dict(N=("urea", 0.46), P=("DAP", 0.46), K=("MOP (muriate of potash)", 0.60))
LIME_T_PER_PH = 2.0
COST_WEIGHT = dict(N=1.0, P=1.2, K=1.0, ph=1.5)          # relative effort of one normalised step (editable)
MAX_ADD = dict(N=60, P=40, K=40)                         # realistic one-season top-up limits
PH_RANGE = (-1.0, 1.0)


def _grid(base, lever, n):
    return base + (np.linspace(*PH_RANGE, n) if lever == "ph" else np.linspace(0, MAX_ADD[lever], n))


def _si(model, X, crop):
    return model.suitability_index(X)[:, list(model.classes_).index(crop)]


def _effort(base, new, lever):
    scale = {"N": MAX_ADD["N"], "P": MAX_ADD["P"], "K": MAX_ADD["K"], "ph": 1.0}[lever]
    return COST_WEIGHT[lever] * abs(new - base) / scale


def search(model, field: pd.Series, crop: str, target=15.0, cap=100.0, levers=None, season_scenarios=None):
    """-> (base_SI, goal, plans sorted by effort, best_effort_plan_or_None)."""
    base_row = field[FEATURES].astype(float)
    levers = [l for l in (levers or CONTROLLABLE) if not np.isnan(base_row[l])]
    base_si = float(_si(model, base_row.to_frame().T, crop)[0])
    goal = min(base_si + target, cap)
    cands, bests = [], []
    for k in (1, 2):
        for combo in itertools.combinations(levers, k):
            grids = [_grid(base_row[l], l, 41 if k == 1 else 17) for l in combo]
            pts = np.array(list(itertools.product(*grids)))
            X = pd.DataFrame([base_row] * len(pts)).reset_index(drop=True)
            for j, l in enumerate(combo):
                X[l] = pts[:, j]
            if "ph" in combo:
                X["ph"] = X["ph"].clip(3.5, 9.5)
            si = _si(model, X, crop)
            ib = int(si.argmax())
            bests.append(dict(levers=combo, changes={l: float(X.loc[ib, l] - base_row[l]) for l in combo},
                              new_SI=float(si[ib]), effort=sum(_effort(base_row[l], X.loc[ib, l], l) for l in combo)))
            for i in np.where(si >= goal - 1e-9)[0]:
                if k == 2 and any(abs(X.loc[i, l] - base_row[l]) < (0.05 if l == "ph" else 0.5) for l in combo):
                    continue
                cands.append(dict(levers=combo, changes={l: float(X.loc[i, l] - base_row[l]) for l in combo},
                                  new_SI=float(si[i]), effort=sum(_effort(base_row[l], X.loc[i, l], l) for l in combo)))
    for name, clim in (season_scenarios or {}).items():          # user-supplied sowing-window climates
        X = base_row.copy()
        for k, v in clim.items():
            X[k] = v
        si = float(_si(model, X.to_frame().T, crop)[0])
        if si >= goal:
            cands.append(dict(levers=("season",), changes={"season": name}, new_SI=si, effort=0.3))
    cands.sort(key=lambda c: (c["effort"], len(c["levers"])))
    single_ok = {c["levers"][0] for c in cands if len(c["levers"]) == 1}
    seen, plans = set(), []
    for c in cands:
        if c["levers"] in seen: continue
        if len(c["levers"]) == 2 and any(l in single_ok for l in c["levers"]): continue   # one lever already suffices
        seen.add(c["levers"]); plans.append(c)
    best_effort = None
    if not plans and bests:                                       # goal unreachable -> best achievable improvement
        mx = max(b["new_SI"] for b in bests)
        near = sorted([b for b in bests if b["new_SI"] >= mx - 1.0], key=lambda b: b["effort"])
        best_effort = near[0] if mx > base_si + 3.0 else None
    return base_si, goal, plans, best_effort


def to_advice(plan, field: pd.Series):
    """Plan -> farmer-readable sentences with approximate fertiliser / lime quantities."""
    out = []
    for l, d in plan["changes"].items():
        if l == "season":
            out.append(f"Shift sowing window to: {d}."); continue
        if l == "ph":
            if abs(d) < 0.05: continue
            if d > 0:
                out.append(f"Raise soil pH by ~{d:.1f} (from {field['ph']:.1f}): ~{abs(d)*LIME_T_PER_PH:.1f} t/ha agricultural lime (very rough).")
            else:
                out.append(f"Lower soil pH by ~{abs(d):.1f} (from {field['ph']:.1f}): elemental sulphur / acidifying fertiliser - get a lab dose.")
            continue
        if d < 0.5: continue
        name, frac = GRADE[l]
        out.append(f"Add ~{d:.0f} kg/ha {l} (~{d/frac:.0f} kg/ha {name}).")
    return out


def robustness(model, field: pd.Series, crop: str, plan, sigma=0.05, n=60):
    """Median SI AFTER applying the plan under 5% input error. The forest is piecewise-constant, so a plan that
    only works at one exact value is flagged FRAGILE."""
    new = field[FEATURES].astype(float).copy()
    for l, d in plan["changes"].items():
        if l != "season":
            new[l] = new[l] + d
    X = perturb_inputs(new.to_frame().T, model.std_, sigma, n, np.random.default_rng(3))
    return float(np.median(_si(model, X, crop)))


def recommend(model, field: pd.Series, crops: list, target=15.0, season_scenarios=None, top_plans=2):
    """What-if advice for several crops (typically rank 1 and rank 2)."""
    res = {}
    for c in crops:
        base, goal, plans, be = search(model, field, c, target, season_scenarios=season_scenarios)
        for p in plans:
            p["robust_SI"] = robustness(model, field, c, p)
            p["robust"] = p["robust_SI"] >= base + 0.75 * (goal - base)
        plans.sort(key=lambda p: (not p["robust"], p["effort"]))
        tbl = model.fuzzy_.explain(field, c).dropna(subset=["membership"])
        worst = tbl.sort_values(["membership", "weight"], ascending=[True, False]).iloc[0]
        res[c] = dict(base_SI=base, goal=goal, reachable=bool(plans),
                      plans=[dict(p, advice=to_advice(p, field)) for p in plans[:top_plans]],
                      best_effort=None if be is None else dict(be, advice=to_advice(be, field)),
                      limiting=(worst.feature if worst.membership < 0.999 else None),
                      limiting_controllable=bool(worst.feature in CONTROLLABLE))
    return res


def live_rerank(model, base_field: pd.Series, deltas: dict, top=5):
    """For the app's what-if sliders: SI of every crop before / after applying `deltas` (e.g. {'N': +20, 'ph': -0.5}).
    Returns a DataFrame of the top crops with rank change."""
    new = base_field.copy()
    for k, d in deltas.items():
        if not np.isnan(new[k]):
            new[k] = new[k] + d
    new["ph"] = np.clip(new["ph"], 3.5, 9.9) if not np.isnan(new["ph"]) else new["ph"]
    a = model.suitability_index(base_field.to_frame().T)[0]
    b = model.suitability_index(new.to_frame().T)[0]
    t = pd.DataFrame(dict(crop=model.classes_, SI_before=a, SI_after=b))
    t["rank_before"] = t.SI_before.rank(ascending=False, method="first").astype(int)
    t["rank_after"] = t.SI_after.rank(ascending=False, method="first").astype(int)
    t["change"] = t.rank_before - t.rank_after
    t["delta_SI"] = t.SI_after - t.SI_before
    keep = set(t.nsmallest(top, "rank_before").crop) | set(t.nsmallest(top, "rank_after").crop)
    return t[t.crop.isin(keep)].sort_values("rank_after").reset_index(drop=True)
