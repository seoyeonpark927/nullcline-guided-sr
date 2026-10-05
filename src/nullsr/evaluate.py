"""Decide whether a PySR run recovered the target equation.

Every candidate in hall_of_fame.csv is expanded with SymPy and compared with
the target polynomial. A run succeeds if ANY candidate passes; the simplest
passing candidate is reported.

Targets
  direct : du/dt = f * (-u^3 + u^2 + u - v)      (inputs u, v)
  nullcline:   v = -u^3 + u^2 + u                (input u)

Criteria
  strict : exactly the target monomials (+ optional constant), every
           coefficient within +/-15 % of its true value, |constant| <= 0.15 f
  relaxed: exactly the target monomials (+ optional constant), negative cubic
           coefficient; coefficient values are not checked
"""

import re
from pathlib import Path

import pandas as pd
import sympy as sp

U, V = sp.symbols("u v")
_NAMES = {"x0": U, "x1": V, "u": U, "v": V, "marker": sp.Integer(0)}

TARGETS = {
    # monomial (power of u, power of v) -> coefficient for f = 1
    "direct": {(3, 0): -1.0, (2, 0): 1.0, (1, 0): 1.0, (0, 1): -1.0},
    "nullcline": {(3, 0): -1.0, (2, 0): 1.0, (1, 0): 1.0},
}
TOL = 0.15
ZERO = 1e-8


def parse(equation):
    text = str(equation).strip().strip('"').replace("^", "**")
    expr = sp.sympify(text, locals=_NAMES)
    if not isinstance(expr, sp.Basic):
        raise sp.SympifyError(equation)
    return sp.expand(expr.subs(_NAMES))


def coefficients(expr):
    if not expr.free_symbols <= {U, V}:
        return None
    try:
        return {m: float(c) for m, c in sp.Poly(expr, U, V).as_dict().items()}
    except (sp.PolynomialError, TypeError):
        return None


def passes(expr, target, scale, mode):
    coef = coefficients(expr)
    if coef is None:
        return False
    for mono, value in coef.items():
        if mono not in target and mono != (0, 0) and abs(value) > ZERO:
            return False
    for mono, true in target.items():
        value = coef.get(mono, 0.0)
        if abs(value) <= ZERO:
            return False
        if mode == "strict" and abs(value - true * scale) > TOL * scale:
            return False
    if coef[(3, 0)] >= 0:
        return False
    if mode == "strict" and abs(coef.get((0, 0), 0.0)) > TOL * scale:
        return False
    return True


def score_hall_of_fame(csv_path, kind, f):
    """Return {success_strict, equation_strict, complexity_strict, ...relaxed}."""
    target = TARGETS[kind]
    scale = f if kind == "direct" else 1.0
    hof = pd.read_csv(csv_path).sort_values("Complexity")
    parsed = []
    for _, row in hof.iterrows():
        try:
            parsed.append((row, parse(row["Equation"])))
        except (sp.SympifyError, TypeError, SyntaxError):
            continue

    result = {}
    for mode in ("strict", "relaxed"):
        hit = next((row for row, expr in parsed if passes(expr, target, scale, mode)), None)
        result[f"success_{mode}"] = int(hit is not None)
        result[f"complexity_{mode}"] = None if hit is None else int(hit["Complexity"])
        result[f"equation_{mode}"] = None if hit is None else hit["Equation"]
    return result


_KEY = re.compile(r"(?<![a-z])(f|dt|seed|lambda|iter)=([0-9.eE+-]+?)(?=_|$)")


def run_parameters(run_dir):
    """Read f, dt, seed, lambda, iter from a run folder name."""
    params = {k: float(v) for k, v in _KEY.findall(Path(run_dir).name)}
    params.setdefault("seed", 0)
    params["seed"] = int(params["seed"])
    if "dt" in params:
        params["dt"] = round(params["dt"], 6)
    return params


def score_runs(runs_dir, kind):
    """Score every hall_of_fame.csv below runs_dir; one row per run."""
    rows = []
    for csv_path in sorted(Path(runs_dir).rglob("hall_of_fame.csv")):
        params = run_parameters(csv_path.parent)
        if "f" not in params:
            continue
        rows.append({**params, **score_hall_of_fame(csv_path, kind, params["f"]),
                     "run": csv_path.parent.name})
    return pd.DataFrame(rows)
