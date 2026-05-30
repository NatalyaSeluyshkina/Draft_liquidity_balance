import numpy as np
import pandas as pd
import pytest

from src.selection import (FeatureSelector, choose_best_selector, compare_selectors,
                       jaccard_stability, nogueira_stability, select_correlation,
                       select_lasso, select_mi, select_phik, select_rfe_rf,
                       top_features_report)


def _signal_noise(n=500, seed=0):
    """signal коррелирует с y, остальные — шум."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "signal": rng.normal(size=n),
        "signal2": rng.normal(size=n),
        "noise1": rng.normal(size=n),
        "noise2": rng.normal(size=n),
        "big_noise": rng.normal(scale=1000, size=n),  # большой масштаб
    })
    y = (X["signal"] + 0.5 * X["signal2"] + 0.05 * rng.normal(size=n)).values
    return X, y


# --- из плана ---------------------------------------------------------------
def test_select_mi_picks_informative():
    X, y = _signal_noise()
    chosen = select_mi(X, y, k=1)
    assert chosen == ["signal"]


def test_jaccard_in_range():
    X, y = _signal_noise(300)
    stab = jaccard_stability(select_mi, X, y, n_splits=3, k=2)
    assert 0.0 <= stab <= 1.0


# --- проверка ИСПРАВЛЕННОГО бага: Lasso должен масштабировать -----------------
def test_lasso_selects_signal():
    X, y = _signal_noise()
    chosen = select_lasso(X, y)
    assert "signal" in chosen


def test_lasso_scale_invariant():
    """Главное свойство, которое чинит StandardScaler: отбор не должен зависеть
    от масштаба признака. Умножаем 'signal' на 1000 — результат отбора обязан
    совпасть. Без масштабирования (баг скелета) — не совпал бы."""
    X, y = _signal_noise()
    base = set(select_lasso(X, y))
    X2 = X.copy()
    X2["signal"] = X2["signal"] * 1000.0
    rescaled = set(select_lasso(X2, y))
    assert base == rescaled
    assert "signal" in base


# --- wrapper ----------------------------------------------------------------
def test_rfe_rf_selects_signal():
    X, y = _signal_noise()
    chosen = select_rfe_rf(X, y, k=2)
    assert "signal" in chosen


# --- стабильность -----------------------------------------------------------
def test_nogueira_identical_sets_is_one():
    feats = list("ABCDE")
    sets = [{"A", "B"}, {"A", "B"}, {"A", "B"}]
    assert np.isclose(nogueira_stability(sets, feats), 1.0)


def test_nogueira_disjoint_sets_low():
    feats = list("ABCDEF")
    sets = [{"A", "B"}, {"C", "D"}, {"E", "F"}]
    assert nogueira_stability(sets, feats) < 0.5


# --- сравнение и выбор -------------------------------------------------------
def test_compare_and_choose():
    X, y = _signal_noise(400)
    # ограниченный набор быстрых методов, чтобы тест шёл секунды
    from src.selection import select_correlation, select_mi, select_lasso
    quick = {
        "Correlation (filter, lin)": (select_correlation, "filter", "linear"),
        "MutualInfo (filter, nonlin)": (select_mi, "filter", "nonlinear"),
        "Lasso (embedded, lin)": (select_lasso, "embedded", "linear"),
    }
    cmp = compare_selectors(X, y, selectors=quick, n_splits=3)
    assert {"category", "kind", "jaccard", "nogueira"}.issubset(cmp.columns)
    assert set(cmp["category"]) == {"filter", "embedded"}
    best = choose_best_selector(cmp)
    assert best in cmp.index


# --- sklearn-трансформер ----------------------------------------------------
def test_feature_selector_transformer():
    X, y = _signal_noise()
    sel = FeatureSelector(method="rfe_rf", k=2).fit(X, y)
    Xt = sel.transform(X)
    assert Xt.shape[1] == 2
    assert "signal" in sel.get_feature_names_out().tolist()


def test_feature_selector_in_pipeline():
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    X, y = _signal_noise()
    pipe = Pipeline([("sel", FeatureSelector(method="mi", k=3)),
                     ("model", Ridge())]).fit(X, y)
    pred = pipe.predict(X)
    assert pred.shape == (len(y),)


def test_phik_runs_or_falls_back():
    X, y = _signal_noise(300)
    chosen = select_phik(X, y, k=2)
    assert isinstance(chosen, list) and len(chosen) == 2


def test_top_features_report():
    X, y = _signal_noise(400)
    rep = top_features_report(X, y, k=3)
    assert "mutual_info" in rep.columns
    assert len(rep) == 3
    assert "signal" in rep.index  # информативный фактор должен быть в топе
