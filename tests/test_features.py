import numpy as np
import pandas as pd
import pytest

from src.features import (add_calendar, add_ewma, add_exog_features, add_fourier, add_lags, add_rolling,
                      build_features, lag_market_exog)


def _toy(n=120, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "date": pd.date_range("2018-01-01", periods=n, freq="B"),
        "balance": rng.normal(scale=0.3, size=n),
        "income": rng.random(n),
        "outcome": rng.random(n),
    })


# --- из плана ---------------------------------------------------------------
def test_add_lags():
    df = pd.DataFrame({"balance": np.arange(5, dtype=float)})
    out = add_lags(df, "balance", [1, 2])
    assert out["balance_lag_1"].iloc[2] == 1.0
    assert out["balance_lag_2"].iloc[2] == 0.0


def test_add_rolling_no_leakage():
    df = pd.DataFrame({"balance": [1.0, 2.0, 3.0, 4.0, 5.0]})
    out = add_rolling(df, "balance", [3], stats=("mean",))
    # для строки 3 окно = [1,2,3] (значения t-3..t-1), среднее = 2.0
    assert out["balance_roll_mean_3"].iloc[3] == 2.0


def test_add_calendar_dow():
    df = pd.DataFrame({"date": pd.to_datetime(["2020-01-06", "2020-01-07"])})
    out = add_calendar(df)
    assert out["dow"].iloc[0] == 0  # понедельник
    assert out["dow"].iloc[1] == 1


def test_build_features_no_nans():
    out = build_features(_toy())
    assert out.isna().sum().sum() == 0
    assert len(out) < 120  # часть строк теряется на лагах


# --- дополнительные проверки корректности -----------------------------------
def test_rolling_does_not_use_current_value():
    # если бы окно включало текущий день, среднее отличалось бы
    df = pd.DataFrame({"balance": [10.0, 0.0, 0.0, 0.0, 100.0]})
    out = add_rolling(df, "balance", [2], stats=("mean",))
    # строка 4: окно [t-2,t-1]=[0,0] -> 0, текущее 100 НЕ участвует
    assert out["balance_roll_mean_2"].iloc[4] == 0.0


def test_fourier_periodicity():
    out = add_fourier(_toy(), weekly_period=5)
    # гармоника недели должна почти повторяться через 5 бизнес-дней
    assert np.isclose(out["sin_w"].iloc[0], out["sin_w"].iloc[5], atol=1e-9)


def test_market_exog_is_lagged():
    df = _toy(30)
    df["usdrub"] = np.arange(30, dtype=float)
    out = lag_market_exog(df, cols=("usdrub",))
    assert "usdrub" not in out.columns  # contemporaneous удалён
    assert out["usdrub_lag1"].iloc[5] == 4.0  # значение предыдущего дня


def test_build_features_returns_cols():
    out, cols = build_features(_toy(), return_cols=True)
    assert "balance" not in cols and "date" not in cols
    assert all(c in out.columns for c in cols)
    assert len(cols) > 10


def test_exog_merge_and_lag():
    df = _toy(60)
    exog = pd.DataFrame({"date": df["date"], "key_rate": 7.5,
                         "is_tax_day": 0, "moex_ret": np.arange(60.0)})
    out, cols = build_features(df, exog=exog, return_cols=True)
    assert "key_rate" in cols          # директивный экзоген — без лага
    assert "moex_ret_lag1" in cols     # рыночный — лагирован
    assert "moex_ret" not in cols


def test_rolling_has_moments():
    df = _toy(80)
    out = add_rolling(df, "balance", [10])
    assert "balance_roll_skew_10" in out.columns
    assert "balance_roll_kurt_10" in out.columns


def test_build_features_rolling_on_components():
    out, cols = build_features(_toy(120), return_cols=True)
    # rolling/ewma должны строиться и по income/outcome, не только balance
    assert any(c.startswith("income_roll_") for c in cols)
    assert any(c.startswith("outcome_ewma_") for c in cols)
    assert any(c.startswith("balance_ewma_") for c in cols)


def test_salary_day_flag():
    df = pd.DataFrame({"date": pd.to_datetime(["2020-01-05", "2020-01-06"])})
    out = add_calendar(df, salary_days=(5, 20))
    assert out["is_salary_day"].tolist() == [1, 0]


def test_integration_with_p1_exog_schema():
    """Контракт P1->P2: exog в формате build_exog (уровни key_rate/ruonia/
    moex_close/usdrub + налоговый календарь) корректно преобразуется в
    модельные признаки, сырые уровни не утекают."""
    df = _toy(200)
    d = df["date"]
    exog = pd.DataFrame({
        "date": d,
        "is_tax_day": d.dt.day.isin([25, 28]).astype(int),
        "days_to_next_tax": (28 - d.dt.day) % 28,
        "days_since_last_tax": d.dt.day % 28,
        "key_rate": 7.0,
        "ruonia": 6.7,
        "moex_close": np.linspace(2500, 2700, len(d)),
        "usdrub": np.linspace(65, 70, len(d)),
    })
    out, cols = build_features(df, exog=exog, return_cols=True)
    # производные экзогены созданы
    for c in ("key_rate", "key_rate_diff5", "ruonia_spread_lag1",
              "moex_ret_lag1", "usdrub_ret_lag1"):
        assert c in cols, c
    # сырые рыночные уровни НЕ должны попасть в признаки (lookahead)
    for c in ("ruonia", "moex_close", "usdrub"):
        assert c not in cols
    assert out.isna().sum().sum() == 0
