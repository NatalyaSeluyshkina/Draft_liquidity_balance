"""
features.py — Feature engineering для прогноза сальдо ликвидности.

Зона ответственности P2 (Feature Engineer), трек 7.

Контракт интерфейса:
    Вход  (от P1): DataFrame c колонками
        [date, balance, income, outcome, key_rate, ruonia, moex_ret,
         usdrub, is_tax_day, days_to_next_tax, days_since_last_tax, is_holiday, ...]
        отсортированный по date, только бизнес-дни.
    Выход (к P3): DataFrame с готовыми фичами, целевой `balance`
        и атрибутом-списком фич (см. build_features(..., return_cols=True)).

Принципы:
  * Все признаки, опирающиеся на таргет/потоки, строятся ТОЛЬКО из прошлого
    (shift>=1). Прогноз делается в конце дня t-1 на день t.
  * Рыночные экзогены (moex_ret, usdrub, ruonia) на день t в конце t-1 ещё
    НЕ наблюдаемы -> по умолчанию лагируются на 1 день.
  * Календарные/директивные экзогены (key_rate, is_tax_day, is_holiday,
    days_to/since_tax) известны заранее -> используются без лага.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --- конфигурация по умолчанию (совпадает со спецификацией §4) ---------------
DEFAULT_LAGS = [1, 2, 3, 5, 7, 14, 21]
DEFAULT_WINDOWS = [5, 10, 21]
# mean/std/min/max + моменты: в работах-эталонах именно skew/kurt по Income/
# Outcome попадали в топ значимых признаков фильтрационных методов.
DEFAULT_ROLL_STATS = ("mean", "std", "min", "max", "skew", "kurt")
DEFAULT_EWMA_SPANS = [5, 10]
# по каким рядам строим rolling/ewma (компоненты сальдо часто информативнее
# самого сальдо — их потоки предсказуемее по отдельности)
DEFAULT_ROLL_COLS = ("balance", "income", "outcome")

# Доп. рыночные уровни, которые могут прийти от P1 под этими именами и которые
# нужно лагировать на 1 день (на случай расширения exog). Базовые уровни
# key_rate/ruonia/moex_close/usdrub обрабатывает add_exog_features.
MARKET_EXOG = ("inflation", "brent", "moex_ret", "ruonia_spread")


# ---------------------------------------------------------------------------
# Базовые конструкторы признаков
# ---------------------------------------------------------------------------
def add_lags(df: pd.DataFrame, col: str, lags: list[int]) -> pd.DataFrame:
    """Лаги столбца `col`. lag_k = значение k дней назад.

    Чем руководствовались: дневное сальдо обладает короткой памятью
    (значимые ACF на лагах 1,2 и недельный 5/7); лаги дают модели прямой
    доступ к авторегрессионной структуре без обучения отдельной AR-модели.
    """
    out = df.copy()
    for k in lags:
        out[f"{col}_lag_{k}"] = out[col].shift(k)
    return out


def add_rolling(
    df: pd.DataFrame,
    col: str,
    windows: list[int],
    stats=DEFAULT_ROLL_STATS,
) -> pd.DataFrame:
    """Скользящие статистики по `col` (mean/std/min/max/skew/kurt).

    Ключевая деталь против утечки: окно строится по shift(1), т.е. в признак
    для дня t попадают значения [t-w, ..., t-1], а не сам t.

    Зачем моменты: асимметрия (skew) и эксцесс (kurt) ловят "тяжесть хвостов"
    и всплески в потоках; в работах-эталонах эти признаки по Income/Outcome
    были среди самых значимых по mutual information / phik.
    """
    out = df.copy()
    base = out[col].shift(1)
    for w in windows:
        # для моментов нужно окно >= 4 наблюдений, иначе kurt не определён;
        # при этом min_periods не может превышать размер окна
        mp = min(w, max(4, w // 2))
        roll = base.rolling(w, min_periods=mp)
        for s in stats:
            out[f"{col}_roll_{s}_{w}"] = getattr(roll, s)()
    return out


def add_ewma(
    df: pd.DataFrame,
    col: str,
    spans=DEFAULT_EWMA_SPANS,
) -> pd.DataFrame:
    """Экспоненциально-взвешенные среднее и std по `col`.

    EWMA реагирует на свежие значения быстрее простого скользящего окна —
    полезно на ряде со сменой режима (после 2020 баланс «уехал» в минус).
    Считается по shift(1) -> без утечки.
    """
    out = df.copy()
    base = out[col].shift(1)
    for sp in spans:
        ew = base.ewm(span=sp, min_periods=2)
        out[f"{col}_ewma_mean_{sp}"] = ew.mean()
        out[f"{col}_ewma_std_{sp}"] = ew.std()
    return out


def add_target_derived(df: pd.DataFrame, target: str = "balance") -> pd.DataFrame:
    """Производные от таргета: первая разность и отклонение от недельной средней.

    Идея: модели проще ловить смену режима через дельты, чем через уровни.
    Всё на shift(1)/прошлых лагах -> без утечки.
    """
    out = df.copy()
    prev = out[target].shift(1)
    out[f"{target}_diff_1"] = prev - out[target].shift(2)
    roll5 = out[target].shift(1).rolling(5, min_periods=2).mean()
    out[f"{target}_dev_roll5"] = prev - roll5
    return out


def add_calendar(
    df: pd.DataFrame,
    date_col: str = "date",
    salary_days=(5, 20),
) -> pd.DataFrame:
    """Календарные признаки.

    День недели/месяца и неделя месяца дают модели сезонный профиль:
    у ликвидности выраженные внутринедельные и внутримесячные паттерны
    (особенно вблизи налоговых дат — их флаги приходят из exog).

    is_salary_day: дни массовых зарплатных выплат (по умолчанию 5 и 20 числа)
    влияют на потоки; флаг дешёвый и в работах-эталонах попадал в признаки.
    Если P1 отдаёт такой флаг через exog — этот столбец можно отключить
    (salary_days=()) во избежание дубля.
    """
    out = df.copy()
    d = pd.to_datetime(out[date_col])
    out["dow"] = d.dt.weekday
    out["dom"] = d.dt.day
    out["wom"] = (d.dt.day - 1) // 7 + 1
    out["month"] = d.dt.month
    out["quarter"] = d.dt.quarter
    out["is_month_end"] = d.dt.is_month_end.astype(int)
    out["is_month_start"] = d.dt.is_month_start.astype(int)
    out["is_quarter_end"] = d.dt.is_quarter_end.astype(int)
    if salary_days:
        out["is_salary_day"] = d.dt.day.isin(salary_days).astype(int)
    return out


def add_fourier(
    df: pd.DataFrame,
    date_col: str = "date",
    weekly_period: int = 5,
    monthly_period: int = 21,
) -> pd.DataFrame:
    """Фурье-гармоники сезонности.

    Почему не по календарным дням: ряд — это бизнес-дни (выходные выброшены),
    поэтому неделя ≈ 5 наблюдений, месяц ≈ 21. Используем позиционный индекс
    бизнес-дня, а не разницу дат, иначе пропуски выходных ломают фазу.
    """
    out = df.copy().reset_index(drop=True)
    t = np.arange(len(out), dtype=float)
    for p, name in ((weekly_period, "w"), (monthly_period, "m")):
        out[f"sin_{name}"] = np.sin(2 * np.pi * t / p)
        out[f"cos_{name}"] = np.cos(2 * np.pi * t / p)
    return out


def add_exog_features(df: pd.DataFrame, key_rate_col: str = "key_rate") -> pd.DataFrame:
    """Преобразует СЫРЫЕ УРОВНИ экзогенов от P1 (build_exog) в модельные признаки.

    P1 отдаёт уровни: key_rate, ruonia, moex_close, usdrub (+ налоговый
    календарь). Превратить их в доходности/спреды/разности — задача feature
    engineering, и здесь же расставляем лаги по наблюдаемости:

      * key_rate — директивная ставка, известна заранее: уровень + Δ за 5 дней,
        БЕЗ лага.
      * ruonia — публикуется с задержкой ~1 день: считаем спред (ruonia−key)
        и лагируем на 1; сырой уровень отбрасываем.
      * moex_close — известен лишь к концу дня t: лог-доходность с лагом 1.
      * usdrub — берём дневную доходность с лагом 1 (консервативно).

    Если каких-то колонок нет — соответствующие признаки просто не создаются.
    """
    out = df.copy()
    if key_rate_col in out.columns:
        out["key_rate_diff5"] = out[key_rate_col] - out[key_rate_col].shift(5)
    if "ruonia" in out.columns and key_rate_col in out.columns:
        out["ruonia_spread_lag1"] = (out["ruonia"] - out[key_rate_col]).shift(1)
    if "moex_close" in out.columns:
        out["moex_ret_lag1"] = np.log(out["moex_close"]).diff().shift(1)
    if "usdrub" in out.columns:
        out["usdrub_ret_lag1"] = out["usdrub"].pct_change().shift(1)
    # сырые рыночные уровни нельзя использовать на день t в конце t-1 -> убираем
    out = out.drop(columns=[c for c in ("ruonia", "moex_close", "usdrub")
                            if c in out.columns])
    return out


def lag_market_exog(
    df: pd.DataFrame, cols=MARKET_EXOG, drop_contemporaneous: bool = True
) -> pd.DataFrame:
    """Лаг произвольных рыночных экзогенов на 1 день (защита от lookahead).

    Универсальная страховка для колонок, не покрытых add_exog_features
    (например, если P1 добавит inflation/индексы под именами из MARKET_EXOG).
    """
    out = df.copy()
    present = [c for c in cols if c in out.columns]
    for c in present:
        out[f"{c}_lag1"] = out[c].shift(1)
    if drop_contemporaneous and present:
        out = out.drop(columns=present)
    return out


# ---------------------------------------------------------------------------
# Оркестратор
# ---------------------------------------------------------------------------
def build_features(
    df: pd.DataFrame,
    exog: pd.DataFrame | None = None,
    target: str = "balance",
    lags=DEFAULT_LAGS,
    windows=DEFAULT_WINDOWS,
    lag_exog: bool = True,
    return_cols: bool = False,
    dropna: bool = True,
):
    """Собирает полную матрицу признаков.

    Параметры
    ---------
    df : DataFrame c [date, balance, income, outcome] (+ возможно экзогены).
    exog : отдельный DataFrame экзогенов с колонкой `date` (если P1 отдаёт их
           отдельно). Если экзогены уже в df, можно не передавать.
    lag_exog : лагировать ли рыночные экзогены на 1 день (рекомендуется True).
    return_cols : вернуть (df, feature_cols) вместо одного df.

    Возвращает
    ----------
    DataFrame с фичами (+ список колонок-фич, если return_cols=True).
    """
    out = df.copy().sort_values("date").reset_index(drop=True)

    # 1. авторегрессионные признаки таргета и его компонент
    out = add_lags(out, target, lags)
    for comp in ("income", "outcome"):
        if comp in out.columns:
            out = add_lags(out, comp, lags)
    # rolling + EWMA по сальдо и его компонентам (в эталонах моментные
    # статистики Income/Outcome были самыми информативными)
    for col in DEFAULT_ROLL_COLS:
        if col in out.columns:
            out = add_rolling(out, col, windows)
            out = add_ewma(out, col)
    out = add_target_derived(out, target)

    # 2. календарь и сезонность
    out = add_calendar(out)
    out = add_fourier(out)

    # 3. внешние факторы
    if exog is not None:
        out = out.merge(exog, on="date", how="left")
    # директивные/календарные экзогены тянем ffill ДО построения разностей
    raw_exog = [c for c in ("key_rate", "ruonia", "moex_close", "usdrub",
                            "is_tax_day", "is_holiday", "days_to_next_tax",
                            "days_since_last_tax") if c in out.columns]
    if raw_exog:
        out[raw_exog] = out[raw_exog].ffill()
    # уровни P1 -> модельные признаки (доходности/спреды/разности) с лагами
    out = add_exog_features(out)
    # прочие рыночные уровни (если P1 их добавит) — лагируем
    if lag_exog:
        out = lag_market_exog(out)

    # 4. чистка
    if dropna:
        out = out.dropna().reset_index(drop=True)

    if return_cols:
        non_features = {"date", target, "income", "outcome"}
        feature_cols = [c for c in out.columns if c not in non_features]
        return out, feature_cols
    return out


def feature_columns(df: pd.DataFrame, target: str = "balance") -> list[str]:
    """Утилита: список колонок-признаков (всё, кроме служебных/таргета)."""
    non_features = {"date", target, "income", "outcome"}
    return [c for c in df.columns if c not in non_features]
