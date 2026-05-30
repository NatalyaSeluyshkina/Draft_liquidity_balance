"""
selection.py — отбор признаков для прогноза сальдо ликвидности.

Зона ответственности P2 (Feature Engineer), трек 8.

Требование ТЗ: модуль отбора признаков; выбранный метод должен быть
БОЛЕЕ СТАБИЛЬНЫМ относительно альтернатив; сравнение минимум с одним методом
из каждой категории (filter / wrapper / embedded); минимум один метод должен
исследовать НЕЛИНЕЙНУЮ зависимость.

Покрытие категорий:
    filter   : select_correlation (линейный), select_mi (НЕЛИНЕЙНЫЙ)
               select_phik (НЕЛИНЕЙНЫЙ) — офлайн, медленный, не в наборе по умолч.
    embedded : select_lasso (линейный),       select_lgbm (НЕЛИНЕЙНЫЙ)
    wrapper  : select_rfe_rf (НЕЛИНЕЙНЫЙ),     select_boruta (НЕЛИНЕЙНЫЙ)

Стабильность измеряется устойчивостью состава отбора между walk-forward окнами:
    * средний попарный Jaccard — основная, интуитивная мера (насколько
      повторяется выбор признаков во времени);
    * индекс Nogueira et al. (2018) — опциональная проверка, скорректированная
      на случайность и размер набора (на случай, если у методов сильно разное
      число отобранных фич, как у Lasso vs top-k).

Выбирается метод с максимальной стабильностью при сопоставимом качестве
прогноза (качество проверяет P3 на бэктесте).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, mutual_info_regression
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 0


# ===========================================================================
# 1. FILTER-методы
# ===========================================================================
def select_correlation(
    X: pd.DataFrame, y: np.ndarray, k: int = 20, method: str = "spearman"
) -> list[str]:
    """Фильтр по |корреляции| с таргетом (линейная/монотонная зависимость).

    Spearman устойчивее к выбросам, чем Pearson. Берём top-k по модулю.
    """
    Xc = X.fillna(0.0)
    ys = pd.Series(np.asarray(y), index=Xc.index)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr = Xc.apply(lambda col: col.corr(ys, method=method)).abs().fillna(0.0)
    k = min(k, X.shape[1])
    return corr.sort_values(ascending=False).head(k).index.tolist()


def select_mi(X: pd.DataFrame, y: np.ndarray, k: int = 20) -> list[str]:
    """Фильтр по взаимной информации (НЕЛИНЕЙНАЯ зависимость).

    mutual_info_regression ловит произвольные (в т.ч. немонотонные) связи —
    это и есть требуемый нелинейный фильтр. Берём top-k.
    """
    mi = mutual_info_regression(X.fillna(0.0), y, random_state=RANDOM_STATE)
    k = min(k, X.shape[1])
    order = np.argsort(mi)[-k:]
    return X.columns[order].tolist()


def select_phik(X: pd.DataFrame, y: np.ndarray, k: int = 20) -> list[str]:
    """Фильтр по корреляции phik (φk) с таргетом (НЕЛИНЕЙНАЯ зависимость).

    ВНИМАНИЕ: phik медленный (биннинг + оценка значимости), на сотне признаков
    — минуты. Поэтому НЕ входит в SELECTORS по умолчанию и не годится для
    горячего пути авто-пайплайна; используйте офлайн как разовый диагностикум
    или на уже сокращённом наборе. Если пакет недоступен — fallback на MI.

    phik ловит произвольные (в т.ч. нелинейные/немонотонные) связи и смешанные
    типы; в работах-эталонах по качеству обходил обычную MI.
    """
    k = min(k, X.shape[1])
    try:
        from phik import phik_from_array

        ys = pd.Series(np.asarray(y))
        scores = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for c in X.columns:
                try:
                    scores[c] = phik_from_array(X[c].fillna(0.0), ys)
                except Exception:
                    scores[c] = 0.0
        s = pd.Series(scores).fillna(0.0).sort_values(ascending=False)
        return s.head(k).index.tolist()
    except Exception:
        return select_mi(X, y, k=k)


# ===========================================================================
# 2. EMBEDDED-методы
# ===========================================================================
def select_lasso(X: pd.DataFrame, y: np.ndarray, alpha: float | None = None) -> list[str]:
    """Embedded L1-отбор (линейный).

    КРИТИЧНО: Lasso штрафует коэффициенты, поэтому признаки ОБЯЗАНЫ быть
    отмасштабированы — иначе отбор смещён к фичам с большим разбросом.
    alpha подбирается по CV (LassoCV), если не задан явно.
    Отбираем признаки с ненулевым коэффициентом.
    """
    Xf = X.fillna(0.0).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xf)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if alpha is None:
            model = LassoCV(cv=3, max_iter=20000, random_state=RANDOM_STATE).fit(Xs, y)
        else:
            from sklearn.linear_model import Lasso
            model = Lasso(alpha=alpha, max_iter=20000).fit(Xs, y)
    mask = np.abs(model.coef_) > 1e-8
    chosen = X.columns[mask].tolist()
    # страховка: если регуляризация занулила всё, вернуть top-1 по |coef|
    if not chosen:
        chosen = [X.columns[int(np.argmax(np.abs(model.coef_)))]]
    return chosen


def select_lgbm(X: pd.DataFrame, y: np.ndarray, k: int = 20) -> list[str]:
    """Embedded отбор по важности LightGBM gain (НЕЛИНЕЙНЫЙ).

    Если lightgbm недоступен — мягко падаем на важности RandomForest,
    чтобы модуль не ломался в чужом окружении.
    """
    k = min(k, X.shape[1])
    Xf = X.fillna(0.0)
    try:
        import lightgbm as lgb

        model = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            random_state=RANDOM_STATE, verbose=-1,
        ).fit(Xf, y)
        imp = model.feature_importances_
    except Exception:
        model = RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ).fit(Xf, y)
        imp = model.feature_importances_
    order = np.argsort(imp)[-k:]
    return X.columns[order].tolist()


# ===========================================================================
# 3. WRAPPER-методы
# ===========================================================================
def select_rfe_rf(X: pd.DataFrame, y: np.ndarray, k: int = 20, step: float = 0.2) -> list[str]:
    """Wrapper: рекурсивное исключение признаков с RandomForest (НЕЛИНЕЙНЫЙ).

    На каждой итерации обучается RF, отбрасывается доля наименее важных фич.
    Это «оберточный» метод: важность признака определяется через качество
    самой модели, а не отдельной статистики. Быстрее и стабильнее Boruta,
    поэтому используется как основной wrapper.
    """
    k = min(k, X.shape[1])
    est = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rfe = RFE(est, n_features_to_select=k, step=step).fit(X.fillna(0.0).values, y)
    return X.columns[rfe.support_].tolist()


def select_boruta(X: pd.DataFrame, y: np.ndarray, max_iter: int = 50) -> list[str]:
    """Wrapper: Boruta поверх RandomForest (НЕЛИНЕЙНЫЙ).

    Boruta сравнивает важность каждой фичи с «теневыми» (перемешанными)
    копиями и оставляет лишь статистически значимо лучшие. Если пакет boruta
    недоступен/несовместим — fallback на RFE-RF (тот же класс wrapper).
    """
    try:
        from boruta import BorutaPy

        rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            b = BorutaPy(rf, n_estimators="auto", max_iter=max_iter,
                         random_state=RANDOM_STATE, verbose=0)
            b.fit(X.fillna(0.0).values, np.asarray(y))
        chosen = X.columns[b.support_].tolist()
        return chosen if chosen else X.columns[b.support_weak_].tolist()
    except Exception:
        return select_rfe_rf(X, y)


# Реестр методов для рутинного сравнения (быстрые: фильтры + embedded + лёгкий
# RFE). Boruta и phik сюда НЕ входят — они корректны, но дороги для авто-цикла
# и вызываются офлайн при желании (см. select_boruta / select_phik).
SELECTORS = {
    "Correlation (filter, lin)": (select_correlation, "filter", "linear"),
    "MutualInfo (filter, nonlin)": (select_mi, "filter", "nonlinear"),
    "Lasso (embedded, lin)": (select_lasso, "embedded", "linear"),
    "LightGBM (embedded, nonlin)": (select_lgbm, "embedded", "nonlinear"),
    "RFE-RF (wrapper, nonlin)": (select_rfe_rf, "wrapper", "nonlinear"),
}


# ===========================================================================
# 4. Метрики стабильности
# ===========================================================================
def _to_mask_matrix(sets: list[set], all_features: list[str]) -> np.ndarray:
    idx = {f: i for i, f in enumerate(all_features)}
    Z = np.zeros((len(sets), len(all_features)), dtype=int)
    for r, s in enumerate(sets):
        for f in s:
            Z[r, idx[f]] = 1
    return Z


def nogueira_stability(sets: list[set], all_features: list[str]) -> float:
    """Индекс стабильности Nogueira et al. (JMLR 2018).

    1.0 — идентичные наборы; ~0 — отбор не лучше случайного. Корректирует на
    число признаков и размер наборов, поэтому сопоставим между методами,
    отбирающими разное число фич (Lasso vs top-k).
    """
    Z = _to_mask_matrix(sets, all_features)
    M, d = Z.shape
    if M < 2:
        return 1.0
    pbar = Z.mean(axis=0)
    # несмещённая дисперсия частоты по каждой фиче
    s2 = Z.var(axis=0, ddof=1)
    kbar = Z.sum() / M
    denom = (kbar / d) * (1 - kbar / d)
    if denom <= 0:
        return 1.0
    return float(1 - s2.mean() / denom)


def jaccard_stability(
    selector_fn,
    X: pd.DataFrame,
    y: np.ndarray,
    n_splits: int = 5,
    return_sets: bool = False,
    **kwargs,
):
    """Средний попарный Jaccard наборов, отобранных в разных walk-forward окнах.

    Уважает временной порядок (TimeSeriesSplit), без перемешивания.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    yv = np.asarray(y)
    sets: list[set] = []
    for tr, _ in tscv.split(X):
        sets.append(set(selector_fn(X.iloc[tr], yv[tr], **kwargs)))
    pairs = [(a, b) for i, a in enumerate(sets) for b in sets[i + 1:]]
    j = (np.mean([len(a & b) / max(len(a | b), 1) for a, b in pairs])
         if pairs else 1.0)
    return (float(j), sets) if return_sets else float(j)


# ===========================================================================
# 5. Сравнение методов и выбор победителя
# ===========================================================================
def compare_selectors(
    X: pd.DataFrame,
    y: np.ndarray,
    selectors: dict | None = None,
    n_splits: int = 5,
    **kwargs,
) -> pd.DataFrame:
    """Считает стабильность всех методов и возвращает отсортированную таблицу.

    Колонки: category, kind, n_selected (медиана по окнам), jaccard, nogueira.

    Основной критерий — средний попарный Jaccard наборов, отобранных в разных
    walk-forward окнах (узнаваемая, интуитивная мера: насколько повторяется
    выбор фич во времени). Nogueira добавлен как скорректированная на
    случайность проверка устойчивости вывода. Чем выше — тем стабильнее метод.
    """
    selectors = selectors or SELECTORS
    all_features = X.columns.tolist()
    rows = []
    for name, (fn, category, kind) in selectors.items():
        try:
            j, sets = jaccard_stability(fn, X, y, n_splits=n_splits, return_sets=True)
            nog = nogueira_stability(sets, all_features)
            n_sel = int(np.median([len(s) for s in sets]))
        except Exception as e:  # метод не должен ронять весь отчёт
            j, nog, n_sel = np.nan, np.nan, -1
            warnings.warn(f"{name} failed: {e}")
        rows.append({"method": name, "category": category, "kind": kind,
                     "n_selected": n_sel, "jaccard": j, "nogueira": nog})
    return (pd.DataFrame(rows)
            .set_index("method")
            .sort_values("jaccard", ascending=False))


def choose_best_selector(comparison: pd.DataFrame) -> str:
    """Имя метода-победителя по Jaccard (тай-брейк по Nogueira).

    Если у кандидатов сопоставимое качество прогноза (это проверяет P3 на
    бэктесте), выбираем самый воспроизводимый по составу набор — это и есть
    «более стабильный относительно альтернатив» из ТЗ.
    """
    valid = comparison.dropna(subset=["jaccard"])
    return valid.sort_values(["jaccard", "nogueira"], ascending=False).index[0]


def top_features_report(
    X: pd.DataFrame, y: np.ndarray, k: int = 10, include_phik: bool = False
) -> pd.DataFrame:
    """Топ-k факторов по каждой score-метрике фильтров с их значениями.

    Формат повторяет то, как победившие команды показывали отбор на защите
    (список «фактор: score» по каждому фильтрационному методу). Удобно для
    слайда: видно, какие признаки методы считают важными и где согласие.
    """
    Xf = X.fillna(0.0)
    yv = np.asarray(y)

    mi = pd.Series(mutual_info_regression(Xf, yv, random_state=RANDOM_STATE),
                   index=X.columns)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr = Xf.apply(lambda c: c.corr(pd.Series(yv, index=Xf.index),
                                         method="spearman")).abs()
    cols = {"mutual_info": mi, "abs_spearman": corr.fillna(0.0)}

    if include_phik:  # медленно — только по явному запросу/на малом наборе
        try:
            from phik import phik_from_array
            ys = pd.Series(yv)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cols["phik"] = pd.Series(
                    {c: phik_from_array(Xf[c], ys) for c in Xf.columns}
                ).fillna(0.0)
        except Exception:
            pass

    table = pd.DataFrame(cols)
    table["mean_rank"] = table.rank(ascending=False).mean(axis=1)
    return table.sort_values("mean_rank").head(k).round(3)


# ===========================================================================
# 6. sklearn-совместимый трансформер для Pipeline(features → selector → model)
# ===========================================================================
from sklearn.base import BaseEstimator, TransformerMixin  # noqa: E402


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Обёртка выбранного метода отбора под sklearn.Pipeline.

    Метод фиксирует набор признаков на train (fit) и применяет его на любых
    данных (transform). Это позволяет P3 собрать единый
    Pipeline(FeatureSelector -> model) и переобучать его без ручных правок.

    Пример:
        sel = FeatureSelector(method="rfe_rf", k=20)
        pipe = Pipeline([("sel", sel), ("model", LGBMRegressor())])
    """

    _METHODS = {
        "correlation": select_correlation,
        "mi": select_mi,
        "phik": select_phik,
        "lasso": select_lasso,
        "lgbm": select_lgbm,
        "rfe_rf": select_rfe_rf,
        "boruta": select_boruta,
    }

    def __init__(self, method: str = "rfe_rf", k: int = 20):
        self.method = method
        self.k = k

    def fit(self, X, y):
        X = pd.DataFrame(X).reset_index(drop=True)
        fn = self._METHODS[self.method]
        kwargs = {"k": self.k} if self.method in (
            "correlation", "mi", "phik", "lgbm", "rfe_rf") else {}
        self.selected_ = fn(X, np.asarray(y), **kwargs)
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def transform(self, X):
        return pd.DataFrame(X)[self.selected_].values

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_)
