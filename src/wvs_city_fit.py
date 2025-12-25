import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd


@dataclass(frozen=True)
class CityVector:
    country_alpha: str
    loc_code: int
    year: int
    n: int
    tradagg_mean: float
    survsagg_mean: float


@dataclass(frozen=True)
class CountryVector:
    country_alpha: str
    n: int
    tradagg_mean: float
    survsagg_mean: float


def load_wvs_city_vectors(csv_path: Union[str, Path], latest_only: bool = True) -> pd.DataFrame:
    """
    Load precomputed WVS city vectors from `data/processed/wvs_city_vectors.csv`.

    Expected columns:
      country_alpha, loc_code, year, n, tradagg_mean, survsagg_mean
    """
    df = pd.read_csv(csv_path)
    required = {"country_alpha", "loc_code", "year", "n", "tradagg_mean", "survsagg_mean"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"WVS city vectors file missing columns {sorted(missing)}: {csv_path}")

    df = df.copy()
    df["country_alpha"] = df["country_alpha"].astype(str).str.upper()
    df["loc_code"] = pd.to_numeric(df["loc_code"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["n"] = pd.to_numeric(df["n"], errors="coerce")
    df["tradagg_mean"] = pd.to_numeric(df["tradagg_mean"], errors="coerce")
    df["survsagg_mean"] = pd.to_numeric(df["survsagg_mean"], errors="coerce")

    df = df.dropna(subset=["country_alpha", "loc_code", "year", "tradagg_mean", "survsagg_mean"])
    df["loc_code"] = df["loc_code"].astype(int)
    df["year"] = df["year"].astype(int)
    df["n"] = df["n"].fillna(0).astype(int)

    if latest_only and not df.empty:
        df = df.sort_values(["country_alpha", "loc_code", "year"], ascending=[True, True, False])
        df = df.drop_duplicates(subset=["country_alpha", "loc_code"], keep="first")

    return df.reset_index(drop=True)


def _variance(df: pd.DataFrame, col: str) -> float:
    v = float(df[col].dropna().var(ddof=0)) if col in df.columns else 0.0
    return v if v and v > 0 else 1.0


def build_city_vector_map(df: pd.DataFrame) -> Dict[Tuple[str, int], CityVector]:
    """Convert vectors DataFrame into lookup map keyed by (country_alpha, loc_code)."""
    m: Dict[Tuple[str, int], CityVector] = {}
    for _, r in df.iterrows():
        key = (str(r["country_alpha"]).upper(), int(r["loc_code"]))
        m[key] = CityVector(
            country_alpha=key[0],
            loc_code=key[1],
            year=int(r["year"]),
            n=int(r["n"]),
            tradagg_mean=float(r["tradagg_mean"]),
            survsagg_mean=float(r["survsagg_mean"]),
        )
    return m


def build_country_vector_map(df: pd.DataFrame) -> Dict[str, CountryVector]:
    """
    Build country-level weighted means from city vectors.
    - weights: n (respondents)
    """
    if df.empty:
        return {}

    tmp = df[["country_alpha", "n", "tradagg_mean", "survsagg_mean"]].copy()
    tmp["country_alpha"] = tmp["country_alpha"].astype(str).str.upper()
    tmp["n"] = pd.to_numeric(tmp["n"], errors="coerce").fillna(0).clip(lower=0).astype(int)

    sum_n = tmp.groupby("country_alpha")["n"].sum()
    trad_sum = (tmp["tradagg_mean"] * tmp["n"]).groupby(tmp["country_alpha"]).sum()
    surv_sum = (tmp["survsagg_mean"] * tmp["n"]).groupby(tmp["country_alpha"]).sum()

    country_map: Dict[str, CountryVector] = {}
    for country in sum_n.index:
        n = int(sum_n.loc[country])
        if n > 0:
            trad_mean = float(trad_sum.loc[country]) / float(n)
            surv_mean = float(surv_sum.loc[country]) / float(n)
        else:
            subset = tmp[tmp["country_alpha"] == country]
            trad_mean = float(subset["tradagg_mean"].mean()) if not subset.empty else float("nan")
            surv_mean = float(subset["survsagg_mean"].mean()) if not subset.empty else float("nan")
        country_map[str(country).upper()] = CountryVector(
            country_alpha=str(country).upper(),
            n=n,
            tradagg_mean=trad_mean,
            survsagg_mean=surv_mean,
        )
    return country_map


def build_global_vector(df: pd.DataFrame) -> Optional[CountryVector]:
    """Build a global weighted mean vector (fallback of last resort)."""
    if df.empty:
        return None

    tmp = df[["n", "tradagg_mean", "survsagg_mean"]].copy()
    tmp["n"] = pd.to_numeric(tmp["n"], errors="coerce").fillna(0).clip(lower=0).astype(int)
    n = int(tmp["n"].sum())
    if n <= 0:
        return None

    trad_mean = float((tmp["tradagg_mean"] * tmp["n"]).sum()) / float(n)
    surv_mean = float((tmp["survsagg_mean"] * tmp["n"]).sum()) / float(n)
    return CountryVector(country_alpha="GLOBAL", n=n, tradagg_mean=trad_mean, survsagg_mean=surv_mean)


def wvs_ksi_distance(
    a: CityVector,
    b: CityVector,
    var_trad: float,
    var_surv: float,
) -> float:
    """
    KSI-like normalized squared distance using 2 WVS axes:
      ksi = (1/2) * [ (ΔTradAgg^2 / VarTradAgg) + (ΔSurvSAgg^2 / VarSurvSAgg) ]
    """
    dt = a.tradagg_mean - b.tradagg_mean
    ds = a.survsagg_mean - b.survsagg_mean
    return 0.5 * ((dt * dt) / (var_trad or 1.0) + (ds * ds) / (var_surv or 1.0))


def wvs_c_fit_from_distance(dist: float, method: str = "inv1p") -> float:
    """
    Convert distance (>=0) to fit score (0~1).
      - inv1p: 1/(1+dist)
      - exp: exp(-dist)
    """
    if dist is None or not isinstance(dist, (int, float)) or math.isnan(dist):
        return float("nan")
    if dist < 0:
        dist = 0.0
    if method == "exp":
        return float(math.exp(-dist))
    return float(1.0 / (1.0 + dist))


def compute_wvs_city_c_fit(
    city_map: Dict[Tuple[str, int], CityVector],
    country_map: Optional[Dict[str, CountryVector]],
    global_vector: Optional[CountryVector],
    var_trad: float,
    var_surv: float,
    home_country_alpha: str,
    home_loc_code: int,
    host_country_alpha: str,
    host_loc_code: int,
    unknown_default: float = 0.85,
    method: str = "inv1p",
    reliability_k: float = 200.0,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute C-Fit from WVS city cultural distance between two locations.

    Improvements:
    - If city vector is missing, fall back to country-level weighted mean, then global mean.
    - If city vector exists but has limited respondents (n), shrink the city vector towards
      its country (or global) vector before computing distance:
        v_eff = w * v_city + (1-w) * v_country, where w = n / (n + reliability_k)
      This stabilizes the distance without artificially inflating the score.
    """

    def _country_or_global(c: str) -> Tuple[Optional[float], Optional[float], int, str]:
        if country_map is not None:
            v_country = country_map.get(c)
            if v_country is not None:
                return v_country.tradagg_mean, v_country.survsagg_mean, int(v_country.n), "country"
        if global_vector is not None:
            return global_vector.tradagg_mean, global_vector.survsagg_mean, int(global_vector.n), "global"
        return None, None, 0, "missing"

    def _resolve(country_alpha: str, loc_code: int) -> Tuple[Optional[float], Optional[float], int, Optional[int], str, float, str]:
        """
        Return (trad_eff, surv_eff, n, year, source, blend_w, fallback_source).
        source in {"city","country","global","missing"}.
        """
        c = str(country_alpha).strip().upper()
        try:
            lc = int(loc_code)
        except Exception:
            lc = -1

        v_city = city_map.get((c, lc))
        if v_city is not None:
            fb_trad, fb_surv, fb_n, fb_src = _country_or_global(c)
            # If no fallback exists, use raw city vector without blending.
            if fb_trad is None or fb_surv is None:
                return (
                    v_city.tradagg_mean,
                    v_city.survsagg_mean,
                    int(v_city.n),
                    int(v_city.year),
                    "city",
                    1.0,
                    "missing",
                )
            # Blend city -> country/global based on n
            if reliability_k and reliability_k > 0 and v_city.n and v_city.n > 0:
                w = float(float(v_city.n) / (float(v_city.n) + float(reliability_k)))
            else:
                w = 0.0
            trad_eff = float(w * float(v_city.tradagg_mean) + (1.0 - w) * float(fb_trad))
            surv_eff = float(w * float(v_city.survsagg_mean) + (1.0 - w) * float(fb_surv))
            return trad_eff, surv_eff, int(v_city.n), int(v_city.year), "city", float(w), fb_src

        fb_trad, fb_surv, fb_n, fb_src = _country_or_global(c)
        if fb_trad is None or fb_surv is None:
            return None, None, 0, None, "missing", 0.0, "missing"
        return float(fb_trad), float(fb_surv), int(fb_n), None, fb_src, 1.0, fb_src

    a_trad, a_surv, a_n, a_year, a_src, a_w, a_fb = _resolve(home_country_alpha, home_loc_code)
    b_trad, b_surv, b_n, b_year, b_src, b_w, b_fb = _resolve(host_country_alpha, host_loc_code)

    # If we still cannot resolve both sides, return unknown default and meta.
    if a_trad is None or a_surv is None or b_trad is None or b_surv is None:
        meta = {
            "dist": None,
            "c_fit_final": float(unknown_default),
            "method": method,
            "home_source": a_src,
            "host_source": b_src,
            "home_n": int(a_n),
            "host_n": int(b_n),
            "home_blend_w": float(a_w),
            "host_blend_w": float(b_w),
            "home_fallback_source": a_fb,
            "host_fallback_source": b_fb,
            "home_year": a_year,
            "host_year": b_year,
            "note": "vector_missing",
        }
        return float(unknown_default), meta

    # Distance
    dt = float(a_trad) - float(b_trad)
    ds = float(a_surv) - float(b_surv)
    dist = 0.5 * ((dt * dt) / (var_trad or 1.0) + (ds * ds) / (var_surv or 1.0))
    c_fit_final = float(wvs_c_fit_from_distance(dist, method=method))

    meta = {
        "dist": float(dist),
        "c_fit_final": float(c_fit_final),
        "method": method,
        "home_source": a_src,
        "host_source": b_src,
        "home_n": int(a_n),
        "host_n": int(b_n),
        "home_blend_w": float(a_w),
        "host_blend_w": float(b_w),
        "home_fallback_source": a_fb,
        "host_fallback_source": b_fb,
        "home_year": a_year,
        "host_year": b_year,
        "note": "ok",
    }
    return float(c_fit_final), meta


def load_wvs_city_vectors_and_stats(
    csv_path: Union[str, Path]
) -> tuple[Dict[Tuple[str, int], CityVector], Dict[str, CountryVector], Optional[CountryVector], float, float]:
    """
    Convenience loader:
      - loads vectors (latest-only)
      - builds map
      - builds country/global fallback vectors
      - returns maps and dimension variances
    """
    df = load_wvs_city_vectors(csv_path, latest_only=True)
    var_trad = _variance(df, "tradagg_mean")
    var_surv = _variance(df, "survsagg_mean")
    city_map = build_city_vector_map(df)
    country_map = build_country_vector_map(df)
    global_vector = build_global_vector(df)
    return city_map, country_map, global_vector, var_trad, var_surv


