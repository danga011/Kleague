import argparse
from pathlib import Path
from typing import Iterable, Optional, Set, Tuple

import pandas as pd


def _parse_countries_arg(countries_arg: Optional[str]) -> Optional[Set[str]]:
    if not countries_arg:
        return None
    parts = [p.strip().upper() for p in countries_arg.split(",") if p.strip()]
    return set(parts) if parts else None


def build_wvs_city_vectors(
    wvs_path: Path,
    out_path: Path,
    countries: Optional[Set[str]] = None,
    min_n: int = 50,
    chunksize: int = 200_000,
) -> pd.DataFrame:
    """
    Build city/location cultural vectors from WVS Time Series dataset.

    We aggregate WVS derived cultural dimensions per (country, location, survey_year):
      - TradAgg, SurvSAgg (Inglehart–Welzel style indices)

    Keys:
      - COUNTRY_ALPHA (e.g., KOR)
      - E179_WVS7LOC (location code; treated as 'city/region id')
      - S020 (survey year)

    Output columns:
      country_alpha, loc_code, year, n, tradagg_mean, survsagg_mean
    """
    usecols = ["COUNTRY_ALPHA", "E179_WVS7LOC", "S020", "TradAgg", "SurvSAgg"]

    # Accumulator: (country, loc, year) -> [trad_sum, surv_sum, n]
    acc: dict[Tuple[str, int, int], list] = {}

    for chunk in pd.read_csv(wvs_path, usecols=usecols, chunksize=chunksize, low_memory=True):
        chunk["COUNTRY_ALPHA"] = chunk["COUNTRY_ALPHA"].astype(str).str.upper()
        if countries is not None:
            chunk = chunk[chunk["COUNTRY_ALPHA"].isin(countries)]
            if chunk.empty:
                continue

        # Normalize numeric columns
        chunk["E179_WVS7LOC"] = pd.to_numeric(chunk["E179_WVS7LOC"], errors="coerce")
        chunk["S020"] = pd.to_numeric(chunk["S020"], errors="coerce")
        chunk["TradAgg"] = pd.to_numeric(chunk["TradAgg"], errors="coerce")
        chunk["SurvSAgg"] = pd.to_numeric(chunk["SurvSAgg"], errors="coerce")

        # Filter valid location + year + indices
        chunk = chunk[
            chunk["E179_WVS7LOC"].notna()
            & (chunk["E179_WVS7LOC"] > 0)  # exclude -1/-2/-4 etc
            & chunk["S020"].notna()
            & (chunk["S020"] > 0)
            & chunk["TradAgg"].notna()
            & chunk["SurvSAgg"].notna()
        ]
        if chunk.empty:
            continue

        chunk["loc_code"] = chunk["E179_WVS7LOC"].astype(int)
        chunk["year"] = chunk["S020"].astype(int)

        grouped = (
            chunk.groupby(["COUNTRY_ALPHA", "loc_code", "year"], sort=False)[["TradAgg", "SurvSAgg"]]
            .agg(trad_sum=("TradAgg", "sum"), surv_sum=("SurvSAgg", "sum"), n=("TradAgg", "size"))
            .reset_index()
        )

        for _, row in grouped.iterrows():
            key = (str(row["COUNTRY_ALPHA"]), int(row["loc_code"]), int(row["year"]))
            if key not in acc:
                acc[key] = [0.0, 0.0, 0]
            acc[key][0] += float(row["trad_sum"])
            acc[key][1] += float(row["surv_sum"])
            acc[key][2] += int(row["n"])

    rows = []
    for (country_alpha, loc_code, year), (trad_sum, surv_sum, n) in acc.items():
        if n < min_n:
            continue
        rows.append(
            {
                "country_alpha": country_alpha,
                "loc_code": int(loc_code),
                "year": int(year),
                "n": int(n),
                "tradagg_mean": float(trad_sum) / float(n),
                "survsagg_mean": float(surv_sum) / float(n),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        return df

    df = df.sort_values(["country_alpha", "loc_code", "year"], ascending=[True, True, False]).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build WVS city/location cultural vectors.")
    parser.add_argument(
        "--wvs_path",
        type=str,
        default="data/raw/WVS_Time_Series_1981-2022_csv_v5_0.csv",
        help="Path to WVS Time Series CSV",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="data/processed/wvs_city_vectors.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--countries",
        type=str,
        default=None,
        help="Comma-separated COUNTRY_ALPHA filters (e.g., KOR,JPN,BRA). Default: all",
    )
    parser.add_argument("--min_n", type=int, default=50, help="Minimum respondents per (country,loc,year)")
    parser.add_argument("--chunksize", type=int, default=200000, help="Read chunksize")
    args = parser.parse_args(list(argv) if argv is not None else None)

    wvs_path = Path(args.wvs_path)
    out_path = Path(args.out_path)
    countries = _parse_countries_arg(args.countries)

    df = build_wvs_city_vectors(
        wvs_path=wvs_path,
        out_path=out_path,
        countries=countries,
        min_n=args.min_n,
        chunksize=args.chunksize,
    )
    print(f"Saved {len(df)} city vectors -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


