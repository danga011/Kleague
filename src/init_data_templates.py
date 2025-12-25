from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent


RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "raw_data.csv"
PLAYER_PROFILE_PATH = PROJECT_ROOT / "data" / "raw" / "player_profile.csv"
PLAYER_MINUTES_PATH = PROJECT_ROOT / "data" / "raw" / "player_minutes_by_match.csv"


def init_player_profile(raw_data_path: Path, out_path: Path) -> int:
    """
    Create/refresh `data/raw/player_profile.csv`.
    - Pre-populates player_id/player_name_ko from raw_data.csv
    - Leaves bio fields empty for manual/ETL fill
    - Preserves existing filled values (by player_id)
    """
    if not raw_data_path.exists():
        raise FileNotFoundError(f"raw_data.csv not found: {raw_data_path}")

    raw_df = pd.read_csv(raw_data_path, usecols=["player_id", "player_name_ko"])
    players = (
        raw_df[["player_id", "player_name_ko"]]
        .drop_duplicates()
        .sort_values(["player_id", "player_name_ko"])
        .reset_index(drop=True)
    )

    template_cols = [
        "player_id",
        "player_name_ko",
        "player_name_en_full",
        "nationality",
        "club_name_ko",
        "preferred_foot",
        "height_cm",
        "weight_kg",
        "birth_year",
    ]
    out_df = players.copy()
    for c in template_cols:
        if c not in out_df.columns:
            out_df[c] = ""
    out_df = out_df[template_cols]

    if out_path.exists():
        old = pd.read_csv(out_path)
        if "player_id" in old.columns:
            old = old.copy()
            old["player_id"] = pd.to_numeric(old["player_id"], errors="coerce")
            old = old.dropna(subset=["player_id"]).drop_duplicates(subset=["player_id"])
            old = old.set_index("player_id")
            for c in template_cols:
                if c in {"player_id", "player_name_ko"}:
                    continue
                if c in old.columns:
                    out_df[c] = out_df["player_id"].map(old[c]).fillna(out_df[c])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    return len(out_df)


def init_player_minutes(out_path: Path) -> None:
    """
    Create `data/raw/player_minutes_by_match.csv` if missing.
    This file should be collected from a reliable source (official lineups/minutes).
    """
    if out_path.exists():
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "game_id",
        "player_id",
        "team_name_ko",
        "minutes_played",
        "is_starter",
        "position_name",
        "sub_in_minute",
        "sub_out_minute",
        "source",
        "notes",
    ]
    pd.DataFrame(columns=cols).to_csv(out_path, index=False)


def main() -> int:
    n = init_player_profile(RAW_DATA_PATH, PLAYER_PROFILE_PATH)
    init_player_minutes(PLAYER_MINUTES_PATH)
    print(f"✅ player_profile template ready: {PLAYER_PROFILE_PATH} (rows={n})")
    print(f"✅ player_minutes template ready: {PLAYER_MINUTES_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


