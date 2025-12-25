import csv
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _read_csv_header(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
    # Strip quotes/spaces
    return [h.strip().strip('"').strip("'").lstrip("\ufeff") for h in header if isinstance(h, str)]


def _check_file(path: Path, required_cols: Optional[Iterable[str]] = None) -> Tuple[bool, List[str], List[str]]:
    """
    Returns:
      (ok, missing_cols, header_cols)
    """
    if not path.exists():
        return False, ["<file missing>"], []

    if required_cols is None:
        return True, [], _read_csv_header(path)

    header = _read_csv_header(path)
    header_set = set(header)
    missing = [c for c in required_cols if c not in header_set]
    return len(missing) == 0, missing, header


def main() -> int:
    required_checks = [
        {
            "name": "K리그 이벤트 데이터(raw_data.csv)",
            "path": PROJECT_ROOT / "data" / "raw" / "raw_data.csv",
            "required": [
                "game_id",
                "action_id",
                "period_id",
                "time_seconds",
                "team_id",
                "player_id",
                "type_name",
                "start_x",
                "start_y",
                "end_x",
                "end_y",
                "player_name_ko",
                "team_name_ko",
                "main_position",
            ],
        },
        {
            "name": "경기 메타(match_info.csv)",
            "path": PROJECT_ROOT / "data" / "raw" / "match_info.csv",
            "required": [
                "game_id",
                "game_date",
                "home_team_name_ko",
                "away_team_name_ko",
                "venue",
                "competition_name",
            ],
        },
        {
            "name": "WVS 도시 벡터(집계 결과)",
            "path": PROJECT_ROOT / "data" / "processed" / "wvs_city_vectors.csv",
            "required": ["country_alpha", "loc_code", "year", "n", "tradagg_mean", "survsagg_mean"],
        },
        {
            "name": "선수 성장도시 매핑",
            "path": PROJECT_ROOT / "data" / "processed" / "player_upbringing_city_map.csv",
            "required": ["player_name_ko", "home_country_alpha", "home_loc_code", "home_city_label"],
        },
        {
            "name": "구단(분석도시) 매핑",
            "path": PROJECT_ROOT / "data" / "processed" / "kleague_team_city_map.csv",
            "required": ["team_name_ko", "host_country_alpha", "host_loc_code", "host_city_label"],
        },
    ]

    optional_checks = [
        {
            "name": "WVS 원본(Time Series) - (재생성용, 제출물에 포함하지 않을 수 있음)",
            "path": PROJECT_ROOT / "data" / "raw" / "WVS_Time_Series_1981-2022_csv_v5_0.csv",
            "required": None,  # huge; only existence check
        },
        {
            "name": "선수 프로필(바이오) - 수집 필요",
            "path": PROJECT_ROOT / "data" / "raw" / "player_profile.csv",
            "required": [
                "player_id",
                "player_name_ko",
                "player_name_en_full",
                "nationality",
                "club_name_ko",
                "preferred_foot",
                "height_cm",
                "weight_kg",
                "birth_year",
            ],
        },
        {
            "name": "경기별 출전시간/라인업 - 수집 필요",
            "path": PROJECT_ROOT / "data" / "raw" / "player_minutes_by_match.csv",
            "required": [
                "game_id",
                "player_id",
                "team_name_ko",
                "minutes_played",
                "is_starter",
            ],
        },
    ]

    ok_required = True
    ok_optional = True
    print("=== K-Scout 데이터 점검 (data_audit) ===")
    print(f"- project_root: {PROJECT_ROOT}")
    print("")

    print("## REQUIRED (앱/기본 파이프라인)")
    for item in required_checks:
        path: Path = item["path"]
        ok, missing_cols, header = _check_file(path, item["required"])
        ok_required = ok_required and ok

        status = "OK" if ok else "MISSING"
        print(f"[{status}] {item['name']}")
        print(f"  - path: {path}")
        if not path.exists():
            print("  - reason: file not found")
        else:
            if item["required"] is None:
                print("  - note: huge file; header check skipped")
            else:
                if missing_cols:
                    print(f"  - missing_cols: {missing_cols}")
                else:
                    print("  - columns: ok")
        print("")

    print("## OPTIONAL (모델 개발/리포트 고도화)")
    for item in optional_checks:
        path: Path = item["path"]
        ok, missing_cols, header = _check_file(path, item["required"])
        ok_optional = ok_optional and ok

        status = "OK" if ok else "TODO"
        print(f"[{status}] {item['name']}")
        print(f"  - path: {path}")
        if not path.exists():
            print("  - reason: file not found")
        else:
            if item["required"] is None:
                print("  - note: header check skipped")
            else:
                if missing_cols:
                    print(f"  - missing_cols: {missing_cols}")
                else:
                    print("  - columns: ok")
        print("")

    if not ok_required:
        print("❌ 데이터 점검 실패(REQUIRED): 누락된 파일/컬럼이 있습니다. 위 항목을 보강하세요.")
        return 2

    if not ok_optional:
        print("⚠️ OPTIONAL 데이터가 아직 부족합니다. 모델 개발/리포트 고도화를 위해 수집을 권장합니다.")

    print("✅ 데이터 점검 통과: 핵심 파일/컬럼이 준비되어 있습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


