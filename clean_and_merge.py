"""
================================================================================
Phase 3: Dataset Standardization and Integration
================================================================================
Data Science Final Project Pipeline - Phase 3

Objective:
    - Consume all 79 aggregated household CSVs generated in Phase 2.
    - Parse location identifiers, survey formats, and temporal data from filenames.
    - Normalize location terminology to ensure consistency (e.g., merging "GALKAYO" and "GAALKACYO").
    - Combine all individual CSVs into one comprehensive dataset.
    - Output File: output/merged/smart_mortality_merged.csv
      (This single file will serve as the input for machine learning models.)
================================================================================
"""

# ==============================================================================
# 1) Imports
# ==============================================================================

from __future__ import annotations

import csv
from pathlib import Path

# ==============================================================================
# 2) Configuration
# ==============================================================================

WORKSPACE_ROOT = Path(__file__).parent.resolve()

# Input: processed household files from Stage 3
INPUT_ROOT = WORKSPACE_ROOT / "output" / "processed_households"
ADMIN_IND = INPUT_ROOT / "admin2_surveys" / "individual_surveys"
ADMIN_AGG = INPUT_ROOT / "admin2_surveys" / "aggregate_surveys"
LHZ_IND   = INPUT_ROOT / "lhz_surveys"    / "individual_surveys"

# Output: single merged file
OUTPUT_DIR = WORKSPACE_ROOT / "output" / "merged"
OUTPUT_FILE = OUTPUT_DIR / "smart_mortality_merged.csv"

# Final merged columns (adds location, survey_type to the household schema)
MERGED_COLUMNS = [
    "location", "survey_type", "month", "year", "recall_period",
    "HH", "Cluster", "Team",
    "Total", "Births", "Deaths", "Joined", "Left",
    "Total_U5", "Births_U5", "Deaths_U5", "Joined_U5", "Left_U5",
    "Person_Time", "Person_Time_U5"
]

# ==============================================================================
# 3) Location Name Standardization
# ==============================================================================
#
# Many survey filenames use variant spellings for the same place.
# This dictionary maps every raw filename variant to a single clean name.

LOCATION_ALIASES: dict[str, str] = {
    # --- Duplicates / spelling variants ---
    "GALKAYO":                  "Gaalkacyo",
    "GAALKACYO":                "Gaalkacyo",
    "GAROWE":                   "Garoowe",
    "GAROOWE":                  "Garoowe",
    "DHUSAMAREEB":              "Dhuusamarreeb",
    "DHUUSAMARREEB":            "Dhuusamarreeb",
    "COASTAL_DEEH":             "Coastal Deeh",
    "COASTALDEEH":              "Coastal Deeh",
    "BELETWEYNE":               "Beletweyne",
    "BELETWEYNE_2":             "Beletweyne",
    "BAY_AGROPASTAGROPASTORAL": "Bay Agropastoral",
    "06_ADDUN":                 "Addun",
    "11_BARI":                  "Bari",
    "ELBARDE_JAN":              "Elbarde",
    "HUDUR_JAN":                "Hudur",
    "BURCO_IDPS":               "Burco",

    # --- Already clean (single spelling) ---
    "ADADO":            "Adado",
    "ADDUN":            "Addun",
    "BAKOOL":           "Bakool",
    "BANADIR":          "Banadir",
    "BAY":              "Bay",
    "BAYDHABA":         "Baydhaba",
    "BERBERA":          "Berbera",
    "BOSSASO":          "Bossaso",
    "BURCO":            "Burco",
    "DHOBLEY":          "Dhobley",
    "DOOLOW":          "Doolow",
    "EASTGOLIS":        "East Golis",
    "HARGEYSA":         "Hargeysa",
    "KISMAYO":          "Kismayo",
    "MOGADISHU":        "Mogadishu",
    "NECHAWD":          "Nechawd",
    "ODWEYNE":          "Odweyne",
    "QARDHO":           "Qardho",
    "SHABELLE_AP":      "Shabelle Agropastoral",
    "SHABELLE_RIVER":   "Shabelle River",
    "UNKNOWN":          "Pump Riverine",
}


def clean_location(raw: str) -> str:
    """Map a raw filename location to its standardized name."""
    return LOCATION_ALIASES.get(raw, raw.title())


# ==============================================================================
# 4) Filename Parsing
# ==============================================================================


def parse_filename(filename: str) -> tuple[str, str]:
    """
    Extract the location name and survey type from a processed CSV filename.

    Filename pattern:
        SOM_YYYY_MM_LOCATION_type_subtype[_N].csv
        e.g. SOM_2018_02_GALKAYO_admin2_individual.csv
             SOM_2022_04_BELETWEYNE_admin2_individual_2.csv
             SOM_2014_07_06_ADDUN_admin2_aggregate.csv

    Returns:
        (clean_location, survey_type)
        e.g. ("Gaalkacyo", "admin2_individual")
    """
    stem = Path(filename).stem  # Remove .csv
    parts = stem.split("_")

    # Find where the type identifier starts (admin2 or lhz)
    type_start_idx = None
    for i, p in enumerate(parts):
        if p.lower() in ("admin2", "lhz"):
            type_start_idx = i
            break

    if type_start_idx is None:
        return clean_location("UNKNOWN"), "unknown"

    # Location is everything between parts[3] and the type identifier
    # (parts[0]=SOM, [1]=year, [2]=month, [3..type_start]=location)
    loc_parts = parts[3:type_start_idx]
    raw_location = "_".join(loc_parts) if loc_parts else "UNKNOWN"

    # Survey type: e.g. "admin2_individual", "admin2_aggregate", "lhz_individual"
    # Take only the type and subtype, ignore trailing _2, _3 suffixes
    type_parts = []
    for p in parts[type_start_idx:]:
        if p.isdigit():
            break  # trailing number suffix like _2
        type_parts.append(p)
    survey_type = "_".join(type_parts)

    return clean_location(raw_location), survey_type


# ==============================================================================
# 5) Merge Logic
# ==============================================================================


def merge_all() -> int:
    """
    Read all processed household CSVs, add location + survey_type columns,
    and write a single merged CSV.

    Returns the total number of data rows written.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[list[str]] = []
    sources = [
        (ADMIN_IND, "admin2_individual"),
        (ADMIN_AGG, "admin2_aggregate"),
        (LHZ_IND,   "lhz_individual"),
    ]

    for folder, expected_type in sources:
        if not folder.exists():
            continue
        for csv_path in sorted(folder.glob("*.csv")):
            location, survey_type = parse_filename(csv_path.name)

            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)  # Skip header row
                if not header:
                    continue

                col_idx = {col.lower(): idx for idx, col in enumerate(header)}

                for row in reader:
                    if not row:
                        continue

                    # Safe value retrieval
                    def g(name: str) -> str:
                        idx = col_idx.get(name.lower())
                        if idx is not None and idx < len(row):
                            return row[idx].strip()
                        return ""

                    all_rows.append([
                        location,
                        survey_type,
                        g("month"),
                        g("year"),
                        g("recall_period"),
                        g("hh"),
                        g("cluster"),
                        g("team"),
                        g("total"),
                        g("births"),
                        g("deaths"),
                        g("joined"),
                        g("left"),
                        g("total_u5"),
                        g("births_u5"),
                        g("deaths_u5"),
                        g("joined_u5"),
                        g("left_u5"),
                        g("person_time"),
                        g("person_time_u5"),
                    ])

    # Write merged CSV
    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(MERGED_COLUMNS)
        writer.writerows(all_rows)

    return len(all_rows)


# ==============================================================================
# 6) Main
# ==============================================================================


def main() -> None:
    print("=" * 60)
    print("Phase 3: Dataset Standardization and Integration")
    print("=" * 60)

    total = merge_all()

    print(f"\nSuccessfully integrated rows: {total:,}")
    print(f"Master Dataset saved to: {OUTPUT_FILE}")

    # Count unique locations
    locations = set()
    with open(OUTPUT_FILE, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            locations.add(row["location"])
    print(f"Total Unique Geographies: {len(locations)}")
    for loc in sorted(locations):
        print(f"  * {loc}")

    print("\n[SUCCESS] Phase 3 executed completely.")


if __name__ == "__main__":
    main()
