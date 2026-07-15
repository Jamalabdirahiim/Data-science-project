"""
================================================================================
Phase 2: Household Summary Statistics & Person-Time Calculation
================================================================================
Data Science Final Project Pipeline - Phase 2 (Reference Part 1, Phase 3 of PDF)

Objective:
    - Ingest the cleaned, isolated mortality CSVs produced in Phase 1.
    - Transform individual-level records (with member columns P1-P20 on one row) 
      into unified household-level statistics.
    - Process aggregate-level datasets (already formatted per household) by simply
      aligning their column structure.
    - Calculate the 'person-time at risk' for the entire household, as well as 
      specifically for the Under-5 (U5) cohort, utilizing the dynamically parsed recall period.
    - Save the resulting household-level datasets to the intermediate directory:
      output/processed_households/

Person-Time Logic:
    Person-time represents the cumulative time individuals were at risk of mortality
    during the recall window.
    We apply the mid-interval assumption (assuming individuals who arrived, departed, 
    were born, or died during the period were present for exactly half the duration):

    Total Person-Time = Recall Window * [Final Members + 0.5 * (Deaths + Departures - Births - Arrivals)]
    U5 Person-Time    = Recall Window * [Final U5 Members + 0.5 * (U5 Deaths + U5 Departures - U5 Births - U5 Arrivals)]
================================================================================
"""

# ==============================================================================
# 1) Imports
# ==============================================================================

from __future__ import annotations

import csv
import re
from pathlib import Path

# ==============================================================================
# 2) Configuration
# ==============================================================================

WORKSPACE_ROOT = Path(__file__).parent.resolve()

# Input directories from Stage 2
INPUT_ROOT = WORKSPACE_ROOT / "output"
ADMIN_IND_IN = INPUT_ROOT / "admin2_surveys" / "individual_surveys"
ADMIN_AGG_IN = INPUT_ROOT / "admin2_surveys" / "aggregate_surveys"
LHZ_IND_IN   = INPUT_ROOT / "lhz_surveys"    / "individual_surveys"

# Output directories for Stage 3 (Intermediate)
OUTPUT_ROOT = WORKSPACE_ROOT / "output" / "processed_households"
ADMIN_IND_OUT = OUTPUT_ROOT / "admin2_surveys" / "individual_surveys"
ADMIN_AGG_OUT = OUTPUT_ROOT / "admin2_surveys" / "aggregate_surveys"
LHZ_IND_OUT   = OUTPUT_ROOT / "lhz_surveys"    / "individual_surveys"

ALL_OUTPUT_DIRS = (ADMIN_IND_OUT, ADMIN_AGG_OUT, LHZ_IND_OUT)

# Standardized columns for the final household-level dataset
HH_COLUMNS = [
    "month", "year", "recall_period", "HH", "Cluster", "Team",
    "Total", "Births", "Deaths", "Joined", "Left",
    "Total_U5", "Births_U5", "Deaths_U5", "Joined_U5", "Left_U5",
    "Person_Time", "Person_Time_U5"
]

# ==============================================================================
# 3) Helper Functions & Directory Management
# ==============================================================================


def ensure_dirs() -> None:
    """Create all intermediate directories if they do not exist."""
    for p in ALL_OUTPUT_DIRS:
        p.mkdir(parents=True, exist_ok=True)


def clear_processed_outputs() -> None:
    """Clear previously processed household files before starting."""
    for folder in ALL_OUTPUT_DIRS:
        for f in folder.glob("*.csv"):
            try:
                f.unlink()
            except OSError:
                pass


def get_val(row: list[str], col_name: str, col_idx: dict[str, int]) -> str:
    """Safely retrieve a cell value by column name, guarding against out-of-bounds index."""
    idx = col_idx.get(col_name.lower())
    if idx is not None and idx < len(row):
        return row[idx].strip()
    return ""


# ==============================================================================
# 4) Processing Logic for Individual Surveys
# ==============================================================================


def process_individual_file(in_path: Path, out_path: Path) -> None:
    """
    Reads an individual-level survey CSV (one row per household with columns
    P1_sex, P1_age... P20_location), aggregates member data, calculates
    person-time, and writes a clean household-level CSV.
    """
    with open(in_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return

        # Map column header to index
        col_idx = {col.lower(): idx for idx, col in enumerate(header)}

        # Find the maximum index of P (e.g. P1 to P20) dynamically
        p_indices = set()
        for col in header:
            m = re.match(r"^P(\d+)_", col, re.IGNORECASE)
            if m:
                p_indices.add(int(m.group(1)))
        max_p = max(p_indices) if p_indices else 0

        rows_to_write = []

        for row in reader:
            if not row:
                continue

            # Skip embedded header rows (from concatenated survey sections)
            if get_val(row, "month", col_idx).lower() == "month":
                continue

            # Basic metadata
            month = get_val(row, "month", col_idx)
            year = get_val(row, "year", col_idx)
            recall_period_str = get_val(row, "recall_period", col_idx)
            recall_period = float(recall_period_str) if recall_period_str else 90.0
            hh_id = get_val(row, "hh", col_idx)
            cluster_id = get_val(row, "cluster", col_idx)
            team_id = get_val(row, "team", col_idx)

            # Initialize counters for this household
            total = 0
            births = 0
            deaths = 0
            joined = 0
            left = 0

            total_u5 = 0
            births_u5 = 0
            deaths_u5 = 0
            joined_u5 = 0
            left_u5 = 0

            # Scan each of the P1..P20 member slots
            for i in range(1, max_p + 1):
                sex_col = f"p{i}_sex"
                age_col = f"p{i}_age"
                join_col = f"p{i}_join"
                left_col = f"p{i}_left"
                born_col = f"p{i}_born"
                died_col = f"p{i}_died"

                sex_val = get_val(row, sex_col, col_idx).lower()
                age_raw = get_val(row, age_col, col_idx)

                # If both sex and age are empty, this member slot is inactive
                if not sex_val and not age_raw:
                    continue

                # Parse age
                try:
                    age = float(age_raw)
                    is_u5 = age < 5.0
                except ValueError:
                    is_u5 = False

                # Extract status flags
                is_born = get_val(row, born_col, col_idx).lower() == "y"
                is_died = get_val(row, died_col, col_idx).lower() == "y"
                is_joined = get_val(row, join_col, col_idx).lower() == "y"
                is_left = get_val(row, left_col, col_idx).lower() == "y"

                # Increment household-level counts
                if is_born:
                    births += 1
                    if is_u5:
                        births_u5 += 1

                if is_died:
                    deaths += 1
                    if is_u5:
                        deaths_u5 += 1

                if is_joined:
                    joined += 1
                    if is_u5:
                        joined_u5 += 1

                if is_left:
                    left += 1
                    if is_u5:
                        left_u5 += 1

                # A person is counted in the final "Total" if they are part of
                # the household at the end (i.e. they haven't died and haven't left)
                if not is_died and not is_left:
                    total += 1
                    if is_u5:
                        total_u5 += 1

            # --- Person-Time Calculations (Mid-Interval Assumption) ---
            # Person-Time = Recall Period * [Total + 0.5 * (Deaths + Left - Births - Joined)]
            person_time = recall_period * (total + 0.5 * (deaths + left - births - joined))
            person_time_u5 = recall_period * (total_u5 + 0.5 * (deaths_u5 + left_u5 - births_u5 - joined_u5))

            # Avoid negative person times (e.g. if data is anomalous)
            person_time = max(0.0, person_time)
            person_time_u5 = max(0.0, person_time_u5)

            rows_to_write.append([
                month, year, int(recall_period), hh_id, cluster_id, team_id,
                total, births, deaths, joined, left,
                total_u5, births_u5, deaths_u5, joined_u5, left_u5,
                round(person_time, 2), round(person_time_u5, 2)
            ])

    # Write the output file
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HH_COLUMNS)
        writer.writerows(rows_to_write)


# ==============================================================================
# 5) Processing Logic for Aggregate Surveys
# ==============================================================================


def process_aggregate_file(in_path: Path, out_path: Path) -> None:
    """
    Reads an aggregate survey CSV (which is already at the household level),
    aligns the column headers, calculates person-time, and writes a clean CSV.
    """
    with open(in_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return

        col_idx = {col.lower(): idx for idx, col in enumerate(header)}

        rows_to_write = []
        for row in reader:
            if not row:
                continue

            # Skip embedded header rows (from concatenated survey sections)
            if get_val(row, "month", col_idx).lower() == "month":
                continue

            # Extract fields safely
            month = get_val(row, "month", col_idx)
            year = get_val(row, "year", col_idx)
            recall_period_str = get_val(row, "recall_period", col_idx)
            recall_period = float(recall_period_str) if recall_period_str else 90.0
            hh_id = get_val(row, "hh", col_idx)
            cluster_id = get_val(row, "cluster", col_idx)
            team_id = get_val(row, "team", col_idx)

            # Read counts directly, default to 0 if not present/empty
            def clean_int(val_str: str) -> int:
                return int(val_str) if val_str.isdigit() else 0

            total = clean_int(get_val(row, "total", col_idx))
            births = clean_int(get_val(row, "births", col_idx))
            deaths = clean_int(get_val(row, "deaths", col_idx))
            joined = clean_int(get_val(row, "joined", col_idx))
            left = clean_int(get_val(row, "left", col_idx))

            total_u5 = clean_int(get_val(row, "total_u5", col_idx))
            births_u5 = clean_int(get_val(row, "births_u5", col_idx))
            deaths_u5 = clean_int(get_val(row, "deaths_u5", col_idx))

            # Aggregate files do not have Joined_U5 / Left_U5, default to 0
            joined_u5 = 0
            left_u5 = 0

            # --- Person-Time Calculations (Mid-Interval Assumption) ---
            person_time = recall_period * (total + 0.5 * (deaths + left - births - joined))
            person_time_u5 = recall_period * (total_u5 + 0.5 * (deaths_u5 + left_u5 - births_u5 - joined_u5))

            person_time = max(0.0, person_time)
            person_time_u5 = max(0.0, person_time_u5)

            rows_to_write.append([
                month, year, int(recall_period), hh_id, cluster_id, team_id,
                total, births, deaths, joined, left,
                total_u5, births_u5, deaths_u5, joined_u5, left_u5,
                round(person_time, 2), round(person_time_u5, 2)
            ])

    # Write the output file
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HH_COLUMNS)
        writer.writerows(rows_to_write)


# ==============================================================================
# 6) Main Executable
# ==============================================================================


def main() -> None:
    print("=" * 60)
    print("Phase 2: Household Summary Statistics & Person-Time Calculation")
    print("=" * 60)

    ensure_dirs()
    clear_processed_outputs()

    # Admin2 Individual Data
    admin2_ind_files = list(ADMIN_IND_IN.glob("*.csv"))
    print(f"Aggregating {len(admin2_ind_files)} Admin2 individual survey datasets...")
    for f in admin2_ind_files:
        process_individual_file(f, ADMIN_IND_OUT / f.name)

    # LHZ Individual Data
    lhz_ind_files = list(LHZ_IND_IN.glob("*.csv"))
    print(f"Aggregating {len(lhz_ind_files)} LHZ individual survey datasets...")
    for f in lhz_ind_files:
        process_individual_file(f, LHZ_IND_OUT / f.name)

    # Admin2 Aggregate Data
    admin2_agg_files = list(ADMIN_AGG_IN.glob("*.csv"))
    print(f"Aligning {len(admin2_agg_files)} Admin2 aggregate survey datasets...")
    for f in admin2_agg_files:
        process_aggregate_file(f, ADMIN_AGG_OUT / f.name)

    print("\n[SUCCESS] Phase 2 executed completely. Output saved at:")
    print(f"  {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
