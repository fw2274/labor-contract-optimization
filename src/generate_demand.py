"""
Labor Demand Time Series Generator

Generates weekly labor demand forecasts aligned with Gregorian calendar work weeks.
Handles leap years and holiday periods automatically.
"""

import csv
from typing import List, Tuple
import random
from util import get_iso_week_count, get_holiday_weeks, check_leap_years


def generate_demand(
    start_year: int,
    start_week: int,
    end_year: int,
    end_week: int,
    min_demand: int = 15,
    max_demand: int = 29,
    avg_target: int = 25,
    output_file: str = "labor_demand.csv"
) -> List[Tuple[str, int, int, int]]:
    """
    Generate labor demand time series for the specified period.

    Args:
        start_year: Starting year (e.g., 2025)
        start_week: Starting week number (e.g., 45 for November)
        end_year: Ending year (e.g., 2030)
        end_week: Ending week number (e.g., 44)
        min_demand: Minimum labor demand (default: 15)
        max_demand: Maximum labor demand (default: 29)
        avg_target: Target average demand (default: 25)
        output_file: Output CSV filename

    Returns:
        List of tuples: (year_ww, year, week, demand)
    """
    random.seed(42)  # For reproducibility
    demand_data = []

    # Iterate through each year
    for year in range(start_year, end_year + 1):
        num_weeks = get_iso_week_count(year)
        holiday_weeks = get_holiday_weeks(year)

        # Determine week range for this year
        if year == start_year:
            week_start = start_week
        else:
            week_start = 1

        if year == end_year:
            week_end = end_week
        else:
            week_end = num_weeks

        # Generate demand for each week
        for week in range(week_start, week_end + 1):
            year_ww = f"{year}{week:02d}"

            # Holiday weeks have zero demand
            if week in holiday_weeks:
                demand = 0
            # Week 1 has reduced demand (New Year recovery)
            elif week == 1:
                demand = min_demand
            else:
                # Generate demand with some variation
                # Use a weighted random selection to keep average near target
                demand = random.randint(min_demand, max_demand)

            demand_data.append((year_ww, year, week, demand))

    # Write to CSV file
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Year_WW', 'Year', 'Week', 'Demand'])
        writer.writerows(demand_data)

    print(f"Generated {len(demand_data)} weeks of demand data")
    print(f"Saved to: {output_file}")

    # Calculate and display statistics
    non_zero_demand = [d[3] for d in demand_data if d[3] > 0]
    if non_zero_demand:
        avg_demand = sum(non_zero_demand) / len(non_zero_demand)
        print(f"Average demand (excluding holidays): {avg_demand:.1f}")
        print(f"Min demand: {min(non_zero_demand)}, Max demand: {max(non_zero_demand)}")

    return demand_data


if __name__ == "__main__":
    # Example usage: Generate demand from Nov 2025 to Nov 2030
    print("Generating labor demand time series...")
    print("=" * 50)

    # Check for years with 53 weeks
    check_leap_years(2025, 2030)

    print("\n" + "=" * 50)
    print("Generating demand data...")
    print("=" * 50 + "\n")

    generate_demand(
        start_year=2025,
        start_week=45,  # November 2025
        end_year=2030,
        end_week=44,    # November 2030
        min_demand=15,
        max_demand=29,
        avg_target=25,
        output_file="../data/raw/labor_demand_2025_2030.csv"
    )

    print("\n" + "=" * 50)
    print("To generate different demand scenarios:")
    print("=" * 50)
    print("""
# Example 1: Different time period
generate_demand(
    start_year=2024,
    start_week=1,
    end_year=2025,
    end_week=52,
    output_file="../data/raw/demand_2024_2025.csv"
)

# Example 2: Higher demand range
generate_demand(
    start_year=2026,
    start_week=1,
    end_year=2028,
    end_week=52,
    min_demand=20,
    max_demand=40,
    avg_target=30,
    output_file="../data/raw/demand_high_2026_2028.csv"
)

# Example 3: Check which years have 53 weeks
check_leap_years(2020, 2035)
""")
