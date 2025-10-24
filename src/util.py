"""
Utility functions for labor demand generation and calendar operations.
"""

import datetime
from typing import List


def get_iso_week_count(year: int) -> int:
    """
    Determine the number of ISO weeks in a given year.

    A year has 53 weeks if:
    - It starts on a Thursday, OR
    - It's a leap year starting on a Wednesday

    Args:
        year: The year to check

    Returns:
        52 or 53 depending on the year
    """
    jan1 = datetime.date(year, 1, 1)
    dec31 = datetime.date(year, 12, 31)

    # Check if the year has 53 weeks using ISO week date system
    if dec31.isocalendar()[1] == 53:
        return 53
    return 52


def get_holiday_weeks(year: int) -> List[int]:
    """
    Get the week numbers that are considered holiday weeks for a given year.

    Holiday periods:
    - Thanksgiving (Week 47, late November)
    - Christmas/New Year (Weeks 51-52)
    - Easter (Week 13-14, spring)
    - July 4th (Week 26, summer)
    - Labor Day (Week 35, early September)

    Args:
        year: The year to get holidays for

    Returns:
        List of week numbers with zero demand
    """
    return [13, 14, 26, 35, 47, 51, 52]


def check_leap_years(start_year: int, end_year: int):
    """
    Check which years in the range have 53 ISO weeks.

    Args:
        start_year: Start of year range
        end_year: End of year range
    """
    print(f"\nISO Week Count from {start_year} to {end_year}:")
    print("-" * 40)

    for year in range(start_year, end_year + 1):
        week_count = get_iso_week_count(year)
        jan1 = datetime.date(year, 1, 1)
        is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

        status = f"{week_count} weeks"
        if week_count == 53:
            status += " ⭐ (53 weeks!)"

        print(f"{year}: {status} (Jan 1 = {jan1.strftime('%A')}, Leap year: {is_leap})")
