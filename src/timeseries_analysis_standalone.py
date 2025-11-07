#!/usr/bin/env python3
"""
Time Series Pattern Detection and Anomaly Detection Module (Standalone Version)

This module provides comprehensive analysis of labor demand time series using
only Python standard library. Includes:
- Basic statistical analysis
- Pattern detection (trends, seasonality)
- Anomaly detection algorithms
- Text-based visualization
"""

import csv
import math
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


class TimeSeriesAnalyzer:
    """
    Comprehensive time series analyzer for labor demand data.
    Uses only Python standard library.
    """

    def __init__(self, data_path: str):
        """
        Initialize the analyzer with data.

        Args:
            data_path: Path to CSV file containing time series data
        """
        self.data = []
        self.demand = []

        # Read CSV data
        with open(data_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append({
                    'Year_WW': row['Year_WW'],
                    'Year': int(row['Year']),
                    'Week': int(row['Week']),
                    'Demand': float(row['Demand'])
                })
                self.demand.append(float(row['Demand']))

        self.non_zero_demand = [d for d in self.demand if d > 0]

    def mean(self, data):
        """Calculate mean."""
        return sum(data) / len(data) if len(data) > 0 else 0

    def median(self, data):
        """Calculate median."""
        sorted_data = sorted(data)
        n = len(sorted_data)
        if n == 0:
            return 0
        mid = n // 2
        if n % 2 == 0:
            return (sorted_data[mid-1] + sorted_data[mid]) / 2
        else:
            return sorted_data[mid]

    def mode(self, data):
        """Calculate mode (most frequent value)."""
        if not data:
            return 0
        counts = {}
        for val in data:
            counts[val] = counts.get(val, 0) + 1
        return max(counts, key=counts.get)

    def variance(self, data):
        """Calculate variance."""
        if len(data) <= 1:
            return 0
        m = self.mean(data)
        return sum((x - m) ** 2 for x in data) / len(data)

    def std(self, data):
        """Calculate standard deviation."""
        return math.sqrt(self.variance(data))

    def percentile(self, data, p):
        """Calculate percentile."""
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * p / 100
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_data[int(k)]
        d0 = sorted_data[int(f)] * (c - k)
        d1 = sorted_data[int(c)] * (k - f)
        return d0 + d1

    def skewness(self, data):
        """Calculate skewness."""
        if len(data) <= 2:
            return 0
        m = self.mean(data)
        s = self.std(data)
        if s == 0:
            return 0
        n = len(data)
        return (n / ((n-1) * (n-2))) * sum(((x - m) / s) ** 3 for x in data)

    def kurtosis(self, data):
        """Calculate excess kurtosis."""
        if len(data) <= 3:
            return 0
        m = self.mean(data)
        s = self.std(data)
        if s == 0:
            return 0
        n = len(data)
        return (n * (n+1) / ((n-1) * (n-2) * (n-3))) * sum(((x - m) / s) ** 4 for x in data) - \
               (3 * (n-1) ** 2 / ((n-2) * (n-3)))

    def compute_basic_statistics(self) -> Dict[str, float]:
        """
        Compute comprehensive statistical measures.

        Returns:
            Dictionary containing statistical measures
        """
        stats_dict = {
            # Central Tendency
            'mean': self.mean(self.demand),
            'median': self.median(self.demand),
            'mode': self.mode(self.demand),

            # Dispersion
            'std': self.std(self.demand),
            'variance': self.variance(self.demand),
            'range': max(self.demand) - min(self.demand),
            'iqr': self.percentile(self.demand, 75) - self.percentile(self.demand, 25),
            'coefficient_of_variation': self.std(self.demand) / self.mean(self.demand) if self.mean(self.demand) != 0 else 0,

            # Distribution Shape
            'skewness': self.skewness(self.demand),
            'kurtosis': self.kurtosis(self.demand),

            # Percentiles
            'p10': self.percentile(self.demand, 10),
            'p25': self.percentile(self.demand, 25),
            'p50': self.percentile(self.demand, 50),
            'p75': self.percentile(self.demand, 75),
            'p90': self.percentile(self.demand, 90),
            'p95': self.percentile(self.demand, 95),
            'p99': self.percentile(self.demand, 99),

            # Min/Max
            'min': min(self.demand),
            'max': max(self.demand),

            # Zero Analysis
            'zero_count': sum(1 for d in self.demand if d == 0),
            'zero_percentage': 100 * sum(1 for d in self.demand if d == 0) / len(self.demand),

            # Non-Zero Statistics
            'mean_non_zero': self.mean(self.non_zero_demand) if len(self.non_zero_demand) > 0 else 0,
            'median_non_zero': self.median(self.non_zero_demand) if len(self.non_zero_demand) > 0 else 0,
            'std_non_zero': self.std(self.non_zero_demand) if len(self.non_zero_demand) > 0 else 0,
        }

        return stats_dict

    def linear_regression(self, x, y):
        """
        Perform simple linear regression.

        Returns: (slope, intercept, r_squared)
        """
        n = len(x)
        if n == 0:
            return 0, 0, 0

        x_mean = self.mean(x)
        y_mean = self.mean(y)

        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0, y_mean, 0

        slope = numerator / denominator
        intercept = y_mean - slope * x_mean

        # Calculate R-squared
        ss_tot = sum((y[i] - y_mean) ** 2 for i in range(n))
        ss_res = sum((y[i] - (slope * x[i] + intercept)) ** 2 for i in range(n))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        return slope, intercept, r_squared

    def detect_trend(self) -> Dict[str, any]:
        """
        Detect overall trend in the time series.

        Returns:
            Dictionary containing trend information
        """
        x = list(range(len(self.demand)))
        slope, intercept, r_squared = self.linear_regression(x, self.demand)

        # Determine trend direction
        if abs(slope) < 0.01:
            trend_direction = 'stable'
        elif slope > 0:
            trend_direction = 'increasing'
        else:
            trend_direction = 'decreasing'

        # Calculate moving averages
        ma_4 = self.moving_average(self.demand, 4)
        ma_13 = self.moving_average(self.demand, 13)
        ma_26 = self.moving_average(self.demand, 26)

        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'direction': trend_direction,
            'ma_4': ma_4,
            'ma_13': ma_13,
            'ma_26': ma_26
        }

    def moving_average(self, data, window):
        """Calculate centered moving average."""
        if len(data) < window:
            return [None] * len(data)

        result = []
        half_window = window // 2

        for i in range(len(data)):
            if i < half_window or i >= len(data) - half_window:
                result.append(None)
            else:
                start = i - half_window
                end = i + half_window + 1
                result.append(self.mean(data[start:end]))

        return result

    def autocorrelation(self, data, lag):
        """Calculate autocorrelation at given lag."""
        n = len(data)
        if lag >= n or lag < 1:
            return 0

        mean_val = self.mean(data)

        c0 = sum((data[i] - mean_val) ** 2 for i in range(n))
        c_lag = sum((data[i] - mean_val) * (data[i+lag] - mean_val) for i in range(n - lag))

        if c0 == 0:
            return 0

        return c_lag / c0

    def detect_seasonality(self) -> Dict[str, any]:
        """
        Detect seasonal patterns in the time series.

        Returns:
            Dictionary containing seasonality information
        """
        # Weekly patterns
        weekly_pattern = defaultdict(list)
        for record in self.data:
            weekly_pattern[record['Week']].append(record['Demand'])

        weekly_avg = {week: self.mean(demands) for week, demands in weekly_pattern.items()}
        weekly_std = {week: self.std(demands) for week, demands in weekly_pattern.items()}

        # Zero demand weeks
        zero_weeks = sorted(set(record['Week'] for record in self.data if record['Demand'] == 0))

        # Autocorrelation for different lags
        max_lag = min(52, len(self.demand) - 1)
        autocorr = [self.autocorrelation(self.demand, lag) for lag in range(1, max_lag + 1)]

        # Find peaks in autocorrelation
        seasonal_peaks = []
        for i in range(1, len(autocorr) - 1):
            if autocorr[i] > 0.3 and autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                seasonal_peaks.append(i + 1)

        # Seasonal strength
        if len(self.demand) >= 52:
            seasonal_component = []
            for i, record in enumerate(self.data):
                week = record['Week']
                seasonal_component.append(weekly_avg.get(week, 0))

            deseasonalized = [self.demand[i] - seasonal_component[i] for i in range(len(self.demand))]
            var_deseasonalized = self.variance(deseasonalized)
            var_original = self.variance(self.demand)
            seasonal_strength = 1 - (var_deseasonalized / var_original) if var_original != 0 else 0
            seasonal_strength = max(0, seasonal_strength)
        else:
            seasonal_strength = 0

        return {
            'weekly_avg': weekly_avg,
            'weekly_std': weekly_std,
            'zero_demand_weeks': zero_weeks,
            'autocorrelation': autocorr,
            'seasonal_peaks': seasonal_peaks,
            'seasonal_strength': seasonal_strength,
            'has_strong_seasonality': seasonal_strength > 0.3
        }

    def detect_anomalies_zscore(self, threshold=3.0) -> Dict[str, any]:
        """
        Detect anomalies using Z-score method.

        Args:
            threshold: Z-score threshold for anomaly detection

        Returns:
            Dictionary containing anomaly information
        """
        mean_val = self.mean(self.non_zero_demand)
        std_val = self.std(self.non_zero_demand)

        anomalies = []
        scores = []

        for i, demand in enumerate(self.demand):
            if demand > 0 and std_val > 0:
                z_score = abs((demand - mean_val) / std_val)
                scores.append(z_score)

                if z_score > threshold:
                    anomalies.append({
                        'index': i,
                        'week': self.data[i]['Year_WW'],
                        'demand': demand,
                        'score': z_score
                    })
            else:
                scores.append(0)

        return {
            'anomalies': anomalies,
            'scores': scores,
            'anomaly_count': len(anomalies),
            'anomaly_percentage': 100 * len(anomalies) / len(self.demand) if len(self.demand) > 0 else 0,
            'method': 'zscore',
            'threshold': threshold
        }

    def detect_anomalies_iqr(self, threshold=1.5) -> Dict[str, any]:
        """
        Detect anomalies using IQR method.

        Args:
            threshold: IQR multiplier for bounds

        Returns:
            Dictionary containing anomaly information
        """
        q1 = self.percentile(self.non_zero_demand, 25)
        q3 = self.percentile(self.non_zero_demand, 75)
        iqr = q3 - q1

        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr

        anomalies = []

        for i, demand in enumerate(self.demand):
            if demand > 0:
                if demand < lower_bound or demand > upper_bound:
                    score = max((lower_bound - demand) / iqr, (demand - upper_bound) / iqr) if iqr > 0 else 0
                    anomalies.append({
                        'index': i,
                        'week': self.data[i]['Year_WW'],
                        'demand': demand,
                        'score': score,
                        'type': 'low' if demand < lower_bound else 'high'
                    })

        return {
            'anomalies': anomalies,
            'anomaly_count': len(anomalies),
            'anomaly_percentage': 100 * len(anomalies) / len(self.demand) if len(self.demand) > 0 else 0,
            'method': 'iqr',
            'threshold': threshold,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }

    def detect_anomalies_moving_window(self, window_size=4, threshold=2.5) -> Dict[str, any]:
        """
        Detect anomalies using moving window approach.

        Args:
            window_size: Size of moving window
            threshold: Z-score threshold for anomaly

        Returns:
            Dictionary containing anomaly information
        """
        anomalies = []

        for i in range(window_size, len(self.demand)):
            window = [self.demand[j] for j in range(i - window_size, i)]
            window_non_zero = [d for d in window if d > 0]

            if len(window_non_zero) > 0 and self.demand[i] > 0:
                mean_val = self.mean(window_non_zero)
                std_val = self.std(window_non_zero)

                if std_val > 0:
                    z_score = abs((self.demand[i] - mean_val) / std_val)

                    if z_score > threshold:
                        anomalies.append({
                            'index': i,
                            'week': self.data[i]['Year_WW'],
                            'demand': self.demand[i],
                            'score': z_score,
                            'window_mean': mean_val
                        })

        return {
            'anomalies': anomalies,
            'anomaly_count': len(anomalies),
            'anomaly_percentage': 100 * len(anomalies) / len(self.demand) if len(self.demand) > 0 else 0,
            'method': 'moving_window',
            'window_size': window_size,
            'threshold': threshold
        }

    def create_text_chart(self, data, width=60, height=15, title="Chart"):
        """Create a simple text-based chart."""
        if not data or all(d is None for d in data):
            return "No data to display"

        # Filter out None values
        valid_data = [(i, d) for i, d in enumerate(data) if d is not None]
        if not valid_data:
            return "No data to display"

        indices, values = zip(*valid_data)

        min_val = min(values)
        max_val = max(values)

        if max_val == min_val:
            return f"All values are {min_val}"

        lines = []
        lines.append(f"\n{title}")
        lines.append("=" * width)

        # Create chart
        for row in range(height, 0, -1):
            threshold = min_val + (max_val - min_val) * row / height
            line = f"{threshold:6.1f} |"

            for idx in range(len(data)):
                if data[idx] is not None and data[idx] >= threshold:
                    line += "█"
                else:
                    line += " "

            lines.append(line)

        lines.append("       +" + "-" * len(data))

        return "\n".join(lines)

    def generate_report(self) -> str:
        """
        Generate a comprehensive text report of all analyses.

        Returns:
            Formatted string report
        """
        report = []
        report.append("=" * 80)
        report.append("TIME SERIES ANALYSIS REPORT - LABOR DEMAND")
        report.append("=" * 80)
        report.append("")

        # Basic Statistics
        report.append("BASIC STATISTICS")
        report.append("-" * 80)
        stats = self.compute_basic_statistics()
        report.append(f"Total Observations: {len(self.demand)}")
        report.append(f"Date Range: Week {self.data[0]['Year_WW']} to Week {self.data[-1]['Year_WW']}")
        report.append("")
        report.append("Central Tendency:")
        report.append(f"  Mean:                    {stats['mean']:.2f}")
        report.append(f"  Median:                  {stats['median']:.2f}")
        report.append(f"  Mode:                    {stats['mode']:.2f}")
        report.append("")
        report.append("Dispersion:")
        report.append(f"  Standard Deviation:      {stats['std']:.2f}")
        report.append(f"  Variance:                {stats['variance']:.2f}")
        report.append(f"  Range:                   {stats['range']:.2f}")
        report.append(f"  IQR:                     {stats['iqr']:.2f}")
        report.append(f"  Coefficient of Variation: {stats['coefficient_of_variation']:.2f}")
        report.append("")
        report.append("Distribution Shape:")
        skew_desc = 'right-skewed' if stats['skewness'] > 0 else 'left-skewed' if stats['skewness'] < 0 else 'symmetric'
        kurt_desc = 'heavy-tailed' if stats['kurtosis'] > 0 else 'light-tailed'
        report.append(f"  Skewness:                {stats['skewness']:.3f} ({skew_desc})")
        report.append(f"  Kurtosis:                {stats['kurtosis']:.3f} ({kurt_desc})")
        report.append("")
        report.append("Percentiles:")
        report.append(f"  10th:  {stats['p10']:.1f}  |  25th:  {stats['p25']:.1f}  |  50th:  {stats['p50']:.1f}")
        report.append(f"  75th:  {stats['p75']:.1f}  |  90th:  {stats['p90']:.1f}  |  95th:  {stats['p95']:.1f}")
        report.append(f"  99th:  {stats['p99']:.1f}")
        report.append("")
        report.append("Range:")
        report.append(f"  Minimum:                 {stats['min']:.1f}")
        report.append(f"  Maximum:                 {stats['max']:.1f}")
        report.append("")
        report.append("Zero Analysis (Holiday Periods):")
        report.append(f"  Zero Count:              {int(stats['zero_count'])} weeks")
        report.append(f"  Zero Percentage:         {stats['zero_percentage']:.1f}%")
        report.append("")
        report.append("Non-Zero Statistics (Excluding Holidays):")
        report.append(f"  Mean:                    {stats['mean_non_zero']:.2f}")
        report.append(f"  Median:                  {stats['median_non_zero']:.2f}")
        report.append(f"  Std Dev:                 {stats['std_non_zero']:.2f}")
        report.append("")

        # Trend Analysis
        report.append("TREND ANALYSIS")
        report.append("-" * 80)
        trend = self.detect_trend()
        report.append(f"Direction:                 {trend['direction'].upper()}")
        report.append(f"Slope:                     {trend['slope']:.4f} workers/week")
        report.append(f"R-squared:                 {trend['r_squared']:.4f}")

        strength_desc = "strong" if trend['r_squared'] > 0.5 else "moderate" if trend['r_squared'] > 0.2 else "weak"
        report.append(f"Trend Strength:            {strength_desc.upper()}")
        report.append("")

        # Seasonality Analysis
        report.append("SEASONALITY ANALYSIS")
        report.append("-" * 80)
        seasonality = self.detect_seasonality()
        report.append(f"Seasonal Strength:         {seasonality['seasonal_strength']:.3f}")
        report.append(f"Has Strong Seasonality:    {'YES' if seasonality['has_strong_seasonality'] else 'NO'}")
        if len(seasonality['seasonal_peaks']) > 0:
            report.append(f"Seasonal Periods (lags):   {', '.join(map(str, seasonality['seasonal_peaks'][:5]))} weeks")
        report.append(f"Zero-Demand Weeks:         {', '.join(map(str, seasonality['zero_demand_weeks']))}")
        report.append("")

        # Weekly pattern statistics
        report.append("Weekly Pattern (Top 10 highest average demand weeks):")
        sorted_weeks = sorted(seasonality['weekly_avg'].items(), key=lambda x: x[1], reverse=True)[:10]
        for week, avg_demand in sorted_weeks:
            std = seasonality['weekly_std'].get(week, 0)
            report.append(f"  Week {week:2d}: {avg_demand:5.2f} ± {std:5.2f}")
        report.append("")

        # Autocorrelation peaks
        report.append("Autocorrelation Analysis:")
        autocorr = seasonality['autocorrelation']
        if len(autocorr) > 0:
            max_lag = min(10, len(autocorr))
            report.append(f"  First 10 lags: {', '.join(f'{autocorr[i]:.3f}' for i in range(max_lag))}")
        report.append("")

        # Anomaly Detection
        report.append("ANOMALY DETECTION")
        report.append("-" * 80)

        # Z-score method
        anomalies_z = self.detect_anomalies_zscore(threshold=3.0)
        report.append(f"Z-Score Method (threshold=3.0):")
        report.append(f"  Anomalies Detected:      {anomalies_z['anomaly_count']} ({anomalies_z['anomaly_percentage']:.1f}%)")
        if anomalies_z['anomaly_count'] > 0:
            report.append(f"  Anomaly Weeks:")
            for detail in anomalies_z['anomalies'][:10]:
                report.append(f"    - Week {detail['week']}: Demand={detail['demand']:.0f}, Z-Score={detail['score']:.2f}")
            if len(anomalies_z['anomalies']) > 10:
                report.append(f"    ... and {len(anomalies_z['anomalies']) - 10} more")
        report.append("")

        # IQR method
        anomalies_iqr = self.detect_anomalies_iqr(threshold=1.5)
        report.append(f"IQR Method (threshold=1.5):")
        report.append(f"  Anomalies Detected:      {anomalies_iqr['anomaly_count']} ({anomalies_iqr['anomaly_percentage']:.1f}%)")
        report.append(f"  Lower Bound:             {anomalies_iqr['lower_bound']:.2f}")
        report.append(f"  Upper Bound:             {anomalies_iqr['upper_bound']:.2f}")
        if anomalies_iqr['anomaly_count'] > 0:
            report.append(f"  Anomaly Weeks:")
            for detail in anomalies_iqr['anomalies'][:10]:
                report.append(f"    - Week {detail['week']}: Demand={detail['demand']:.0f}, Type={detail['type']}")
            if len(anomalies_iqr['anomalies']) > 10:
                report.append(f"    ... and {len(anomalies_iqr['anomalies']) - 10} more")
        report.append("")

        # Moving window method
        anomalies_mw = self.detect_anomalies_moving_window(window_size=4, threshold=2.5)
        report.append(f"Moving Window Method (window=4, threshold=2.5):")
        report.append(f"  Anomalies Detected:      {anomalies_mw['anomaly_count']} ({anomalies_mw['anomaly_percentage']:.1f}%)")
        if anomalies_mw['anomaly_count'] > 0:
            report.append(f"  Anomaly Weeks:")
            for detail in anomalies_mw['anomalies'][:10]:
                report.append(f"    - Week {detail['week']}: Demand={detail['demand']:.0f}, Score={detail['score']:.2f}, Window Mean={detail['window_mean']:.2f}")
            if len(anomalies_mw['anomalies']) > 10:
                report.append(f"    ... and {len(anomalies_mw['anomalies']) - 10} more")
        report.append("")

        # Summary chart
        report.append("DEMAND TIME SERIES VISUALIZATION")
        report.append("-" * 80)
        chart = self.create_text_chart(self.demand, width=70, height=12, title="Labor Demand Over Time")
        report.append(chart)
        report.append("")

        report.append("=" * 80)
        report.append("KEY INSIGHTS:")
        report.append("-" * 80)

        # Generate insights
        insights = []

        # Trend insight
        if trend['direction'] != 'stable':
            insights.append(f"1. The demand shows a {trend['direction']} trend with {strength_desc} correlation (R²={trend['r_squared']:.3f})")
        else:
            insights.append(f"1. The demand is relatively stable over time with no significant trend")

        # Seasonality insight
        if seasonality['has_strong_seasonality']:
            insights.append(f"2. Strong seasonal pattern detected (strength={seasonality['seasonal_strength']:.3f})")
        else:
            insights.append(f"2. Weak or no seasonal pattern (strength={seasonality['seasonal_strength']:.3f})")

        # Variability insight
        cv = stats['coefficient_of_variation']
        if cv > 0.3:
            variability = "high"
        elif cv > 0.15:
            variability = "moderate"
        else:
            variability = "low"
        insights.append(f"3. Demand variability is {variability} (CV={cv:.3f})")

        # Anomaly insight
        if anomalies_z['anomaly_count'] > 0:
            insights.append(f"4. Detected {anomalies_z['anomaly_count']} significant anomalies using Z-score method")
        else:
            insights.append(f"4. No significant anomalies detected - demand pattern is consistent")

        # Holiday pattern
        if len(seasonality['zero_demand_weeks']) > 0:
            insights.append(f"5. Regular zero-demand periods in weeks: {', '.join(map(str, seasonality['zero_demand_weeks']))}")

        for insight in insights:
            report.append(insight)

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)


def main():
    """
    Main function to run comprehensive time series analysis.
    """
    print("=" * 80)
    print("TIME SERIES PATTERN DETECTION AND ANOMALY DETECTION")
    print("=" * 80)
    print()

    # Initialize analyzer
    data_path = '../data/raw/labor_demand_2025_2030.csv'
    print(f"Loading data from: {data_path}")

    analyzer = TimeSeriesAnalyzer(data_path)
    print(f"Loaded {len(analyzer.demand)} observations")
    print()

    # Generate and print report
    print("Generating comprehensive analysis report...")
    print()
    report = analyzer.generate_report()
    print(report)

    # Save report to file
    report_path = '../output/results/timeseries_analysis_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

    # Save detailed statistics to CSV
    stats = analyzer.compute_basic_statistics()
    stats_path = '../output/results/timeseries_statistics.csv'
    with open(stats_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        for key, value in stats.items():
            writer.writerow([key, value])
    print(f"Statistics saved to: {stats_path}")

    # Save anomalies to CSV
    anomalies = analyzer.detect_anomalies_zscore(threshold=3.0)
    if anomalies['anomaly_count'] > 0:
        anomaly_path = '../output/results/detected_anomalies.csv'
        with open(anomaly_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Index', 'Week', 'Demand', 'Z_Score'])
            for detail in anomalies['anomalies']:
                writer.writerow([detail['index'], detail['week'], detail['demand'], detail['score']])
        print(f"Anomalies saved to: {anomaly_path}")

    print("\nAnalysis complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
