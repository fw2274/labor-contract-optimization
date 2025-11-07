#!/usr/bin/env python3
"""
Time Series Pattern Detection and Anomaly Detection Module

This module provides comprehensive analysis of labor demand time series including:
- Basic statistical analysis
- Pattern detection (trends, seasonality, cycles)
- Anomaly detection algorithms
- Visualization capabilities
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesAnalyzer:
    """
    Comprehensive time series analyzer for labor demand data.

    Includes statistical analysis, pattern detection, and anomaly detection.
    """

    def __init__(self, data_path: str):
        """
        Initialize the analyzer with data.

        Args:
            data_path: Path to CSV file containing time series data
        """
        self.data = pd.read_csv(data_path)
        self.data['Date'] = pd.to_datetime(
            self.data['Year'].astype(str) + self.data['Week'].astype(str) + '1',
            format='%Y%W%w'
        )
        self.data = self.data.sort_values('Date').reset_index(drop=True)
        self.demand = self.data['Demand'].values
        self.non_zero_demand = self.demand[self.demand > 0]

    def compute_basic_statistics(self) -> Dict[str, float]:
        """
        Compute comprehensive statistical measures.

        Returns:
            Dictionary containing statistical measures
        """
        stats_dict = {
            # Central Tendency
            'mean': np.mean(self.demand),
            'median': np.median(self.demand),
            'mode': stats.mode(self.demand, keepdims=True)[0][0],

            # Dispersion
            'std': np.std(self.demand),
            'variance': np.var(self.demand),
            'range': np.ptp(self.demand),
            'iqr': np.percentile(self.demand, 75) - np.percentile(self.demand, 25),
            'coefficient_of_variation': np.std(self.demand) / np.mean(self.demand) if np.mean(self.demand) != 0 else 0,

            # Distribution Shape
            'skewness': stats.skew(self.demand),
            'kurtosis': stats.kurtosis(self.demand),

            # Percentiles
            'p10': np.percentile(self.demand, 10),
            'p25': np.percentile(self.demand, 25),
            'p50': np.percentile(self.demand, 50),
            'p75': np.percentile(self.demand, 75),
            'p90': np.percentile(self.demand, 90),
            'p95': np.percentile(self.demand, 95),
            'p99': np.percentile(self.demand, 99),

            # Min/Max
            'min': np.min(self.demand),
            'max': np.max(self.demand),

            # Zero Analysis
            'zero_count': np.sum(self.demand == 0),
            'zero_percentage': 100 * np.sum(self.demand == 0) / len(self.demand),

            # Non-Zero Statistics (excluding holidays)
            'mean_non_zero': np.mean(self.non_zero_demand) if len(self.non_zero_demand) > 0 else 0,
            'median_non_zero': np.median(self.non_zero_demand) if len(self.non_zero_demand) > 0 else 0,
            'std_non_zero': np.std(self.non_zero_demand) if len(self.non_zero_demand) > 0 else 0,
        }

        return stats_dict

    def detect_trend(self) -> Dict[str, any]:
        """
        Detect overall trend in the time series using multiple methods.

        Returns:
            Dictionary containing trend information
        """
        x = np.arange(len(self.demand))

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, self.demand)

        # Moving averages for trend detection
        window_sizes = [4, 13, 26]  # 4-week, quarterly, bi-annual
        moving_averages = {}
        for window in window_sizes:
            moving_averages[f'ma_{window}'] = pd.Series(self.demand).rolling(window=window, center=True).mean()

        # Determine trend direction
        if p_value < 0.05:  # Statistically significant
            if slope > 0:
                trend_direction = 'increasing'
            elif slope < 0:
                trend_direction = 'decreasing'
            else:
                trend_direction = 'stable'
        else:
            trend_direction = 'no significant trend'

        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2,
            'p_value': p_value,
            'std_error': std_err,
            'direction': trend_direction,
            'moving_averages': moving_averages,
            'trend_line': slope * x + intercept
        }

    def detect_seasonality(self) -> Dict[str, any]:
        """
        Detect seasonal patterns in the time series.

        Returns:
            Dictionary containing seasonality information
        """
        # Weekly patterns (within year)
        self.data['Week_of_Year'] = self.data['Week']
        weekly_pattern = self.data.groupby('Week_of_Year')['Demand'].agg(['mean', 'std', 'count'])

        # Check for zero-demand periods (holidays)
        zero_weeks = self.data[self.data['Demand'] == 0]['Week_of_Year'].unique()

        # Autocorrelation for different lags
        max_lag = min(52, len(self.demand) - 1)
        autocorr = [pd.Series(self.demand).autocorr(lag=i) for i in range(1, max_lag + 1)]

        # Find significant peaks in autocorrelation (indicating seasonality)
        autocorr_array = np.array(autocorr)
        peaks, properties = find_peaks(autocorr_array, height=0.3, distance=4)

        # Seasonal strength calculation
        if len(self.demand) >= 52:
            # Compare variance within season vs between seasons
            seasonal_component = self.data.groupby('Week_of_Year')['Demand'].transform('mean')
            deseasonalized = self.demand - seasonal_component[:len(self.demand)]
            seasonal_strength = 1 - (np.var(deseasonalized) / np.var(self.demand))
        else:
            seasonal_strength = 0

        return {
            'weekly_pattern': weekly_pattern,
            'zero_demand_weeks': sorted(zero_weeks),
            'autocorrelation': autocorr,
            'seasonal_peaks': peaks + 1,  # Convert to lag values
            'seasonal_strength': max(0, seasonal_strength),
            'has_strong_seasonality': seasonal_strength > 0.3
        }

    def detect_cycles(self) -> Dict[str, any]:
        """
        Detect cyclic patterns beyond seasonality.

        Returns:
            Dictionary containing cycle information
        """
        # Remove trend
        x = np.arange(len(self.demand))
        trend = self.detect_trend()
        detrended = self.demand - trend['trend_line']

        # FFT for frequency analysis
        fft = np.fft.fft(detrended)
        frequencies = np.fft.fftfreq(len(detrended))

        # Get dominant frequencies (excluding DC component)
        magnitude = np.abs(fft[1:len(fft)//2])
        freqs = frequencies[1:len(frequencies)//2]

        # Find peaks in frequency spectrum
        peaks, properties = find_peaks(magnitude, height=np.mean(magnitude) + np.std(magnitude))

        dominant_cycles = []
        if len(peaks) > 0:
            for peak in peaks[:5]:  # Top 5 cycles
                period = 1 / abs(freqs[peak]) if freqs[peak] != 0 else 0
                if period > 2:  # Only cycles longer than 2 weeks
                    dominant_cycles.append({
                        'period_weeks': period,
                        'strength': magnitude[peak]
                    })

        return {
            'detrended_series': detrended,
            'fft_magnitude': magnitude,
            'frequencies': freqs,
            'dominant_cycles': sorted(dominant_cycles, key=lambda x: x['strength'], reverse=True)
        }

    def detect_anomalies_statistical(self, method='zscore', threshold=3.0) -> Dict[str, any]:
        """
        Detect anomalies using statistical methods.

        Args:
            method: 'zscore', 'iqr', or 'mad'
            threshold: Threshold for anomaly detection

        Returns:
            Dictionary containing anomaly information
        """
        anomalies = np.zeros(len(self.demand), dtype=bool)
        scores = np.zeros(len(self.demand))

        # Only consider non-zero demand for anomalies (zeros are expected holidays)
        non_zero_mask = self.demand > 0

        if method == 'zscore':
            # Z-score method
            mean = np.mean(self.non_zero_demand)
            std = np.std(self.non_zero_demand)
            if std > 0:
                scores[non_zero_mask] = np.abs((self.demand[non_zero_mask] - mean) / std)
                anomalies[non_zero_mask] = scores[non_zero_mask] > threshold

        elif method == 'iqr':
            # Interquartile Range method
            q1 = np.percentile(self.non_zero_demand, 25)
            q3 = np.percentile(self.non_zero_demand, 75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            anomalies[non_zero_mask] = (self.demand[non_zero_mask] < lower_bound) | \
                                       (self.demand[non_zero_mask] > upper_bound)
            scores[non_zero_mask] = np.maximum(
                (lower_bound - self.demand[non_zero_mask]) / iqr if iqr > 0 else 0,
                (self.demand[non_zero_mask] - upper_bound) / iqr if iqr > 0 else 0
            )

        elif method == 'mad':
            # Median Absolute Deviation method
            median = np.median(self.non_zero_demand)
            mad = np.median(np.abs(self.non_zero_demand - median))
            if mad > 0:
                modified_z_scores = 0.6745 * (self.demand[non_zero_mask] - median) / mad
                scores[non_zero_mask] = np.abs(modified_z_scores)
                anomalies[non_zero_mask] = scores[non_zero_mask] > threshold

        # Get anomaly details
        anomaly_indices = np.where(anomalies)[0]
        anomaly_details = []
        for idx in anomaly_indices:
            anomaly_details.append({
                'index': idx,
                'week': self.data.iloc[idx]['Year_WW'],
                'demand': self.demand[idx],
                'score': scores[idx],
                'date': self.data.iloc[idx]['Date']
            })

        return {
            'anomalies': anomalies,
            'scores': scores,
            'anomaly_count': np.sum(anomalies),
            'anomaly_percentage': 100 * np.sum(anomalies) / len(self.demand),
            'anomaly_details': anomaly_details,
            'method': method,
            'threshold': threshold
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
        anomalies = np.zeros(len(self.demand), dtype=bool)
        scores = np.zeros(len(self.demand))

        for i in range(window_size, len(self.demand)):
            # Get window (excluding current point)
            window = self.demand[i-window_size:i]
            window_non_zero = window[window > 0]

            if len(window_non_zero) > 0 and self.demand[i] > 0:
                mean = np.mean(window_non_zero)
                std = np.std(window_non_zero)

                if std > 0:
                    z_score = abs((self.demand[i] - mean) / std)
                    scores[i] = z_score
                    if z_score > threshold:
                        anomalies[i] = True

        anomaly_indices = np.where(anomalies)[0]
        anomaly_details = []
        for idx in anomaly_indices:
            anomaly_details.append({
                'index': idx,
                'week': self.data.iloc[idx]['Year_WW'],
                'demand': self.demand[idx],
                'score': scores[idx],
                'date': self.data.iloc[idx]['Date']
            })

        return {
            'anomalies': anomalies,
            'scores': scores,
            'anomaly_count': np.sum(anomalies),
            'anomaly_percentage': 100 * np.sum(anomalies) / len(self.demand),
            'anomaly_details': anomaly_details,
            'method': 'moving_window',
            'window_size': window_size,
            'threshold': threshold
        }

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
        report.append(f"Date Range: {self.data['Date'].min().strftime('%Y-%m-%d')} to {self.data['Date'].max().strftime('%Y-%m-%d')}")
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
        report.append(f"  Skewness:                {stats['skewness']:.3f} ({'right-skewed' if stats['skewness'] > 0 else 'left-skewed' if stats['skewness'] < 0 else 'symmetric'})")
        report.append(f"  Kurtosis:                {stats['kurtosis']:.3f} ({'heavy-tailed' if stats['kurtosis'] > 0 else 'light-tailed'})")
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
        report.append(f"P-value:                   {trend['p_value']:.4f}")
        if trend['p_value'] < 0.05:
            report.append(f"Statistical Significance:  YES (p < 0.05)")
        else:
            report.append(f"Statistical Significance:  NO (p >= 0.05)")
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

        # Cycle Analysis
        report.append("CYCLE ANALYSIS")
        report.append("-" * 80)
        cycles = self.detect_cycles()
        if len(cycles['dominant_cycles']) > 0:
            report.append("Dominant Cycles Detected:")
            for i, cycle in enumerate(cycles['dominant_cycles'][:5], 1):
                report.append(f"  {i}. Period: {cycle['period_weeks']:.1f} weeks, Strength: {cycle['strength']:.2f}")
        else:
            report.append("No significant cycles detected beyond seasonality")
        report.append("")

        # Anomaly Detection
        report.append("ANOMALY DETECTION")
        report.append("-" * 80)

        # Z-score method
        anomalies_z = self.detect_anomalies_statistical(method='zscore', threshold=3.0)
        report.append(f"Z-Score Method (threshold=3.0):")
        report.append(f"  Anomalies Detected:      {anomalies_z['anomaly_count']} ({anomalies_z['anomaly_percentage']:.1f}%)")
        if anomalies_z['anomaly_count'] > 0:
            report.append(f"  Anomaly Weeks:")
            for detail in anomalies_z['anomaly_details'][:10]:
                report.append(f"    - Week {detail['week']}: Demand={detail['demand']:.0f}, Score={detail['score']:.2f}")
        report.append("")

        # IQR method
        anomalies_iqr = self.detect_anomalies_statistical(method='iqr', threshold=1.5)
        report.append(f"IQR Method (threshold=1.5):")
        report.append(f"  Anomalies Detected:      {anomalies_iqr['anomaly_count']} ({anomalies_iqr['anomaly_percentage']:.1f}%)")
        if anomalies_iqr['anomaly_count'] > 0:
            report.append(f"  Anomaly Weeks:")
            for detail in anomalies_iqr['anomaly_details'][:10]:
                report.append(f"    - Week {detail['week']}: Demand={detail['demand']:.0f}, Score={detail['score']:.2f}")
        report.append("")

        # Moving window method
        anomalies_mw = self.detect_anomalies_moving_window(window_size=4, threshold=2.5)
        report.append(f"Moving Window Method (window=4, threshold=2.5):")
        report.append(f"  Anomalies Detected:      {anomalies_mw['anomaly_count']} ({anomalies_mw['anomaly_percentage']:.1f}%)")
        if anomalies_mw['anomaly_count'] > 0:
            report.append(f"  Anomaly Weeks:")
            for detail in anomalies_mw['anomaly_details'][:10]:
                report.append(f"    - Week {detail['week']}: Demand={detail['demand']:.0f}, Score={detail['score']:.2f}")
        report.append("")

        report.append("=" * 80)

        return "\n".join(report)

    def plot_comprehensive_analysis(self, output_path='output/figures/timeseries_analysis.png'):
        """
        Create comprehensive visualization of all analyses.

        Args:
            output_path: Path to save the figure
        """
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

        # 1. Time Series with Trend
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(self.data['Date'], self.demand, 'b-', linewidth=1.5, alpha=0.7, label='Actual Demand')

        trend = self.detect_trend()
        ax1.plot(self.data['Date'], trend['trend_line'], 'r--', linewidth=2, label='Trend Line')

        # Add moving averages
        for key, ma in trend['moving_averages'].items():
            if key == 'ma_13':
                ax1.plot(self.data['Date'], ma, linewidth=2, alpha=0.8, label=f'MA-13 (Quarterly)')

        ax1.set_title('Time Series with Trend Analysis', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date', fontsize=11)
        ax1.set_ylabel('Demand (Workers)', fontsize=11)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # 2. Distribution
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.hist(self.non_zero_demand, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        ax2.axvline(np.mean(self.non_zero_demand), color='red', linestyle='--', linewidth=2, label='Mean')
        ax2.axvline(np.median(self.non_zero_demand), color='green', linestyle='--', linewidth=2, label='Median')
        ax2.set_title('Demand Distribution (Non-Zero)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Demand', fontsize=10)
        ax2.set_ylabel('Frequency', fontsize=10)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Box Plot
        ax3 = fig.add_subplot(gs[1, 1])
        bp = ax3.boxplot([self.non_zero_demand], patch_artist=True, widths=0.6)
        bp['boxes'][0].set_facecolor('lightblue')
        ax3.set_title('Box Plot (Non-Zero)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Demand', fontsize=10)
        ax3.set_xticklabels(['Demand'])
        ax3.grid(True, alpha=0.3)

        # 4. Seasonality - Weekly Pattern
        ax4 = fig.add_subplot(gs[1, 2])
        seasonality = self.detect_seasonality()
        weekly_avg = seasonality['weekly_pattern']['mean']
        ax4.bar(weekly_avg.index, weekly_avg.values, color='coral', alpha=0.7)
        ax4.set_title('Weekly Seasonality Pattern', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Week of Year', fontsize=10)
        ax4.set_ylabel('Average Demand', fontsize=10)
        ax4.grid(True, alpha=0.3)

        # 5. Autocorrelation
        ax5 = fig.add_subplot(gs[2, 0])
        autocorr = seasonality['autocorrelation']
        lags = range(1, len(autocorr) + 1)
        ax5.bar(lags, autocorr, color='teal', alpha=0.6)
        ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax5.axhline(y=0.3, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Threshold (0.3)')
        ax5.set_title('Autocorrelation Function', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Lag (weeks)', fontsize=10)
        ax5.set_ylabel('Autocorrelation', fontsize=10)
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Detrended Series
        ax6 = fig.add_subplot(gs[2, 1])
        cycles = self.detect_cycles()
        ax6.plot(self.data['Date'], cycles['detrended_series'], 'purple', linewidth=1, alpha=0.7)
        ax6.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax6.set_title('Detrended Series', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Date', fontsize=10)
        ax6.set_ylabel('Detrended Demand', fontsize=10)
        ax6.grid(True, alpha=0.3)

        # 7. Frequency Spectrum
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.plot(cycles['frequencies'], cycles['fft_magnitude'], 'darkgreen', linewidth=1.5)
        ax7.set_title('Frequency Spectrum', fontsize=12, fontweight='bold')
        ax7.set_xlabel('Frequency (1/weeks)', fontsize=10)
        ax7.set_ylabel('Magnitude', fontsize=10)
        ax7.grid(True, alpha=0.3)

        # 8. Anomaly Detection - Z-Score
        ax8 = fig.add_subplot(gs[3, :])
        anomalies_z = self.detect_anomalies_statistical(method='zscore', threshold=3.0)
        ax8.plot(self.data['Date'], self.demand, 'b-', linewidth=1.5, alpha=0.7, label='Demand')

        # Highlight anomalies
        anomaly_dates = self.data.loc[anomalies_z['anomalies'], 'Date']
        anomaly_values = self.demand[anomalies_z['anomalies']]
        ax8.scatter(anomaly_dates, anomaly_values, color='red', s=100, zorder=5,
                   label=f'Anomalies (n={len(anomaly_dates)})', marker='X')

        # Add confidence bands
        mean = np.mean(self.non_zero_demand)
        std = np.std(self.non_zero_demand)
        ax8.axhline(y=mean, color='green', linestyle='--', linewidth=1.5, alpha=0.5, label='Mean')
        ax8.axhline(y=mean + 3*std, color='red', linestyle=':', linewidth=1.5, alpha=0.5, label='±3σ')
        ax8.axhline(y=mean - 3*std, color='red', linestyle=':', linewidth=1.5, alpha=0.5)

        ax8.set_title('Anomaly Detection (Z-Score Method, threshold=3.0)', fontsize=14, fontweight='bold')
        ax8.set_xlabel('Date', fontsize=11)
        ax8.set_ylabel('Demand', fontsize=11)
        ax8.legend(loc='upper left')
        ax8.grid(True, alpha=0.3)

        plt.suptitle('Comprehensive Time Series Analysis - Labor Demand',
                    fontsize=16, fontweight='bold', y=0.995)

        # Save figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
        plt.close()


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
    analyzer = TimeSeriesAnalyzer(data_path)

    print(f"Loaded {len(analyzer.demand)} observations from {data_path}")
    print()

    # Generate and print report
    report = analyzer.generate_report()
    print(report)

    # Save report to file
    report_path = '../output/results/timeseries_analysis_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

    # Generate visualization
    print("\nGenerating comprehensive visualization...")
    analyzer.plot_comprehensive_analysis()

    print("\nAnalysis complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
