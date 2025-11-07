# Time Series Pattern Detection and Anomaly Detection

This document describes the time series analysis capabilities added to the Labor Contract Optimization system.

## Overview

The time series analysis module provides comprehensive pattern detection, statistical analysis, and anomaly detection for labor demand data. It helps identify trends, seasonal patterns, cycles, and unusual demand spikes or drops.

## Features

### 1. **Basic Statistical Analysis**

Comprehensive statistical measures including:

- **Central Tendency**: Mean, median, mode
- **Dispersion**: Standard deviation, variance, range, IQR, coefficient of variation
- **Distribution Shape**: Skewness, kurtosis
- **Percentiles**: 10th, 25th, 50th, 75th, 90th, 95th, 99th
- **Zero Analysis**: Holiday period detection and statistics
- **Non-Zero Statistics**: Metrics excluding holiday periods

### 2. **Pattern Detection**

#### Trend Analysis
- Linear regression to detect overall demand trend
- Moving averages (4-week, 13-week, 26-week)
- Trend direction classification (increasing, decreasing, stable)
- R-squared correlation strength
- Statistical significance testing

#### Seasonality Analysis
- Weekly pattern detection
- Autocorrelation analysis
- Seasonal strength calculation
- Zero-demand period identification (holidays)
- Peak demand week identification

#### Cycle Detection (Full version only)
- FFT-based frequency analysis
- Dominant cycle identification
- Detrended series analysis

### 3. **Anomaly Detection**

Multiple algorithms for detecting unusual demand patterns:

#### Z-Score Method
- Identifies outliers based on standard deviations from mean
- Configurable threshold (default: 3.0σ)
- Best for normally distributed data
- Excludes holiday periods (zero demand)

#### IQR (Interquartile Range) Method
- Robust to outliers in the data
- Uses quartiles to define normal range
- Configurable threshold (default: 1.5 × IQR)
- Classifies anomalies as "high" or "low"

#### Moving Window Method
- Detects anomalies relative to recent history
- Adaptive to local patterns
- Configurable window size (default: 4 weeks)
- Good for detecting sudden changes

### 4. **Visualization**

- Text-based demand visualization (standalone version)
- Comprehensive multi-panel plots (full version with matplotlib)
- Anomaly highlighting
- Trend lines and moving averages

## Usage

### Running the Analysis

```bash
cd src
python3 timeseries_analysis_standalone.py
```

### Output Files

The analysis generates three output files:

1. **`output/results/timeseries_analysis_report.txt`**
   - Comprehensive text report with all analyses
   - Statistical summaries
   - Pattern detection results
   - Anomaly listings
   - Key insights

2. **`output/results/timeseries_statistics.csv`**
   - All statistical metrics in CSV format
   - Easy to import into other tools
   - Machine-readable format

3. **`output/results/detected_anomalies.csv`** (if anomalies found)
   - List of detected anomalies
   - Week identifier, demand value, and anomaly score
   - Ready for further investigation

## Key Insights from Current Data (2025-2030)

Based on the analysis of 261 weeks of labor demand data:

### Statistical Summary
- **Mean Demand**: 18.79 workers (21.70 excluding holidays)
- **Median Demand**: 20.0 workers
- **Standard Deviation**: 8.44 workers (4.36 excluding holidays)
- **Coefficient of Variation**: 0.45 (high variability)

### Distribution Characteristics
- **Skewness**: -1.216 (left-skewed due to holiday zeros)
- **Kurtosis**: 0.657 (heavy-tailed distribution)
- **Range**: 0-29 workers

### Holiday Periods
- **Zero-Demand Weeks**: 35 weeks (13.4% of total)
- **Regular Holiday Weeks**: 13, 14, 26, 35, 47, 51, 52

### Trend Analysis
- **Direction**: Stable (no significant trend)
- **Slope**: +0.0086 workers/week
- **R-squared**: 0.006 (very weak correlation)
- **Interpretation**: Demand remains consistent over the 5-year period

### Seasonality
- **Seasonal Strength**: 0.822 (strong seasonality)
- **Pattern**: Weekly variations with consistent holiday periods
- **Peak Demand Weeks**: 28, 11, 29, 37, 9, 40, 43 (23-26 workers)
- **Low Demand Weeks**: Holiday periods + weeks 13, 14, 26, 35, 47, 51, 52

### Anomaly Detection
- **Z-Score Method (3.0σ)**: 0 anomalies detected
- **IQR Method (1.5×IQR)**: 0 anomalies detected
- **Moving Window (4-week, 2.5σ)**: 35 local variations detected

**Interpretation**: The demand pattern is highly consistent with predictable holiday periods. The moving window method detects local variations, but there are no significant outliers when looking at the overall distribution.

## Implementation Details

### Two Versions Available

1. **`timeseries_analysis_standalone.py`**
   - Uses only Python standard library
   - No external dependencies
   - Portable and lightweight
   - Text-based visualization
   - **Recommended for production use**

2. **`timeseries_analysis.py`** (requires installation)
   - Uses pandas, numpy, matplotlib, scipy, seaborn
   - Advanced visualizations
   - More sophisticated algorithms
   - Better performance on large datasets

### Architecture

```
TimeSeriesAnalyzer
├── __init__(data_path)              # Load and parse CSV data
├── compute_basic_statistics()       # Statistical measures
├── detect_trend()                   # Trend analysis
├── detect_seasonality()             # Seasonal pattern detection
├── detect_cycles()                  # Cycle detection (full version)
├── detect_anomalies_zscore()        # Z-score anomaly detection
├── detect_anomalies_iqr()           # IQR anomaly detection
├── detect_anomalies_moving_window() # Moving window anomaly detection
├── generate_report()                # Comprehensive text report
└── plot_comprehensive_analysis()    # Visualization (full version)
```

## Customization

### Adjusting Anomaly Detection Sensitivity

#### Z-Score Method
```python
# More sensitive (detect more anomalies)
anomalies = analyzer.detect_anomalies_zscore(threshold=2.0)

# Less sensitive (detect fewer anomalies)
anomalies = analyzer.detect_anomalies_zscore(threshold=4.0)
```

#### IQR Method
```python
# More sensitive
anomalies = analyzer.detect_anomalies_iqr(threshold=1.0)

# Less sensitive
anomalies = analyzer.detect_anomalies_iqr(threshold=2.0)
```

#### Moving Window Method
```python
# Shorter window (more responsive to recent changes)
anomalies = analyzer.detect_anomalies_moving_window(window_size=2, threshold=2.5)

# Longer window (more stable baseline)
anomalies = analyzer.detect_anomalies_moving_window(window_size=8, threshold=2.5)
```

## Integration with Labor Optimization

The time series analysis complements the existing optimization system by:

1. **Validating Demand Data**: Identifying unusual patterns before optimization
2. **Forecasting Support**: Understanding trends and seasonality for future planning
3. **Contract Strategy**: Informing contract mix based on demand variability
4. **Risk Assessment**: Identifying high-variability periods requiring flexibility
5. **Performance Monitoring**: Detecting when actual demand deviates from patterns

## Use Cases

### 1. Pre-Optimization Validation
Run analysis before optimization to ensure demand data quality:
```bash
python3 timeseries_analysis_standalone.py
# Review anomalies before proceeding with optimization
python3 labor_optimization.py
```

### 2. Contract Strategy Planning
Use seasonality and trend analysis to inform contract type selection:
- **High seasonality** → More 1-week contracts for flexibility
- **Stable demand** → More 52-week contracts for cost savings
- **Growing trend** → Consider longer contracts with buffer

### 3. Demand Forecasting
Use historical patterns to predict future demand:
- Identify seasonal peaks and troughs
- Project trend continuation
- Account for regular holiday periods

### 4. Risk Management
Identify periods requiring extra attention:
- High variability weeks
- Anomaly-prone periods
- Seasonal transition points

## Future Enhancements

Potential additions for future versions:

1. **Advanced Forecasting**
   - ARIMA/SARIMA models
   - Prophet-based forecasting
   - Machine learning predictions

2. **Real-time Monitoring**
   - Streaming anomaly detection
   - Alert generation
   - Dashboard integration

3. **Multivariate Analysis**
   - Cross-correlation with external factors
   - Cost impact analysis
   - Contract performance metrics

4. **Interactive Visualizations**
   - Web-based dashboard
   - Drill-down capabilities
   - Customizable date ranges

5. **Automated Reporting**
   - Scheduled analysis runs
   - Email notifications
   - Trend alerts

## References

- **Time Series Analysis**: Box, G. E. P., & Jenkins, G. M. (2015). Time Series Analysis: Forecasting and Control.
- **Anomaly Detection**: Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly detection: A survey.
- **Statistical Methods**: Hyndman, R. J., & Athanasopoulos, G. (2018). Forecasting: Principles and Practice.

## License

MIT License - Same as parent project

## Contributing

To extend the time series analysis module:

1. Add new detection methods to the `TimeSeriesAnalyzer` class
2. Update the `generate_report()` method to include new analyses
3. Add corresponding tests and documentation
4. Update this README with new features

## Support

For questions or issues:
- Review the generated reports in `output/results/`
- Check the inline documentation in the source code
- Consult the main project README.md
