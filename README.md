# Labor Contract Optimization

**AI Vibe Coding Hackathon** submission from [IndustrialEngineer.ai](https://industrialengineer.ai)

A data science project that optimizes labor contract allocation to minimize costs while meeting weekly demand constraints.

## Project Structure

```
optimization/
├── src/                    # Source code
│   ├── generate_demand.py  # Generate demand time series
│   ├── labor_optimization.py  # Optimization solver
│   ├── analytics.py        # Analysis and visualization
│   └── util.py             # Utility functions
├── data/
│   ├── raw/                # Original input data
│   │   └── labor_demand_2025_2030.csv
│   └── processed/          # Transformed/analyzed data
│       └── supply_allocation.csv
├── output/
│   ├── figures/            # Generated visualizations
│   │   ├── demand_analysis.png
│   │   └── supply_allocation_analysis.png
│   └── results/            # Optimization results
│       ├── labor_optimization_results.csv
│       └── contract_fulfillment_detailed.csv
├── run_pipeline.sh         # Main pipeline script
└── README.md
```

## Features

### 1. Demand Generation ([src/generate_demand.py](src/generate_demand.py))
- Generates realistic labor demand time series
- Aligned with Gregorian calendar work weeks (ISO 8601)
- Handles leap years (53-week years)
- Includes holiday periods with zero demand
- Customizable parameters (min/max demand, time range)

### 2. Optimization ([src/labor_optimization.py](src/labor_optimization.py))
- **Objective**: Minimize total labor cost
- **Constraints**: Meet weekly demand exactly
- **Contract Types**:
  - 1-week: $1,800
  - 13-week: $22,100
  - 26-week: $41,600
  - 52-week: $78,000
- **Solver**: PuLP with CBC MILP solver
- **Outputs**:
  - Contract list with individual IDs
  - Detailed week-by-week fulfillment mapping

### 3. Analytics ([src/analytics.py](src/analytics.py))
- **Demand Analysis**:
  - Statistical summary (mean, median, std, percentiles)
  - Time series visualization
  - Distribution analysis
  - Yearly comparisons
  - Weekly seasonality patterns

- **Supply Allocation Analysis**:
  - Stacked area chart showing which contracts fulfill each week
  - Contract type contribution breakdown
  - Supply vs demand comparison
  - Surplus analysis
  - Pie chart of labor-weeks by contract type

### 4. Time Series Pattern Detection ([src/timeseries_analysis_standalone.py](src/timeseries_analysis_standalone.py))
- **Statistical Analysis**:
  - Comprehensive statistics (central tendency, dispersion, distribution shape)
  - Percentile analysis and range metrics
  - Holiday period detection and analysis

- **Pattern Detection**:
  - Trend analysis with linear regression
  - Seasonality detection and strength measurement
  - Autocorrelation analysis
  - Weekly pattern identification

- **Anomaly Detection**:
  - Z-Score method (outlier detection)
  - IQR method (robust outlier detection)
  - Moving window method (local anomaly detection)
  - Configurable sensitivity thresholds

- **Outputs**:
  - Comprehensive text report with insights
  - CSV exports of statistics and anomalies
  - Text-based visualization

See [TIMESERIES_ANALYSIS.md](TIMESERIES_ANALYSIS.md) for detailed documentation.

## Quick Start

### Installation

```bash
pip install pulp pandas matplotlib seaborn
```

### Usage

#### Option 1: Run Complete Pipeline (Recommended)

Run everything with a single command:

```bash
./run_pipeline.sh
```

This will:
1. Check and install dependencies
2. Generate demand data
3. Run optimization
4. Generate analytics and visualizations
5. Display summary of all generated files

#### Option 2: Run Individual Steps

1. **Generate demand data**:
```bash
cd src
python3 generate_demand.py
```

2. **Run optimization**:
```bash
python3 labor_optimization.py
```

3. **Generate analytics**:
```bash
python3 analytics.py
```

4. **Run time series pattern detection**:
```bash
python3 timeseries_analysis_standalone.py
```

## Output Files

### 1. Contract List (`output/results/labor_optimization_results.csv`)
Each row represents one contract:
```csv
Contract_ID,Start_Week,Duration,Cost,End_Week
C0001,202545,1,1800,202545
C0002,202545,1,1800,202545
...
```

### 2. Detailed Fulfillment (`output/results/contract_fulfillment_detailed.csv`)
Shows which contracts fulfill demand for each week:
```csv
Year_WW,Demand,Total_Supply,Surplus,Count_1wk,Count_13wk,Count_26wk,Count_52wk,Contract_IDs_1wk,Contract_IDs_13wk,Contract_IDs_26wk,Contract_IDs_52wk
202545,25,25,0,25,0,0,0,"C0001,C0002,...",None,None,None
202546,16,16,0,3,0,0,13,"C0026,C0027,C0028",None,None,"C0029,C0030,..."
...
```

**Key columns**:
- `Year_WW`: Work week identifier (YYYYWW format)
- `Demand`: Required labor for that week
- `Total_Supply`: Total workers assigned
- `Surplus`: Excess workers (Supply - Demand)
- `Count_Xwk`: Number of X-week contracts active that week
- `Contract_IDs_Xwk`: Specific contract IDs active that week

### 3. Supply Allocation (`data/processed/supply_allocation.csv`)
Aggregated supply by contract type per week:
```csv
Year_WW,Demand,Weekly_1wk,Quarterly_13wk,BiAnnual_26wk,Annual_52wk,Total_Supply,Surplus
202545,25,25,0,0,0,25,0
202546,16,3,0,0,13,16,0
...
```

## Example Results

For the 2025-2030 demand scenario:
- **Total Cost**: $8,573,600
- **Total Demand**: 4,905 labor-weeks
- **Average Cost per Labor-Week**: $1,747.93 (3% savings vs weekly-only)

**Optimal Contract Mix**:
- 1,438 weekly contracts (26.6% of labor-weeks)
- 7 × 26-week contracts (3.4%)
- 73 × 52-week contracts (70.1%)
- 0 × 13-week contracts (not cost-effective for this demand pattern)

## Customization

### Generate Custom Demand

```python
from src.generate_demand import generate_demand

generate_demand(
    start_year=2024,
    start_week=1,
    end_year=2026,
    end_week=52,
    min_demand=20,
    max_demand=40,
    output_file="../data/raw/custom_demand.csv"
)
```

### Modify Contract Types

Edit `CONTRACT_TYPES` in [src/labor_optimization.py](src/labor_optimization.py):

```python
CONTRACT_TYPES = {
    1: {'duration': 1, 'cost': 2000},    # Increased weekly cost
    4: {'duration': 4, 'cost': 7500},    # New 4-week contract
    52: {'duration': 52, 'cost': 75000}  # Reduced annual cost
}
```

## Key Insights

1. **52-week contracts provide base capacity**: Long-term contracts handle consistent demand
2. **1-week contracts provide flexibility**: Short-term contracts cover demand spikes
3. **13-week contracts rarely optimal**: Mid-range contracts often not cost-effective
4. **Minimal surplus maintained**: Average 1.96 workers/week surplus (9.5% overhead)

## License

MIT
