"""
Analytics for Labor Demand and Supply Optimization

Provides statistical analysis and visualizations for:
1. Demand input analysis (distribution, trends, seasonality)
2. Supply allocation analysis (how contracts fulfill demand)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Tuple

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


def analyze_demand(
    demand_file: str,
    output_dir: str = "../output/figures"
) -> pd.DataFrame:
    """
    Analyze demand data with statistical summaries and visualizations.

    Args:
        demand_file: Path to demand CSV file
        output_dir: Directory to save plots

    Returns:
        DataFrame with demand statistics
    """
    # Load demand data
    demand_df = pd.read_csv(demand_file)

    print("=" * 70)
    print("DEMAND ANALYSIS")
    print("=" * 70)

    # Basic statistics
    print("\n1. BASIC STATISTICS")
    print("-" * 70)
    print(f"Total weeks: {len(demand_df)}")
    print(f"Date range: {demand_df['Year_WW'].iloc[0]} to {demand_df['Year_WW'].iloc[-1]}")
    print(f"\nDemand Statistics:")
    print(demand_df['Demand'].describe())

    # Non-zero demand statistics
    non_zero_demand = demand_df[demand_df['Demand'] > 0]['Demand']
    print(f"\nNon-zero Demand Statistics:")
    print(non_zero_demand.describe())

    # Holiday weeks (zero demand)
    holiday_weeks = demand_df[demand_df['Demand'] == 0]
    print(f"\nHoliday weeks (zero demand): {len(holiday_weeks)}")
    print(f"Percentage of weeks with zero demand: {len(holiday_weeks)/len(demand_df)*100:.1f}%")

    # Yearly breakdown
    print("\n2. YEARLY BREAKDOWN")
    print("-" * 70)
    yearly_stats = demand_df.groupby('Year')['Demand'].agg([
        ('Total', 'sum'),
        ('Average', 'mean'),
        ('Min', 'min'),
        ('Max', 'max'),
        ('Std', 'std')
    ]).round(2)
    print(yearly_stats)

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Demand Analysis', fontsize=16, fontweight='bold')

    # 1. Time series plot
    ax1 = axes[0, 0]
    ax1.plot(range(len(demand_df)), demand_df['Demand'], linewidth=1, color='steelblue')
    ax1.fill_between(range(len(demand_df)), demand_df['Demand'], alpha=0.3, color='steelblue')
    ax1.set_xlabel('Week Index')
    ax1.set_ylabel('Demand (Labor Count)')
    ax1.set_title('Demand Time Series')
    ax1.grid(True, alpha=0.3)

    # 2. Distribution histogram
    ax2 = axes[0, 1]
    ax2.hist(demand_df['Demand'], bins=30, color='coral', edgecolor='black', alpha=0.7)
    ax2.axvline(demand_df['Demand'].mean(), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {demand_df["Demand"].mean():.1f}')
    ax2.axvline(demand_df['Demand'].median(), color='green', linestyle='--',
                linewidth=2, label=f'Median: {demand_df["Demand"].median():.1f}')
    ax2.set_xlabel('Demand (Labor Count)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Demand Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Yearly comparison boxplot
    ax3 = axes[1, 0]
    demand_df.boxplot(column='Demand', by='Year', ax=ax3)
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Demand (Labor Count)')
    ax3.set_title('Demand Distribution by Year')
    plt.sca(ax3)
    plt.xticks(rotation=45)

    # 4. Weekly pattern (average demand by week number)
    ax4 = axes[1, 1]
    weekly_pattern = demand_df.groupby('Week')['Demand'].mean()
    ax4.bar(weekly_pattern.index, weekly_pattern.values, color='mediumseagreen', alpha=0.7)
    ax4.set_xlabel('Week Number (1-52)')
    ax4.set_ylabel('Average Demand')
    ax4.set_title('Average Demand by Week of Year (Seasonality)')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/demand_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Demand analysis plot saved to: {output_dir}/demand_analysis.png")

    return demand_df


def analyze_supply_allocation(
    demand_file: str,
    results_file: str,
    output_dir: str = "../output/figures",
    processed_dir: str = "../data/processed"
) -> Dict:
    """
    Analyze how different contract types fulfill demand over time.

    Args:
        demand_file: Path to demand CSV file
        results_file: Path to optimization results CSV file
        output_dir: Directory to save plots

    Returns:
        Dictionary with supply allocation analysis
    """
    # Load data
    demand_df = pd.read_csv(demand_file)
    results_df = pd.read_csv(results_file)

    weeks = demand_df['Year_WW'].tolist()
    n_weeks = len(weeks)

    print("\n" + "=" * 70)
    print("SUPPLY ALLOCATION ANALYSIS")
    print("=" * 70)

    # Initialize supply tracking by contract type
    supply_by_type = {
        1: [0] * n_weeks,
        13: [0] * n_weeks,
        26: [0] * n_weeks,
        52: [0] * n_weeks
    }

    # Calculate supply contribution from each contract type
    for _, row in results_df.iterrows():
        start_week = row['Start_Week']
        end_week = row['End_Week']
        duration = row['Duration']

        # Find start and end indices
        if start_week in weeks and end_week in weeks:
            start_idx = weeks.index(start_week)
            end_idx = weeks.index(end_week)

            # Add supply for the duration of the contract (each contract is one worker)
            for w in range(start_idx, end_idx + 1):
                if w < n_weeks:
                    supply_by_type[duration][w] += 1

    # Create supply allocation dataframe
    supply_df = pd.DataFrame({
        'Year_WW': weeks,
        'Demand': demand_df['Demand'],
        'Weekly_1wk': supply_by_type[1],
        'Quarterly_13wk': supply_by_type[13],
        'BiAnnual_26wk': supply_by_type[26],
        'Annual_52wk': supply_by_type[52]
    })

    supply_df['Total_Supply'] = (supply_df['Weekly_1wk'] +
                                   supply_df['Quarterly_13wk'] +
                                   supply_df['BiAnnual_26wk'] +
                                   supply_df['Annual_52wk'])

    supply_df['Surplus'] = supply_df['Total_Supply'] - supply_df['Demand']

    # Save supply allocation data
    supply_df.to_csv(f'{processed_dir}/supply_allocation.csv', index=False)
    print(f"\n✓ Supply allocation data saved to: {processed_dir}/supply_allocation.csv")

    # Print statistics
    print("\n1. SUPPLY ALLOCATION SUMMARY")
    print("-" * 70)
    print(f"Average supply by contract type:")
    print(f"  1-week contracts:  {supply_df['Weekly_1wk'].mean():.2f} workers/week")
    print(f"  13-week contracts: {supply_df['Quarterly_13wk'].mean():.2f} workers/week")
    print(f"  26-week contracts: {supply_df['BiAnnual_26wk'].mean():.2f} workers/week")
    print(f"  52-week contracts: {supply_df['Annual_52wk'].mean():.2f} workers/week")
    print(f"\nTotal average supply: {supply_df['Total_Supply'].mean():.2f} workers/week")
    print(f"Total average demand: {supply_df['Demand'].mean():.2f} workers/week")
    print(f"Average surplus: {supply_df['Surplus'].mean():.2f} workers/week")

    # Calculate contribution percentages
    total_labor_weeks = supply_df['Total_Supply'].sum()
    print("\n2. CONTRACT TYPE CONTRIBUTION")
    print("-" * 70)
    for duration in [1, 13, 26, 52]:
        col_name = f"{'Weekly_1wk' if duration == 1 else 'Quarterly_13wk' if duration == 13 else 'BiAnnual_26wk' if duration == 26 else 'Annual_52wk'}"
        contribution = supply_df[col_name].sum()
        percentage = (contribution / total_labor_weeks * 100) if total_labor_weeks > 0 else 0
        print(f"  {duration}-week contracts: {contribution:,.0f} labor-weeks ({percentage:.1f}%)")

    # Create visualizations
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # 1. Stacked area chart - Supply allocation over time
    ax1 = fig.add_subplot(gs[0, :])
    x_range = range(len(supply_df))

    ax1.fill_between(x_range, 0, supply_df['Annual_52wk'],
                     label='52-week (Annual)', alpha=0.8, color='#2E86AB')
    ax1.fill_between(x_range, supply_df['Annual_52wk'],
                     supply_df['Annual_52wk'] + supply_df['BiAnnual_26wk'],
                     label='26-week (Bi-annual)', alpha=0.8, color='#A23B72')
    ax1.fill_between(x_range, supply_df['Annual_52wk'] + supply_df['BiAnnual_26wk'],
                     supply_df['Annual_52wk'] + supply_df['BiAnnual_26wk'] + supply_df['Quarterly_13wk'],
                     label='13-week (Quarterly)', alpha=0.8, color='#F18F01')
    ax1.fill_between(x_range,
                     supply_df['Annual_52wk'] + supply_df['BiAnnual_26wk'] + supply_df['Quarterly_13wk'],
                     supply_df['Total_Supply'],
                     label='1-week (Weekly)', alpha=0.8, color='#C73E1D')

    ax1.plot(x_range, supply_df['Demand'], 'k--', linewidth=2, label='Demand', alpha=0.7)

    ax1.set_xlabel('Week Index', fontsize=11)
    ax1.set_ylabel('Labor Count', fontsize=11)
    ax1.set_title('Supply Allocation by Contract Type (Stacked)', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 2. Supply vs Demand
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(x_range, supply_df['Demand'], label='Demand', linewidth=2, color='red', alpha=0.7)
    ax2.plot(x_range, supply_df['Total_Supply'], label='Total Supply', linewidth=2, color='blue', alpha=0.7)
    ax2.fill_between(x_range, supply_df['Demand'], supply_df['Total_Supply'],
                     where=(supply_df['Total_Supply'] >= supply_df['Demand']),
                     interpolate=True, alpha=0.3, color='green', label='Surplus')
    ax2.set_xlabel('Week Index', fontsize=11)
    ax2.set_ylabel('Labor Count', fontsize=11)
    ax2.set_title('Total Supply vs Demand', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 3. Surplus distribution
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(supply_df['Surplus'], bins=30, color='teal', edgecolor='black', alpha=0.7)
    ax3.axvline(supply_df['Surplus'].mean(), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {supply_df["Surplus"].mean():.1f}')
    ax3.set_xlabel('Surplus (Supply - Demand)', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Surplus Distribution', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Create a separate axes for pie chart in the top-right area
    ax_pie = fig.add_axes([0.65, 0.55, 0.25, 0.25])  # [left, bottom, width, height]

    contributions = [
        supply_df['Annual_52wk'].sum(),
        supply_df['BiAnnual_26wk'].sum(),
        supply_df['Quarterly_13wk'].sum(),
        supply_df['Weekly_1wk'].sum()
    ]
    labels = ['52-week\n(Annual)', '26-week\n(Bi-annual)', '13-week\n(Quarterly)', '1-week\n(Weekly)']
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

    wedges, texts, autotexts = ax_pie.pie(contributions, labels=labels, colors=colors,
                                           autopct='%1.1f%%', startangle=90)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    ax_pie.set_title('Labor-Weeks by Contract Type', fontsize=12, fontweight='bold')

    plt.savefig(f'{output_dir}/supply_allocation_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Supply allocation plot saved to: {output_dir}/supply_allocation_analysis.png")

    # Check for constraint violations
    violations = supply_df[supply_df['Total_Supply'] < supply_df['Demand']]
    if len(violations) > 0:
        print("\n⚠ WARNING: Demand not fully met for the following weeks:")
        print(violations[['Year_WW', 'Demand', 'Total_Supply', 'Surplus']])
    else:
        print("\n✓ All demand constraints satisfied!")

    return {
        'supply_df': supply_df,
        'results_df': results_df,
        'total_labor_weeks': total_labor_weeks
    }


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("LABOR DEMAND AND SUPPLY ANALYTICS")
    print("=" * 70)

    # Analyze demand
    demand_df = analyze_demand(
        demand_file="../data/raw/labor_demand_2025_2030.csv",
        output_dir="../output/figures"
    )

    # Analyze supply allocation
    supply_analysis = analyze_supply_allocation(
        demand_file="../data/raw/labor_demand_2025_2030.csv",
        results_file="../output/results/labor_optimization_results.csv",
        output_dir="../output/figures",
        processed_dir="../data/processed"
    )

    # Generate Gantt charts
    print("\n" + "=" * 70)
    print("GENERATING CONTRACT GANTT CHARTS")
    print("=" * 70)

    # Simplified version (recommended for viewing)
    try:
        from contract_gantt_simplified import create_simplified_gantt
        print("\n>> Creating simplified Gantt chart (long-term contracts only)...")
        create_simplified_gantt(
            demand_file="../data/raw/labor_demand_2025_2030.csv",
            contracts_file="../output/results/labor_optimization_results.csv",
            output_file="../output/figures/contract_gantt_simplified.png"
        )
    except Exception as e:
        print(f"Note: Simplified Gantt chart generation skipped ({e})")

    # Full version (all contracts)
    try:
        from contract_gantt import create_contract_gantt
        print("\n>> Creating full Gantt chart (all 1518 contracts)...")
        create_contract_gantt(
            demand_file="../data/raw/labor_demand_2025_2030.csv",
            contracts_file="../output/results/labor_optimization_results.csv",
            output_file="../output/figures/contract_gantt_chart.png"
        )
    except Exception as e:
        print(f"Note: Full Gantt chart generation skipped ({e})")

    print("\n" + "=" * 70)
    print("✓ ANALYTICS COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - output/figures/demand_analysis.png")
    print("  - output/figures/supply_allocation_analysis.png")
    print("  - output/figures/contract_gantt_simplified.png (RECOMMENDED)")
    print("  - output/figures/contract_gantt_chart.png (full, 1518 contracts)")
    print("  - data/processed/supply_allocation.csv")
