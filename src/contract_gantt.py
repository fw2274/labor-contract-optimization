"""
Simplified Contract Gantt Chart

Shows only long-term and mid-term contracts (52, 26, 13 weeks) in consecutive blocks.
Excludes 1-week contracts for clarity.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import numpy as np


# Contract type colors
CONTRACT_COLORS = {
    1: '#C73E1D',    # Red - Weekly
    13: '#F18F01',   # Orange - Quarterly
    26: '#A23B72',   # Purple - Bi-annual
    52: '#2E86AB'    # Blue - Annual
}

CONTRACT_LABELS = {
    1: '1-week',
    13: '13-week',
    26: '26-week',
    52: '52-week'
}


def create_simplified_gantt(
    demand_file: str,
    contracts_file: str,
    output_file: str = "../output/figures/contract_gantt_simplified.png"
):
    """
    Create a simplified Gantt chart showing only long-term contracts.

    Args:
        demand_file: Path to demand CSV file
        contracts_file: Path to contracts CSV file
        output_file: Path to save the chart
    """
    # Load data
    demand_df = pd.read_csv(demand_file)
    contracts_df = pd.read_csv(contracts_file)

    weeks = demand_df['Year_WW'].tolist()
    demand = demand_df['Demand'].tolist()
    n_weeks = len(weeks)

    print(f"Loaded {len(contracts_df)} total contracts")

    # Filter for only long-term contracts (exclude 1-week)
    long_contracts = contracts_df[contracts_df['Duration'] > 1].copy()
    print(f"Filtered to {len(long_contracts)} long-term contracts (13, 26, 52 weeks)")

    # Create week index mapping
    week_to_idx = {week: idx for idx, week in enumerate(weeks)}

    # Prepare contract data
    contract_data = []
    for _, row in long_contracts.iterrows():
        start_idx = week_to_idx.get(row['Start_Week'], None)
        end_idx = week_to_idx.get(row['End_Week'], None)

        if start_idx is not None and end_idx is not None:
            contract_data.append({
                'id': row['Contract_ID'],
                'start_idx': start_idx,
                'end_idx': end_idx,
                'duration': row['Duration'],
                'start_week': row['Start_Week'],
                'end_week': row['End_Week'],
                'cost': row['Cost']
            })

    # Sort by duration (longest first), then by start time
    contract_data.sort(key=lambda x: (-x['duration'], x['start_idx']))

    n_contracts = len(contract_data)
    print(f"Displaying {n_contracts} contracts")

    # Create figure with 2 panels
    fig = plt.figure(figsize=(22, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[5, 1], hspace=0.3)

    # Panel 1: Gantt Chart
    ax1 = fig.add_subplot(gs[0])

    print("Creating Gantt chart...")
    for i, contract in enumerate(contract_data):
        start = contract['start_idx']
        end = contract['end_idx']
        duration_type = contract['duration']
        width = end - start + 1

        # Draw rectangle
        rect = Rectangle(
            (start, i + 0.1),
            width,
            0.8,
            facecolor=CONTRACT_COLORS[duration_type],
            edgecolor='white',
            linewidth=1,
            alpha=0.9
        )
        ax1.add_patch(rect)

        # Add contract ID label
        label_text = contract['id']
        ax1.text(
            start + width/2,
            i + 0.5,
            label_text,
            ha='center',
            va='center',
            fontsize=8,
            color='white',
            weight='bold'
        )

    # Add weekly contracts as orange blocks in separate section (no overlap)
    # Calculate weekly supply needed per week
    weekly_supply_by_week = []
    for week_idx in range(n_weeks):
        long_term_supply = sum(1 for c in contract_data
                              if c['start_idx'] <= week_idx <= c['end_idx'])
        weekly_needed = max(0, demand[week_idx] - long_term_supply)
        weekly_supply_by_week.append(weekly_needed)

    # Weekly contracts start after long-term contracts (no overlap)
    weekly_y_start = n_contracts + 1.5  # Gap between sections

    # Draw orange blocks for weekly contracts as aggregated height
    for week_idx in range(n_weeks):
        if weekly_supply_by_week[week_idx] > 0:
            height = weekly_supply_by_week[week_idx]
            rect = Rectangle(
                (week_idx, weekly_y_start),
                1,  # width = 1 week
                height * 0.8,  # height = number of workers
                facecolor='#FF8C00',  # Orange color for weekly
                edgecolor='white',
                linewidth=0.5,
                alpha=0.85
            )
            ax1.add_patch(rect)

    # Calculate total chart height
    max_weekly_height = max(weekly_supply_by_week) if weekly_supply_by_week else 0
    total_chart_height = weekly_y_start + max_weekly_height + 2

    # Configure Panel 1
    ax1.set_xlim(0, n_weeks)
    ax1.set_ylim(0, total_chart_height)
    ax1.set_xlabel('Week Index (0 = Week 202545, 260 = Week 203044)',
                   fontsize=13, fontweight='bold')
    ax1.set_ylabel('Contract Index', fontsize=13, fontweight='bold')
    ax1.set_title('Contract Gantt Chart: Long-Term Contracts + Weekly Flex (Orange)',
                  fontsize=15, fontweight='bold', pad=25)

    # Add year markers
    year_labels = ['2025\nWk45', '2026\nWk1', '2027\nWk1', '2028\nWk1', '2029\nWk1', '2030\nWk1']
    year_positions = [0, 8, 60, 112, 164, 216]

    for pos, label in zip(year_positions, year_labels):
        if pos < n_weeks:
            ax1.axvline(pos, color='black', linestyle='--', linewidth=1.5, alpha=0.4)
            ax1.text(pos, total_chart_height + total_chart_height*0.01, label,
                    ha='left', va='bottom', fontsize=10, fontweight='bold')

    ax1.grid(True, axis='x', alpha=0.2, linewidth=0.5)
    ax1.set_yticks(range(0, int(total_chart_height), max(1, int(total_chart_height)//10)))

    # Add separator line between long-term and weekly
    ax1.axhline(weekly_y_start - 0.75, color='gray', linestyle='-', linewidth=2, alpha=0.6)
    ax1.text(n_weeks * 0.02, weekly_y_start - 0.5, 'Weekly Contracts (aggregated) →',
             va='bottom', ha='left', fontsize=10, style='italic', color='gray')

    # Create legend (will be placed at bottom later)
    legend_elements = []
    for dur in [52, 26, 13]:
        count = sum(1 for c in contract_data if c['duration'] == dur)
        if count > 0:
            legend_elements.append(
                mpatches.Patch(
                    facecolor=CONTRACT_COLORS[dur],
                    label=f'{CONTRACT_LABELS[dur]} ({count} contracts)',
                    edgecolor='white'
                )
            )
    # Add weekly to legend
    total_weekly = int(sum(weekly_supply_by_week))
    legend_elements.append(
        mpatches.Patch(
            facecolor='#FF8C00',
            label=f'1-week (orange) ({total_weekly} labor-weeks)',
            edgecolor='white'
        )
    )


    # Panel 2: Statistics and Legend at bottom
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')

    # Create contract statistics summary
    stats_lines = []
    stats_lines.append("CONTRACT STATISTICS:")
    stats_lines.append("")

    # Calculate total cost and labor-weeks for percentages
    all_contracts = contracts_df
    total_cost_all = all_contracts['Cost'].sum()
    total_labor_weeks_all = sum(c['end_idx'] - c['start_idx'] + 1 for c in contract_data) + sum(weekly_supply_by_week)

    # Calculate statistics for each contract type
    for dur in [52, 26, 13]:
        contracts_of_type = [c for c in contract_data if c['duration'] == dur]
        if contracts_of_type:
            count = len(contracts_of_type)
            total_cost = sum(c['cost'] for c in contracts_of_type)
            total_labor_weeks = sum(c['end_idx'] - c['start_idx'] + 1 for c in contracts_of_type)
            cost_pct = (total_cost / total_cost_all * 100) if total_cost_all > 0 else 0
            lw_pct = (total_labor_weeks / total_labor_weeks_all * 100) if total_labor_weeks_all > 0 else 0
            stats_lines.append(
                f"  {CONTRACT_LABELS[dur]:12s}: {count:3d} contracts ({cost_pct:5.1f}%)  |  "
                f"Cost: ${total_cost:,}  |  {total_labor_weeks:,} labor-weeks ({lw_pct:5.1f}%)"
            )

    # Add weekly statistics
    total_weekly_lw = int(sum(weekly_supply_by_week))
    weekly_count = len(contracts_df[contracts_df['Duration'] == 1])
    weekly_cost = weekly_count * 1800  # $1,800 per weekly contract
    cost_pct = (weekly_cost / total_cost_all * 100) if total_cost_all > 0 else 0
    lw_pct = (total_weekly_lw / total_labor_weeks_all * 100) if total_labor_weeks_all > 0 else 0
    stats_lines.append(
        f"  {'1-week':12s}: {weekly_count:3d} contracts ({cost_pct:5.1f}%)  |  "
        f"Cost: ${weekly_cost:,}  |  {total_weekly_lw:,} labor-weeks ({lw_pct:5.1f}%)"
    )

    stats_text = "\n".join(stats_lines)

    # Display statistics in the panel
    ax2.text(0.05, 0.7, stats_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3, pad=0.8))

    # Place legend at bottom center with white background
    legend = fig.legend(handles=legend_elements, loc='lower center',
                       bbox_to_anchor=(0.5, 0.02), ncol=4, fontsize=11,
                       title='Contract Types', title_fontsize=12,
                       frameon=True, fancybox=True, shadow=True,
                       framealpha=1.0, edgecolor='gray')
    legend.get_frame().set_facecolor('white')

    # Add spacing for clean layout
    plt.subplots_adjust(bottom=0.08)

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Simplified Gantt chart saved to: {output_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("SIMPLIFIED GANTT CHART SUMMARY")
    print("=" * 70)
    print(f"Long-term contracts displayed: {n_contracts}")
    print(f"  52-week: {sum(1 for c in contract_data if c['duration'] == 52)}")
    print(f"  26-week: {sum(1 for c in contract_data if c['duration'] == 26)}")
    print(f"  13-week: {sum(1 for c in contract_data if c['duration'] == 13)}")
    print(f"\n1-week contracts: {len(contracts_df[contracts_df['Duration'] == 1])} (not shown in chart)")
    print(f"Time span: {n_weeks} weeks ({weeks[0]} to {weeks[-1]})")

    return {
        'n_contracts': n_contracts,
        'contract_data': contract_data
    }


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("SIMPLIFIED CONTRACT GANTT CHART")
    print("=" * 70)

    result = create_simplified_gantt(
        demand_file="../data/raw/labor_demand_2025_2030.csv",
        contracts_file="../output/results/labor_optimization_results.csv",
        output_file="../output/figures/contract_gantt_simplified.png"
    )

    print("\n✓ Complete!")
