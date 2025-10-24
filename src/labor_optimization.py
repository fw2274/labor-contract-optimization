"""
Labor Contract Optimization

Minimizes labor cost by optimally selecting contract types to meet weekly demand.
Uses linear programming to find the optimal mix of contract durations.
"""

import pandas as pd
from pulp import LpMinimize, LpProblem, LpVariable, lpSum, LpStatus, value


# Contract types: duration in weeks and total cost
CONTRACT_TYPES = {
    1: {'duration': 1, 'cost': 1800},    # weekly
    13: {'duration': 13, 'cost': 22100}, # 13-week
    26: {'duration': 26, 'cost': 41600}, # 26-week
    52: {'duration': 52, 'cost': 78000}  # 52-week
}


def optimize_labor_contracts(
    demand_file: str,
    output_file: str = "../output/results/labor_optimization_results.csv",
    detailed_output_file: str = "../output/results/contract_fulfillment_detailed.csv"
) -> dict:
    """
    Optimize labor contract assignments to minimize total cost while meeting demand.

    Args:
        demand_file: Path to CSV file with columns: Year_WW, Year, Week, Demand
        output_file: Path to save optimization results summary
        detailed_output_file: Path to save detailed contract fulfillment mapping

    Returns:
        Dictionary with optimization results including total cost, status, and assignments
    """
    # Load demand data
    demand_df = pd.read_csv(demand_file)
    weeks = demand_df['Year_WW'].tolist()
    demand = demand_df['Demand'].tolist()
    n_weeks = len(weeks)

    print(f"Loaded {n_weeks} weeks of demand data")
    print(f"Total demand: {sum(demand)} labor-weeks")
    print(f"Average weekly demand: {sum(demand)/n_weeks:.1f}")

    # Create optimization problem
    prob = LpProblem("Labor_Contract_Optimization", LpMinimize)

    # Decision variables: number of contracts of type d starting at week t
    # x[t][d] = number of contracts with duration d starting at week t
    x = {}
    for t in range(n_weeks):
        x[t] = {}
        for duration in CONTRACT_TYPES.keys():
            x[t][duration] = LpVariable(
                f"contract_w{t}_d{duration}",
                lowBound=0,
                cat='Integer'
            )

    # Objective: minimize total cost
    total_cost = lpSum([
        x[t][d] * CONTRACT_TYPES[d]['cost']
        for t in range(n_weeks)
        for d in CONTRACT_TYPES.keys()
    ])
    prob += total_cost

    # Constraints: supply must meet or exceed demand for each week
    for t in range(n_weeks):
        # Calculate supply at week t from all active contracts
        supply_at_t = lpSum([
            x[start_week][duration]
            for start_week in range(n_weeks)
            for duration in CONTRACT_TYPES.keys()
            # Contract is active at week t if it started at start_week and hasn't expired
            if start_week <= t < min(start_week + duration, n_weeks)
        ])

        # Supply must meet demand
        prob += supply_at_t >= demand[t], f"demand_week_{t}"

    print("\nSolving optimization problem...")
    print(f"Variables: {len(x) * len(CONTRACT_TYPES)}")
    print(f"Constraints: {n_weeks}")

    # Solve the problem
    prob.solve()

    # Extract results
    status = LpStatus[prob.status]
    print(f"\nOptimization Status: {status}")

    if status != "Optimal":
        print("Warning: Optimal solution not found!")
        return {
            'status': status,
            'total_cost': None,
            'assignments': None
        }

    total_cost_value = value(prob.objective)
    print(f"Total Cost: ${total_cost_value:,.2f}")

    # Extract contract assignments with individual contract IDs
    assignments = []
    contract_id = 1
    for t in range(n_weeks):
        for duration in CONTRACT_TYPES.keys():
            num_contracts = value(x[t][duration])
            if num_contracts > 0:
                for c in range(int(num_contracts)):
                    assignments.append({
                        'Contract_ID': f"C{contract_id:04d}",
                        'Start_Week': weeks[t],
                        'Duration': duration,
                        'Cost': CONTRACT_TYPES[duration]['cost'],
                        'End_Week': weeks[min(t + duration - 1, n_weeks - 1)]
                    })
                    contract_id += 1

    # Create results dataframe
    results_df = pd.DataFrame(assignments)

    if not results_df.empty:
        results_df = results_df.sort_values('Start_Week')

        # Save detailed contract list
        results_df.to_csv(output_file, index=False)
        print(f"\nContract list saved to: {output_file}")

        # Create detailed fulfillment mapping
        print("\nGenerating detailed contract fulfillment mapping...")
        fulfillment_data = []

        for idx, row in demand_df.iterrows():
            week_idx = idx
            year_ww = row['Year_WW']
            week_demand = row['Demand']

            # Find all contracts active during this week
            active_contracts = []
            for _, contract in results_df.iterrows():
                start_idx = weeks.index(contract['Start_Week'])
                end_idx = weeks.index(contract['End_Week'])

                if start_idx <= week_idx <= end_idx:
                    active_contracts.append({
                        'id': contract['Contract_ID'],
                        'duration': contract['Duration'],
                        'start': contract['Start_Week'],
                        'end': contract['End_Week']
                    })

            # Count contracts by type
            count_1wk = sum(1 for c in active_contracts if c['duration'] == 1)
            count_13wk = sum(1 for c in active_contracts if c['duration'] == 13)
            count_26wk = sum(1 for c in active_contracts if c['duration'] == 26)
            count_52wk = sum(1 for c in active_contracts if c['duration'] == 52)

            total_supply = len(active_contracts)
            surplus = total_supply - week_demand

            # Create contract ID lists by type
            ids_1wk = ','.join([c['id'] for c in active_contracts if c['duration'] == 1]) or 'None'
            ids_13wk = ','.join([c['id'] for c in active_contracts if c['duration'] == 13]) or 'None'
            ids_26wk = ','.join([c['id'] for c in active_contracts if c['duration'] == 26]) or 'None'
            ids_52wk = ','.join([c['id'] for c in active_contracts if c['duration'] == 52]) or 'None'

            fulfillment_data.append({
                'Year_WW': year_ww,
                'Demand': week_demand,
                'Total_Supply': total_supply,
                'Surplus': surplus,
                'Count_1wk': count_1wk,
                'Count_13wk': count_13wk,
                'Count_26wk': count_26wk,
                'Count_52wk': count_52wk,
                'Contract_IDs_1wk': ids_1wk,
                'Contract_IDs_13wk': ids_13wk,
                'Contract_IDs_26wk': ids_26wk,
                'Contract_IDs_52wk': ids_52wk
            })

        fulfillment_df = pd.DataFrame(fulfillment_data)
        fulfillment_df.to_csv(detailed_output_file, index=False)
        print(f"Detailed fulfillment mapping saved to: {detailed_output_file}")

        # Print summary statistics
        print("\nContract Summary:")
        print("-" * 60)
        contract_summary = results_df.groupby('Duration').agg({
            'Contract_ID': 'count',
            'Cost': 'sum'
        }).reset_index()
        contract_summary.columns = ['Duration', 'Num_Contracts', 'Total_Cost']
        print(contract_summary.to_string(index=False))

        print("\n" + "=" * 60)
        print(f"TOTAL COST: ${total_cost_value:,.2f}")
        print("=" * 60)

        # Calculate average cost per labor-week
        total_labor_weeks = sum(demand)
        if total_labor_weeks > 0:
            avg_cost_per_labor_week = total_cost_value / total_labor_weeks
            print(f"Average cost per labor-week: ${avg_cost_per_labor_week:.2f}")
    else:
        print("\nNo contracts needed (all demand is zero)")
        fulfillment_df = None

    # Verify solution meets demand
    print("\nVerifying solution...")
    supply_by_week = [0] * n_weeks
    for _, contract in results_df.iterrows():
        start_idx = weeks.index(contract['Start_Week'])
        end_idx = weeks.index(contract['End_Week'])

        for w in range(start_idx, end_idx + 1):
            supply_by_week[w] += 1

    # Check for any shortfalls
    shortfalls = []
    for t in range(n_weeks):
        if supply_by_week[t] < demand[t]:
            shortfalls.append((weeks[t], demand[t], supply_by_week[t]))

    if shortfalls:
        print("WARNING: Demand not met for the following weeks:")
        for wk, dem, sup in shortfalls:
            print(f"  Week {wk}: Demand={dem}, Supply={sup}")
    else:
        print("✓ All demand constraints satisfied")

    return {
        'status': status,
        'total_cost': total_cost_value,
        'assignments': results_df if not results_df.empty else None,
        'fulfillment': fulfillment_df,
        'supply_by_week': supply_by_week,
        'demand_by_week': demand
    }


if __name__ == "__main__":
    # Example usage
    print("Labor Contract Optimization")
    print("=" * 60)

    result = optimize_labor_contracts(
        demand_file="../data/raw/labor_demand_2025_2030.csv",
        output_file="../output/results/labor_optimization_results.csv",
        detailed_output_file="../output/results/contract_fulfillment_detailed.csv"
    )

    if result['status'] == 'Optimal':
        print("\n✓ Optimization completed successfully!")
    else:
        print(f"\n✗ Optimization failed with status: {result['status']}")
