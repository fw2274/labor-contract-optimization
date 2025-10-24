#!/bin/bash

# Labor Contract Optimization Pipeline
# Runs the complete workflow: demand generation → optimization → analytics

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to print header
print_header() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
    echo ""
}

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    print_error "python3 is not installed. Please install Python 3."
    exit 1
fi

# Check if required packages are installed
print_step "Checking dependencies..."
python3 -c "import pulp, pandas, matplotlib, seaborn" 2>/dev/null || {
    print_warning "Some required packages are missing."
    print_step "Installing required packages..."
    pip install pulp pandas matplotlib seaborn
}
print_success "All dependencies installed"

# Navigate to src directory
cd "$(dirname "$0")/src" || exit 1

# Step 1: Generate demand data
print_header "STEP 1: Generating Demand Data"
print_step "Running generate_demand.py..."
python3 generate_demand.py

if [ $? -eq 0 ]; then
    print_success "Demand data generated successfully"
else
    print_error "Failed to generate demand data"
    exit 1
fi

# Step 2: Run optimization
print_header "STEP 2: Running Optimization"
print_step "Running labor_optimization.py..."
python3 labor_optimization.py

if [ $? -eq 0 ]; then
    print_success "Optimization completed successfully"
else
    print_error "Optimization failed"
    exit 1
fi

# Step 3: Generate analytics
print_header "STEP 3: Generating Analytics"
print_step "Running analytics.py..."
python3 analytics.py

if [ $? -eq 0 ]; then
    print_success "Analytics generated successfully"
else
    print_error "Analytics generation failed"
    exit 1
fi

# Print final summary
print_header "PIPELINE COMPLETED SUCCESSFULLY"

echo "Generated files:"
echo ""
echo "Data Files:"
echo "  📊 ../data/raw/labor_demand_2025_2030.csv"
echo "  📊 ../data/processed/supply_allocation.csv"
echo ""
echo "Results:"
echo "  📋 ../output/results/labor_optimization_results.csv"
echo "  📋 ../output/results/contract_fulfillment_detailed.csv"
echo ""
echo "Visualizations:"
echo "  📈 ../output/figures/demand_analysis.png"
echo "  📈 ../output/figures/supply_allocation_analysis.png"
echo ""

print_success "All steps completed successfully!"
echo ""
