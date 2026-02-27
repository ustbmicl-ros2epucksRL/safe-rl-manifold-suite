#!/bin/bash
# IROS 2026 Experiment Runner
# Run all experiments for the paper tables

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv"
RESULTS_DIR="${SCRIPT_DIR}/results"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo -e "${GREEN}========================================"
    echo "$1"
    echo -e "========================================${NC}"
    echo ""
}

print_usage() {
    echo "IROS 2026 Experiment Runner"
    echo ""
    echo "Usage: $0 [OPTIONS] COMMAND"
    echo ""
    echo "Commands:"
    echo "  all           Run all experiments (Tables III, IV, V)"
    echo "  ablation      Run Table III ablation study"
    echo "  noise         Run Table IV noise experiments"
    echo "  ekf           Run Table V EKF comparison"
    echo "  quick         Quick test run (reduced steps/seeds)"
    echo ""
    echo "Options:"
    echo "  --env ENV     Environment: goal, circle, push (default: goal)"
    echo "  --seeds N     Number of random seeds (default: 5)"
    echo "  --steps N     Training steps (default: 50000)"
    echo "  --episodes N  Evaluation episodes (default: 50)"
    echo "  --help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 quick                    # Quick test run"
    echo "  $0 all --env goal           # Full experiments on Goal task"
    echo "  $0 ablation --seeds 3       # Ablation with 3 seeds"
    echo ""
}

# Default parameters
ENV="goal"
SEEDS=5
STEPS=50000
EPISODES=50
COMMAND=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            ENV="$2"
            shift 2
            ;;
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --episodes)
            EPISODES="$2"
            shift 2
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
        all|ablation|noise|ekf|quick)
            COMMAND="$1"
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

if [ -z "$COMMAND" ]; then
    print_usage
    exit 1
fi

# Check virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Virtual environment not found. Running setup first...${NC}"
    bash "$SCRIPT_DIR/setup.sh"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Quick test parameters
if [ "$COMMAND" == "quick" ]; then
    STEPS=5000
    EPISODES=10
    SEEDS=2
    COMMAND="all"
    echo -e "${YELLOW}Quick test mode: steps=$STEPS, episodes=$EPISODES, seeds=$SEEDS${NC}"
fi

print_header "IROS 2026 EXPERIMENTS"
echo "Environment: $ENV"
echo "Training steps: $STEPS"
echo "Evaluation episodes: $EPISODES"
echo "Random seeds: $SEEDS"
echo "Results directory: $RESULTS_DIR"

cd "$SCRIPT_DIR"

# Run experiments based on command
case $COMMAND in
    all)
        print_header "Running ALL Experiments"

        # Table III: Ablation
        print_header "Table III: Ablation Study"
        python3 scripts/run_full_experiments.py \
            --experiment ablation \
            --env "$ENV" \
            --train-steps "$STEPS" \
            --episodes "$EPISODES" \
            --seeds "$SEEDS" \
            --output "$RESULTS_DIR"

        # Table V: EKF Comparison
        print_header "Table V: EKF Comparison"
        python3 scripts/run_full_experiments.py \
            --experiment ekf_compare \
            --env "$ENV" \
            --train-steps "$STEPS" \
            --episodes "$EPISODES" \
            --seeds "$SEEDS" \
            --output "$RESULTS_DIR"

        # Table IV: Noise
        print_header "Table IV: Noise Experiments"
        python3 scripts/run_table4_noise.py \
            --env "$ENV" \
            --train-steps "$STEPS" \
            --episodes "$EPISODES" \
            --seeds "$SEEDS" \
            --output "$RESULTS_DIR"
        ;;

    ablation)
        print_header "Table III: Ablation Study"
        python3 scripts/run_full_experiments.py \
            --experiment ablation \
            --env "$ENV" \
            --train-steps "$STEPS" \
            --episodes "$EPISODES" \
            --seeds "$SEEDS" \
            --output "$RESULTS_DIR"
        ;;

    noise)
        print_header "Table IV: Noise Experiments"
        python3 scripts/run_table4_noise.py \
            --env "$ENV" \
            --train-steps "$STEPS" \
            --episodes "$EPISODES" \
            --seeds "$SEEDS" \
            --output "$RESULTS_DIR"
        ;;

    ekf)
        print_header "Table V: EKF Comparison"
        python3 scripts/run_full_experiments.py \
            --experiment ekf_compare \
            --env "$ENV" \
            --train-steps "$STEPS" \
            --episodes "$EPISODES" \
            --seeds "$SEEDS" \
            --output "$RESULTS_DIR"
        ;;
esac

print_header "EXPERIMENTS COMPLETE"
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "Result files:"
ls -la "$RESULTS_DIR"/*.json 2>/dev/null || echo "No JSON results found"
