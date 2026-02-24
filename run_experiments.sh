#!/bin/bash
# =============================================================================
# COSMOS Framework - Multi-Environment Experiment Runner
# =============================================================================
# Runs experiments across multiple environments to validate the method
#
# Usage:
#   chmod +x run_experiments.sh
#   ./run_experiments.sh [experiment_name]
#
# Examples:
#   ./run_experiments.sh              # Run all experiments
#   ./run_experiments.sh quick        # Quick test (few episodes)
#   ./run_experiments.sh formation    # Formation environment only
#   ./run_experiments.sh safety       # Safety comparison only
#   ./run_experiments.sh ablation     # Ablation study
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
EXPERIMENT_NAME=${1:-"all"}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="experiments/${TIMESTAMP}"
NUM_SEEDS=3
USE_WANDB=false

# Training parameters
QUICK_EPISODES=50
SHORT_EPISODES=200
FULL_EPISODES=1000

echo -e "${BLUE}"
echo "============================================================"
echo "  COSMOS - Multi-Environment Experiments"
echo "============================================================"
echo -e "${NC}"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Timestamp:  ${TIMESTAMP}"
echo ""

# Create output directory
mkdir -p ${OUTPUT_DIR}/logs
mkdir -p ${OUTPUT_DIR}/results
mkdir -p ${OUTPUT_DIR}/checkpoints

# =============================================================================
# Helper Functions
# =============================================================================

run_training() {
    local env=$1
    local algo=$2
    local safety=$3
    local episodes=$4
    local seed=$5
    local extra_args=${6:-""}

    local exp_name="${env}_${algo}_${safety}_seed${seed}"
    local log_file="${OUTPUT_DIR}/logs/${exp_name}.log"

    echo -e "${CYAN}Running: ${exp_name}${NC}"

    python -m cosmos.train \
        env=${env} \
        algo=${algo} \
        safety=${safety} \
        experiment.num_episodes=${episodes} \
        experiment.seed=${seed} \
        experiment.name=${exp_name} \
        logging.use_wandb=${USE_WANDB} \
        hydra.run.dir=${OUTPUT_DIR}/hydra/${exp_name} \
        ${extra_args} \
        2>&1 | tee ${log_file}

    echo -e "${GREEN}✓ Completed: ${exp_name}${NC}"
}

run_multi_seed() {
    local env=$1
    local algo=$2
    local safety=$3
    local episodes=$4
    local extra_args=${5:-""}

    for seed in $(seq 1 ${NUM_SEEDS}); do
        run_training ${env} ${algo} ${safety} ${episodes} ${seed} "${extra_args}"
    done
}

# =============================================================================
# Experiment 1: Quick Test (验证安装)
# =============================================================================
run_quick_test() {
    echo -e "${YELLOW}"
    echo "============================================================"
    echo "  Quick Test - Verifying Installation"
    echo "============================================================"
    echo -e "${NC}"

    # Test each environment with short runs
    run_training formation_nav mappo cosmos ${QUICK_EPISODES} 1
    run_training epuck_sim mappo cbf ${QUICK_EPISODES} 1

    # Test Safety-Gym if available
    python -c "import safety_gymnasium" 2>/dev/null && \
        run_training safety_gym mappo cbf ${QUICK_EPISODES} 1 \
            "env.env_id=SafetyPointGoal1-v0" || \
        echo "Skipping Safety-Gym (not installed)"

    # Test VMAS if available
    python -c "import vmas" 2>/dev/null && \
        run_training vmas mappo none ${QUICK_EPISODES} 1 || \
        echo "Skipping VMAS (not installed)"

    echo -e "${GREEN}Quick test completed!${NC}"
}

# =============================================================================
# Experiment 2: Formation Control (主实验)
# =============================================================================
run_formation_experiments() {
    echo -e "${YELLOW}"
    echo "============================================================"
    echo "  Formation Control Experiments"
    echo "============================================================"
    echo -e "${NC}"

    local episodes=${FULL_EPISODES}

    # Safety method comparison
    echo -e "${CYAN}[1/3] Safety Method Comparison${NC}"
    run_multi_seed formation_nav mappo cosmos ${episodes}
    run_multi_seed formation_nav mappo cbf ${episodes}
    run_multi_seed formation_nav mappo none ${episodes}

    # Algorithm comparison
    echo -e "${CYAN}[2/3] Algorithm Comparison${NC}"
    run_multi_seed formation_nav mappo cosmos ${episodes}
    run_multi_seed formation_nav qmix cosmos ${episodes}
    run_multi_seed formation_nav maddpg cosmos ${episodes}

    # Scalability (number of agents)
    echo -e "${CYAN}[3/3] Scalability Test${NC}"
    for n_agents in 4 6 8; do
        run_training formation_nav mappo cosmos ${episodes} 1 \
            "env.num_agents=${n_agents}"
    done

    echo -e "${GREEN}Formation experiments completed!${NC}"
}

# =============================================================================
# Experiment 3: Safety Comparison (安全性对比)
# =============================================================================
run_safety_experiments() {
    echo -e "${YELLOW}"
    echo "============================================================"
    echo "  Safety Comparison Experiments"
    echo "============================================================"
    echo -e "${NC}"

    local episodes=${FULL_EPISODES}

    # Check if Safety-Gym is available
    if ! python -c "import safety_gymnasium" 2>/dev/null; then
        echo -e "${RED}Safety-Gymnasium not installed. Skipping...${NC}"
        echo "Install with: pip install safety-gymnasium"
        return
    fi

    # Different Safety-Gym environments
    echo -e "${CYAN}[1/2] Safety-Gym Environments${NC}"
    for env_id in "SafetyPointGoal1-v0" "SafetyCarGoal1-v0"; do
        run_multi_seed safety_gym mappo cbf ${episodes} \
            "env.env_id=${env_id}"
        run_multi_seed safety_gym mappo none ${episodes} \
            "env.env_id=${env_id}"
    done

    # E-puck simulation
    echo -e "${CYAN}[2/2] E-puck Simulation${NC}"
    run_multi_seed epuck_sim mappo cosmos ${episodes}
    run_multi_seed epuck_sim mappo cbf ${episodes}
    run_multi_seed epuck_sim mappo none ${episodes}

    echo -e "${GREEN}Safety experiments completed!${NC}"
}

# =============================================================================
# Experiment 4: Ablation Study (消融实验)
# =============================================================================
run_ablation_experiments() {
    echo -e "${YELLOW}"
    echo "============================================================"
    echo "  Ablation Study"
    echo "============================================================"
    echo -e "${NC}"

    local episodes=${SHORT_EPISODES}

    # COSMOS components ablation
    echo -e "${CYAN}[1/2] COSMOS Components${NC}"

    # Full COSMOS
    run_multi_seed formation_nav mappo cosmos ${episodes}

    # CBF only (no manifold)
    run_multi_seed formation_nav mappo cbf ${episodes}

    # No safety
    run_multi_seed formation_nav mappo none ${episodes}

    # Hyperparameter sensitivity
    echo -e "${CYAN}[2/2] Hyperparameter Sensitivity${NC}"
    for lr in 1e-4 3e-4 1e-3; do
        run_training formation_nav mappo cosmos ${episodes} 1 \
            "algo.actor_lr=${lr} algo.critic_lr=${lr}"
    done

    echo -e "${GREEN}Ablation experiments completed!${NC}"
}

# =============================================================================
# Experiment 5: VMAS Large-Scale (大规模实验)
# =============================================================================
run_vmas_experiments() {
    echo -e "${YELLOW}"
    echo "============================================================"
    echo "  VMAS Large-Scale Experiments"
    echo "============================================================"
    echo -e "${NC}"

    # Check if VMAS is available
    if ! python -c "import vmas" 2>/dev/null; then
        echo -e "${RED}VMAS not installed. Skipping...${NC}"
        echo "Install with: pip install vmas"
        return
    fi

    local episodes=${SHORT_EPISODES}

    # Different scenarios
    echo -e "${CYAN}[1/2] VMAS Scenarios${NC}"
    for scenario in "navigation" "formation_control"; do
        run_multi_seed vmas mappo none ${episodes} \
            "env.scenario=${scenario}"
    done

    # Scalability
    echo -e "${CYAN}[2/2] Agent Scalability${NC}"
    for n_agents in 4 8 16 32; do
        run_training vmas mappo none ${episodes} 1 \
            "env.num_agents=${n_agents} env.num_envs=32"
    done

    echo -e "${GREEN}VMAS experiments completed!${NC}"
}

# =============================================================================
# Main Execution
# =============================================================================

case ${EXPERIMENT_NAME} in
    "quick")
        run_quick_test
        ;;
    "formation")
        run_formation_experiments
        ;;
    "safety")
        run_safety_experiments
        ;;
    "ablation")
        run_ablation_experiments
        ;;
    "vmas")
        run_vmas_experiments
        ;;
    "all")
        run_quick_test
        run_formation_experiments
        run_safety_experiments
        run_ablation_experiments
        run_vmas_experiments
        ;;
    *)
        echo -e "${RED}Unknown experiment: ${EXPERIMENT_NAME}${NC}"
        echo ""
        echo "Available experiments:"
        echo "  quick     - Quick installation test"
        echo "  formation - Formation control experiments"
        echo "  safety    - Safety comparison experiments"
        echo "  ablation  - Ablation study"
        echo "  vmas      - VMAS large-scale experiments"
        echo "  all       - Run all experiments"
        exit 1
        ;;
esac

# =============================================================================
# Generate Summary Report
# =============================================================================
echo ""
echo -e "${YELLOW}Generating summary report...${NC}"

python << 'EOF'
import os
import json
import glob

output_dir = os.environ.get('OUTPUT_DIR', 'experiments/latest')
results = []

# Collect results from all experiments
for log_file in glob.glob(f'{output_dir}/logs/*.log'):
    exp_name = os.path.basename(log_file).replace('.log', '')
    results.append({
        'experiment': exp_name,
        'log_file': log_file,
    })

# Save summary
summary_file = f'{output_dir}/results/summary.json'
with open(summary_file, 'w') as f:
    json.dump({
        'num_experiments': len(results),
        'experiments': results,
    }, f, indent=2)

print(f'Summary saved to: {summary_file}')
print(f'Total experiments: {len(results)}')
EOF

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  All Experiments Completed!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "Results saved to: ${OUTPUT_DIR}"
echo ""
echo "To view logs:"
echo "  ls ${OUTPUT_DIR}/logs/"
echo ""
echo "To analyze results:"
echo "  python scripts/analyze_results.py ${OUTPUT_DIR}"
echo ""
