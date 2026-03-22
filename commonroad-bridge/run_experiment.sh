#!/usr/bin/env bash
# =============================================================================
# AdaFrenetic + CommonRoad Bridge — Experiment Runner
# =============================================================================
# Edit the CONFIGURATION section below, then run:
#   chmod +x run_experiment.sh
#   ./run_experiment.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADAFRENETIC_DIR="$SCRIPT_DIR/../adafrenetic-sbst22"

# ===== CONFIGURATION =========================================================
# Edit these values to customize your experiment

TIME_BUDGET=500              # AdaFrenetic generation time (seconds)
MAP_SIZE=1000                  # Map size (meters)

# Scenario settings for the CommonRoad ST simulation
SPEED=65                     # Vehicle speed (km/h)
OOB_TOLERANCE=0.75           # OBE tolerance (0.0 - 1.0)

# ==============================================================================

# Create per-run output folder
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$SCRIPT_DIR/results/run_${TIMESTAMP}_v${SPEED}_t${OOB_TOLERANCE}"
mkdir -p "$RUN_DIR"

echo "============================================="
echo "  Step 1: Patching AdaFrenetic (if needed)"
echo "============================================="
bash "$SCRIPT_DIR/patch_adafrenetic.sh" "$ADAFRENETIC_DIR"

echo ""
echo "============================================="
echo "  Step 2: Running AdaFrenetic (CommonRoad executor)"
echo "============================================="
echo "  Speed: ${SPEED} km/h | OOB tolerance: ${OOB_TOLERANCE}"
echo "  Time budget: ${TIME_BUDGET}s (random: ~${TIME_BUDGET}*20%, then mutation/evolution)"
echo "  Output: $RUN_DIR"
echo ""

# Run AdaFrenetic with CommonRoad executor — real physics feedback loop
cd "$ADAFRENETIC_DIR"
source venv/bin/activate

python competition.py \
    --time-budget "$TIME_BUDGET" \
    --executor commonroad \
    --map-size "$MAP_SIZE" \
    --module-path src \
    --module-name generators.adaptive_random_frenet_generator \
    --class-name AdaFrenetic \
    --speed-limit "$SPEED" \
    --oob-tolerance "$OOB_TOLERANCE" 2>&1 | tee adafrenetic_output.log

deactivate

# Copy AdaFrenetic log and experiment CSV into run folder
cp adafrenetic_output.log "$RUN_DIR/"
cp experiments/*-results.csv "$RUN_DIR/" 2>/dev/null || true

echo ""
echo "============================================="
echo "  Step 3: Analysis + CSV + plots (from log)"
echo "============================================="
echo ""

# Parse stats from log + simulate only the worst road for its plot
cd "$SCRIPT_DIR"
source venv/bin/activate

python bridge_simulator.py \
    --adafrenetic-dir "$ADAFRENETIC_DIR" \
    --speed "$SPEED" \
    --oob-tolerance "$OOB_TOLERANCE" \
    --map-size "$MAP_SIZE" \
    --output-dir "$RUN_DIR"

deactivate

echo ""
echo "============================================="
echo "  Done!"
echo "  All results in: $RUN_DIR"
echo "============================================="
