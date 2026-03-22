#!/usr/bin/env bash
# =============================================================================
# Patches for AdaFrenetic to work with mock executor on macOS
# =============================================================================
# Fixes:
#   1. base_generator.py: crash when simulations/beamng_executor doesn't exist
#   2. base_generator.py: df.append deprecated in pandas 2.0
#   3. tests_evaluation.py: np.NaN removed in numpy 2.0
#   4. executors.py: mock executor sleeps 5s per test (too slow)
#   5. adaptive_random_frenet_generator.py: unnecessary sleep(10) + df.at pandas 2.0 fix
#   6. Copy commonroad_executor.py into code_pipeline/
#   7. competition.py: add 'commonroad' executor option
#   8. adaptive_random_frenet_generator.py: fix random_budget to 20% of time budget
#
# Usage: bash patch_adafrenetic.sh /path/to/adafrenetic-sbst22
# =============================================================================

set -e

ADAFRENETIC_DIR="${1:?Usage: bash patch_adafrenetic.sh /path/to/adafrenetic-sbst22}"

if [ ! -f "$ADAFRENETIC_DIR/competition.py" ]; then
    echo "ERROR: $ADAFRENETIC_DIR does not look like an AdaFrenetic repo"
    exit 1
fi

MARKER="$ADAFRENETIC_DIR/.patched"
if [ -f "$MARKER" ]; then
    echo "  Patches already applied (found $MARKER)"
    exit 0
fi

echo "  Applying patches to AdaFrenetic..."

# --- Patch 1: base_generator.py — use Python for multi-line patch ---
python3 -c "
import sys
path = sys.argv[1] + '/src/generators/base_generator.py'
with open(path, 'r') as f:
    content = f.read()

# Add 'import pandas as pd' after first import line
if 'import pandas as pd' not in content:
    content = 'import pandas as pd\n' + content

# Wrap simulation_file in try/except
old = '''            # Retrieving file name
            last_file = sorted(Path('simulations/beamng_executor').iterdir(), key=os.path.getmtime)[-1]
            info['simulation_file'] = last_file.name'''
new = '''            # Retrieving file name
            try:
                last_file = sorted(Path('simulations/beamng_executor').iterdir(), key=os.path.getmtime)[-1]
                info['simulation_file'] = last_file.name
            except (FileNotFoundError, IndexError):
                info['simulation_file'] = None'''
content = content.replace(old, new)

# Replace deprecated df.append with pd.concat
content = content.replace(
    'self.df = self.df.append(info, ignore_index=True)',
    'self.df = pd.concat([self.df, pd.DataFrame([info])], ignore_index=True)'
)

with open(path, 'w') as f:
    f.write(content)
print('  Patched base_generator.py')
" "$ADAFRENETIC_DIR"

# --- Patch 2: tests_evaluation.py — np.NaN → np.nan + degenerate segment crash ---
sed -i '' 's/np\.NaN/np.nan/g' "$ADAFRENETIC_DIR/code_pipeline/tests_evaluation.py"
python3 -c "
import sys
path = sys.argv[1] + '/code_pipeline/tests_evaluation.py'
with open(path, 'r') as f:
    content = f.read()
old = '''    def _compute_sparseness(self):
        # Compute distance among the OOB and take the avg of their maximum distance
        max_distances_starting_from = {}

        for (oob1, oob2) in combinations(self.oobs, 2):
            # Compute distance between cells
            distance = iterative_levenshtein(oob1['interesting segment'], oob2['interesting segment'])'''
new = '''    def _compute_sparseness(self):
        # Compute distance among the OOB and take the avg of their maximum distance
        max_distances_starting_from = {}

        # Sample if too many OOBs — O(n^2) pairwise Levenshtein is too slow for hundreds
        import random as _rnd
        oobs_for_sparseness = self.oobs
        if len(self.oobs) > 100:
            self.logger.info("Sampling 100 of %d OOBs for sparseness (full set too slow)", len(self.oobs))
            oobs_for_sparseness = _rnd.sample(self.oobs, 100)

        for (oob1, oob2) in combinations(oobs_for_sparseness, 2):
            # Skip degenerate segments (need >= 2 points for distance calculation)
            if len(oob1['interesting segment']) < 2 or len(oob2['interesting segment']) < 2:
                continue
            # Compute distance between cells
            distance = iterative_levenshtein(oob1['interesting segment'], oob2['interesting segment'])'''
content = content.replace(old, new)
with open(path, 'w') as f:
    f.write(content)
print('  Patched sparseness degenerate segment crash')
" "$ADAFRENETIC_DIR"
echo "  Patched tests_evaluation.py"

# --- Patch 3: executors.py — reduce mock sleep ---
sed -i '' 's/time\.sleep(5)/time.sleep(0.1)/' "$ADAFRENETIC_DIR/code_pipeline/executors.py"
echo "  Patched executors.py"

# --- Patch 4: adaptive_random_frenet_generator.py — reduce sleep + fix pandas .at → .loc ---
sed -i '' 's/sleep(10)/sleep(1)/' "$ADAFRENETIC_DIR/src/generators/adaptive_random_frenet_generator.py"
sed -i '' 's/self\.df\.at\[parent\.index/self.df.loc[parent.index/' "$ADAFRENETIC_DIR/src/generators/adaptive_random_frenet_generator.py"
echo "  Patched adaptive_random_frenet_generator.py"

# --- Patch 5: AdaFrenetic random_budget — use 20% of time budget, not fixed 3600s ---
python3 -c "
import sys
path = sys.argv[1] + '/src/generators/adaptive_random_frenet_generator.py'
with open(path, 'r') as f:
    content = f.read()

old = '''class AdaFrenetic(CustomFrenetGenerator):
    def __init__(self, time_budget=None, executor=None, map_size=None):
        super().__init__(executor=executor, map_size=map_size,
                         kill_ancestors=0, strict_father=False, random_budget=3600,
                         crossover_candidates=20, crossover_frequency=40)'''
new = '''class AdaFrenetic(CustomFrenetGenerator):
    def __init__(self, time_budget=None, executor=None, map_size=None):
        # Spend 20% of time on random generation, then switch to mutation/evolution
        budget = 3600
        if executor and hasattr(executor, \"time_budget\") and executor.time_budget.time_budget:
            budget = int(executor.time_budget.time_budget * 0.2)
        super().__init__(executor=executor, map_size=map_size,
                         kill_ancestors=0, strict_father=False, random_budget=budget,
                         crossover_candidates=20, crossover_frequency=40)'''
content = content.replace(old, new)

with open(path, 'w') as f:
    f.write(content)
print('  Patched AdaFrenetic random_budget')
" "$ADAFRENETIC_DIR"

# --- Patch 6: Copy commonroad_executor.py ---

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SCRIPT_DIR/commonroad_executor.py" ]; then
    cp "$SCRIPT_DIR/commonroad_executor.py" "$ADAFRENETIC_DIR/code_pipeline/commonroad_executor.py"
    echo "  Copied commonroad_executor.py"
fi

# --- Patch 6: competition.py — add 'commonroad' executor ---
python3 -c "
import sys
path = sys.argv[1] + '/competition.py'
with open(path, 'r') as f:
    content = f.read()

# Add 'commonroad' to executor choices
old_choices = \"'mock', 'beamng', 'dave2']\"
new_choices = \"'mock', 'beamng', 'dave2', 'commonroad']\"
content = content.replace(old_choices, new_choices)

# Add CommonRoad executor instantiation block
if 'commonroad_executor' not in content:
    old_block = '''    # Register the shutdown hook for post processing results'''
    new_block = '''    elif executor == \"commonroad\":
        from code_pipeline.commonroad_executor import CommonRoadExecutor
        the_executor = CommonRoadExecutor(result_folder, map_size,
                                          generation_budget=generation_budget,
                                          execution_budget=execution_budget,
                                          time_budget=time_budget,
                                          oob_tolerance=oob_tolerance,
                                          max_speed_in_kmh=speed_limit,
                                          road_visualizer=road_visualizer)

    # Register the shutdown hook for post processing results'''
    content = content.replace(old_block, new_block, 1)

with open(path, 'w') as f:
    f.write(content)
print('  Patched competition.py')
" "$ADAFRENETIC_DIR"

# Create required directories
mkdir -p "$ADAFRENETIC_DIR/experiments" "$ADAFRENETIC_DIR/simulations/beamng_executor" "$ADAFRENETIC_DIR/results"

touch "$MARKER"
echo "  All patches applied successfully."
