import os
from Melodie import Config

# API Settings
API_KEY = "abc123"

# API URL Options (uncomment one):
# Option 1: Remote server (original)
# API_URL = "http://10.13.12.164:7899/v1"
# Option 2: Local server (localhost)
API_URL = "http://localhost:7899/v1"
# Option 3: Local server (0.0.0.0)
# API_URL = "http://0.0.0.0:7899/v1"

CHAT_ENDPOINT = f"{API_URL}/chat/completions"
MODEL_NAME = "Qwen/Qwen3-8B"

# Concurrency Settings
MAX_DIALOGS_PER_MICROSTEP = 18  # Maximum concurrent dialogs (as per requirements)
MAX_CONCURRENT_CALLS = 50  # Number of concurrent API calls for other operations

# Simulation Parameters
# Note: Using extended workplace network with 95 agents (30 workplace + 65 external)
MAX_STEPS = 10
AGENT_ALPHA = 0.5
BATCH_RUNS = 10  # Number of times to run the entire simulation for statistical robustness

# Belief Distribution Settings (Normal)
# Options: "neutral", "pro", "resist"
BELIEF_DISTRIBUTION_TYPE = "neutral"
BELIEF_MEANS = {
    "neutral": 0.0,
    "pro": 0.4,
    "resist": -0.4,
}
BELIEF_STD = 0.3

config = Config(
    project_name="LLMIP_basic",
    project_root=os.path.dirname(__file__),
    input_folder="data/input",
    output_folder="data/output",
)
