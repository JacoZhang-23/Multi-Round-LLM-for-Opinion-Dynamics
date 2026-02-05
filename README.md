# Multi-Round LLM for Opinion Dynamics

## Overview
This project implements a multi-agent simulation where LLM-powered agents engage in multi-round dialogues about vaccination attitudes. Agents interact through a social network, exchange opinions, and update their opinions based on conversations. The system supports batch simulations with aggregated statistics.

## Inputs
### Required Data Files
Located under `data/input/`:
- `workplace_36013030400w1_extended_population.csv` - Agent population data
- `workplace_36013030400w1_extended_network.csv` - Social network structure

### Configuration
Edit parameters in `config.py`:

**Key Settings:**
- `MAX_STEPS`: Number of simulation steps per run
- `BATCH_RUNS`: Number of runs for aggregation
- `MAX_DIALOGS_PER_MICROSTEP`: Concurrent dialogue limit
- `OPINION_DISTRIBUTION_TYPE`: Initial opinion distribution (neutral/pro/resist)
- `API_URL`, `API_KEY`, `MODEL_NAME`: LLM API configuration

## Multi-Round Dialogue System

### Dialogue Structure
Each conversation consists of **4 rounds** of exchanges:
1. **Round 1**: Person B initiates the conversation (2-3 sentences)
2. **Rounds 2-4**: Alternating responses between Person A and B (2-3 sentences each)
3. **Reflection**: Person A reflects on the conversation and reports updated opinion

### Core Prompts

**Initial Message (Person B):**
```
You are Person B having a conversation about vaccination. 
Your background: {profile}
Your current attitude: You {attitude_description}

Start a brief conversation with Person A about vaccination (2-3 sentences).
```

**Subsequent Responses:**
```
You are Person A/B responding/continuing.
Your background: {profile}
Your current view: You {attitude_description}

Respond naturally to what was said (2-3 sentences).
```

**Opinion Elicitation:**
```
After this conversation, reflect on your current view about vaccination.

Provide a JSON object with:
{
  "summary_sentence": "One sentence describing your current view",
  "opinion_score": <number between -1.0 (strongly oppose) and 1.0 (strongly support)>
}

Your view BEFORE: You {attitude_description} (score: {opinion_score})
```

## Outputs
Output directory: `data/output/simulation_<timestamp>/`

**Single Run (BATCH_RUNS = 1):**
- `run_01/` - All simulation outputs
- `model_data_mean.csv` - Model metrics (same as run_01)
- `model_data_std.csv` - Standard deviation (mostly zeros)

**Batch Runs (BATCH_RUNS > 1):**
- `run_01/` ... `run_NN/` - Individual run outputs
- `model_data_mean.csv` - Mean across all runs
- `model_data_std.csv` - Standard deviation across runs
- `visualizations/` - Aggregated visualizations

### Per-Run Files (inside each `run_XX/`)
- `model_data.csv` - Step-by-step metrics (vaccination rate, average opinions, etc.)
- `agent_data.csv` - Per-agent metrics by step
- `agent_profiles.json` - Agent profiles and opinion histories
- `all_dialogues.json` - Complete dialogue logs
- `most_impactful_dialogues_report.txt` - Top influential dialogues
- `network_data.json` - Network topology
- `visualizations/` - Charts and network evolution frames

### Key Metrics
- `Vaccination_Rate` - Proportion of vaccinated agents
- `Average_Opinion_LLM` - Mean opinion from LLM elicitation
- `Opinion_Std_Dev_LLM` - Opinion standard deviation

## Usage

**Run Single Simulation:**
```bash
python main.py
```

**Configure Batch Runs:**
Edit `config.py`:
```python
BATCH_RUNS = 10  # Number of simulation runs
MAX_STEPS = 20   # Steps per run
```

## Project Structure
```
src_v1/
├── main.py                    # Entry point
├── model.py                   # Simulation model
├── agent.py                   # Agent with dialogue logic
├── tools.py                   # Utility functions
├── config.py                  # Configuration
├── analysis.py                # Data analysis
├── visualize_batch_results.py # Visualization
├── requirements.txt           # Dependencies
└── data/
    └── input/                 # Input CSV files
```

## Requirements
- Python 3.8+
- See `requirements.txt` for dependencies

Install dependencies:
```bash
pip install -r requirements.txt
```

## Example: Most Impactful Dialogue

Below is an example of the most influential dialogue from a simulation run, ranked by the magnitude of opinion change. The system automatically identifies and logs these high-impact conversations.

```
MOST IMPACTFUL DIALOGUES (by magnitude of opinion change)
================================================================================

┌──────────────────────────────────────────────────────────────────────────────┐
│ DIALOGUE #1 - Strong ↓ Negative Influence                                   │
└──────────────────────────────────────────────────────────────────────────────┘

⏱  Simulation Tick: 0
👥 Participants:
   • Agent 12 (Listener/Influenced)
   • Agent 15 (Speaker/Influencer)

📈 Opinion Metrics:
   • Initial Opinion (Self):      0.518
   • Initial Opinion (Neighbor): -0.258
   • Final Opinion (Self):       -0.500
   • Opinion Change:             -1.018 (↓ Negative)
   • Change Magnitude:           1.018

💬 DIALOGUE TRANSCRIPT
--------------------------------------------------------------------------------
[Turn 1] Agent 15 (Speaker):
Hey, I've been thinking a lot about vaccines lately. I get why they're important, 
but there's so much conflicting info online—I'm just not sure what to trust. 
What's your take?

[Turn 2] Agent 12 (Listener):
I totally get the confusion—there's a lot of noise out there. But as a nurse, 
I've seen firsthand how vaccines protect not just individuals but whole communities. 
They're rigorously tested and updated based on real-world data, which helps us 
stay safe and healthy, especially for kids.

[Turn 3] Agent 15 (Speaker):
I hear you—there's definitely a lot of mixed messages. Some studies seem to 
contradict others, and I'm worried about long-term effects I can't fully understand. 
It's hard to know what to believe when so much of the info feels unreliable.

[Turn 4] Agent 12 (Listener):
I get that it's overwhelming—science can feel confusing when so much of it's out there. 
But vaccines go through strict testing and continuous monitoring, and the data on their 
safety and effectiveness is pretty clear, especially for things like measles or 
whooping cough. For your kids, the benefits of protection far outweigh the risks, 
even with new info coming in.

💭 POST-DIALOGUE REFLECTION
--------------------------------------------------------------------------------
Agent 12's Summary: "I'm still cautious about vaccines due to conflicting information, 
but I appreciate the effort to address concerns, though I remain skeptical."
Self-Reported Score: -0.500
```
