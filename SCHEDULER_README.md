# Micro-Step Dialog Scheduler - Implementation Guide

## 🎯 Overview

This implementation provides a **strict micro-step based dialog scheduler** for agent-based simulations with the following guarantees:

- ✅ **Exclusive Participation**: Each agent participates in at most ONE dialog at any time
- ✅ **Bounded Concurrency**: Maximum 3 concurrent dialogs per micro-step
- ✅ **Lock-Based Consistency**: Agents are locked during dialogs, preventing state conflicts
- ✅ **Immediate Updates**: Attitude changes are applied immediately after each dialog
- ✅ **Complete Coverage**: All network edges are eventually processed

## 🏗️ Architecture

### Key Components

#### 1. Agent Locking (`agent.py`)
```python
class VaxAgent:
    def __init__(self, ...):
        self.is_locked = False  # Lock status
        self.dialog_memory = {}  # Per-neighbor memory
```

#### 2. Micro-Step Scheduler (`model.py`)
```python
async def async_actions(self):
    # STEP 1: Get all possible edges
    all_edges = list(self.network.edges())
    
    # STEP 2: Micro-step loop
    while edges_remaining:
        # STEP 2a: Shuffle edges randomly
        random.shuffle(remaining_edges)
        
        # STEP 2b: Select non-conflicting dialogs
        for (i, j) in edges:
            if neither_locked(i, j):
                lock(i, j)
                schedule_dialog(i, j)
                if len(dialogs) >= MAX_DIALOGS_PER_MICROSTEP:
                    break
        
        # STEP 3: Execute selected dialogs in parallel
        await asyncio.gather(*dialog_tasks)
        
        # STEP 4: Commit updates and release locks
        for dialog in completed:
            update_attitudes(i, j)  # IMMEDIATE
            update_memory(i, j)
            unlock(i, j)
```

## 📊 Example Execution

For a network with 7 edges:
```
Micro-step 1: 2 dialogs [(1,2), (0,4)]  → Locked: {0,1,2,4}
Micro-step 2: 2 dialogs [(2,3), (0,1)]  → Locked: {0,1,2,3}
Micro-step 3: 2 dialogs [(0,2), (3,4)]  → Locked: {0,2,3,4}
Micro-step 4: 1 dialog  [(1,3)]         → Locked: {1,3}

Total: 4 micro-steps to process 7 edges
```

## 🔒 Invariants (Always True)

1. **No Overlap**: No two concurrent dialogs share an agent
2. **Bounded Parallelism**: ≤ 3 dialogs per micro-step
3. **State Consistency**: Locks prevent race conditions
4. **Immediate Effect**: Updates apply before next micro-step
5. **Deterministic Order**: Given same random seed, same execution

## ⚙️ Configuration

In `config.py`:
```python
MAX_DIALOGS_PER_MICROSTEP = 3  # Hard limit on concurrency
MAX_STEPS = 5                   # Number of simulation steps
AGENT_ALPHA = 0.5               # Attitude update weight
```

## 🚀 Usage

### Running the Simulation
```bash
cd src_v1
python main.py
```

### Testing the Scheduler
```bash
cd src_v1
python test_scheduler.py
```

## 📈 Performance Characteristics

For a network with **N agents** and **E edges**:

- **Micro-steps per step**: ⌈E / MAX_DIALOGS⌉ (best case) to E (worst case)
- **API calls per step**: 5 × E (each dialog needs 5 API calls)
- **Theoretical speedup**: ~3x over fully sequential execution
- **Actual speedup**: Depends on network structure

### Example: 95 Agents, 160 Edges
- **Sequential**: 160 dialogs one-by-one = 160 micro-steps
- **With scheduler (max=3)**: ~54 micro-steps (160/3 ≈ 53.3)
- **API calls**: 160 × 5 = 800 per step
- **Total for 5 steps**: 4,000 API calls

## 🔄 Comparison with Previous System

| Aspect | Old System | New System |
|--------|-----------|------------|
| Concurrency Model | 95 agents in parallel | 3 dialogs per micro-step |
| Agent Participation | All neighbors at once | One dialog at a time |
| State Updates | Batched (advance/step) | Immediate |
| Locking | Implicit (asyncio) | Explicit (is_locked) |
| Predictability | Non-deterministic order | Deterministic with seed |
| Memory | Global history | Per-neighbor memory |

## 🎓 Design Rationale

This design follows the **simulation controller pattern** where:

1. **Separation of Concerns**: 
   - Scheduler handles WHEN dialogs happen
   - Agents handle WHAT is discussed
   
2. **Exclusive Resources**:
   - Agents are mutually exclusive resources
   - Locks prevent concurrent access
   
3. **Determinism**:
   - Random shuffling with seed ensures reproducibility
   - Order effects are intentional, not bugs

4. **Scalability**:
   - Bounded parallelism prevents API overload
   - Micro-steps allow progress tracking

## 🐛 Troubleshooting

### Issue: Too slow
**Solution**: Increase `MAX_DIALOGS_PER_MICROSTEP` (but respect API limits)

### Issue: API rate limiting
**Solution**: Decrease `MAX_DIALOGS_PER_MICROSTEP` or add delays

### Issue: Non-deterministic results
**Solution**: Set random seed: `random.seed(42)` before simulation

### Issue: Agent lock never released
**Solution**: Ensure try-finally blocks around lock operations

## 📚 References

- **Discrete Event Simulation**: Time advances in discrete steps
- **Resource Allocation**: Agents as exclusive resources
- **Actor Model**: Message-passing between isolated agents
- **Lock-Free Programming**: Avoid deadlocks via ordered locking

## ✅ Validation

Run the test suite:
```bash
python test_scheduler.py  # Unit tests for scheduler logic
python final_check.py     # Integration tests
```

Expected output:
```
✅ All tests passed!
   - Lock exclusivity verified
   - Bounded concurrency verified
   - Complete edge coverage verified
```

---

**Last Updated**: 2026-01-09
**Version**: 1.0
**Author**: Simulation Controller System
