# FHRR Cross-Space Alignment Experiments

Reproducible tests for measuring `k_min` (minimum anchors for alignment) in FHRR unitary alignment.

## Structure

```
engine/
├── active_inference_agent.py    # FHRR agent with Phase Diff Decode operator
├── cross_space_alignment.py     # Alignment framework
└── alignment_budget_analysis.py # Information budget experiments
```

## Tests

### 1. Phase Diff Decode Operator
Verifies that the decode_action method correctly computes angular distance on S^1 with wraparound handling.

### 2. Cross-Space Alignment
Measures `k_min` for anchor-based alignment between two random FHRR spaces.

### 3. Information Budget
Tests whether features can reduce `k_min` below dimension `d`.

## Key Questions Being Tested

1. **Identifiability**: Can we distinguish correct correspondence from false ones?
2. **Alignment Budget**: What is minimum `k` (anchors) for reliable alignment?
3. **Transfer**: Does knowledge transfer after alignment?

## Running Locally

```bash
pip install numpy
python -m engine.active_inference_agent
python -m engine.cross_space_alignment
python -m engine.alignment_budget_analysis
```

## Expected Outputs

### Test 1: Phase Diff Decode
- Simple phase difference: ✓
- Wraparound case: ✓ (linear subtraction fails, wrapped diff works)
- Action application: ✓

### Test 2: Cross-Space Alignment
- k_min = d for anchor-based alignment
- Transfer error = 0 when alignment succeeds

### Test 3: Feature Overlap
- Critical overlap threshold for feature-based alignment

## Caveats

These are **toy experiments** with specific assumptions:
- Random unitary transforms
- Ideal feature/topology conditions
- Procrustes-based alignment

Results should not be generalized without verification in target environment.
