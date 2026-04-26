# Code Surface Compliance Report
- generated_at: 2026-04-26T16:39:45

## Modified Shared Files
```
M bin/ensemble.py
 M bin/evaluate.py
 M bin/go.py
 M bin/model.py
 M bin/tune.py
 M lib/__init__.py
 M lib/data.py
 M lib/deep.py
 M lib/env.py
 M lib/metrics.py
 M lib/util.py
?? esam/
?? outputs/
?? personb/
?? results/
```

## Compliance Assessment
- `paper/lib/data.py`: modified (needs constrained patch rationale or revert).
- `paper/bin/model.py`: modified (allowed for minimal ESAM wiring, requires review).
- `paper/lib/metrics.py`: modified (manifesto disallows scoring function changes, requires revert/justification).
- `paper/bin/evaluate.py`: modified (manifesto disallows seed loop changes, requires revert/justification).
- `paper/lib/deep.py`: modified (shared core touched; requires strict justification).

## Rerun Requirement
- Any changes to shared training/eval core after previous runs => previous runs stale for strict manifesto comparison.
