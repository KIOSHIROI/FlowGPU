# Optional state invariants

The generated host guards cover declarations; the audit sheet flags state the
host does not check. `flowgpu.compiler.runtime.invariants` turns those flags
into optional checks:

```python
from flowgpu.compiler.runtime.invariants import (
    check_cache_position, check_mask, check_valid)

check_cache_position(loc, expected=next_slot, slots=32)
check_valid(valid, slots=32)
check_mask(mask, valid=valid)
```

- `check_cache_position`: rejects out-of-range and skipped positions;
- `check_valid`: rejects count outside `1..slots`;
- `check_mask`: first `valid` entries must be 0, the rest <= `-1e8` (matches
  the documented mask convention).

They are opt-in and separate from generated code: a caller that has already
proven these facts pays no runtime cost. Applied to the Phase-2 matrix, the
five observable semantic faults (missing/short/zero mask, skipped/duplicate
position) are all rejected, closing the gap the guards leave.

## Boundary

These are host-side assertions, not compiler proofs. They can be skipped,
misused, or called with the wrong `expected`; the audit sheet's risk flags
remain the honest statement of what is not checked automatically.
