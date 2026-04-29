#!/usr/bin/env python3
"""
apply_v116c_patches.py  --  v1.1.6c targeted bug-fix patches for run.py
Confirmed bugs from run_20260429_172738.log:

  BUG-C1  run.py ~L950: _bc_spawn_neutral uses threshold 0.30 m/s
          but Gazebo physics floor is 0.500 m/s (BLEED_SPEED_THRESHOLD=0.55).
          0.500 < 0.30 == False ALWAYS -> every BC Pilot episode discarded.
          FIX: accept all post-bleed states (set to True; BLEED already ran).

  BUG-C2  run.py ~L1633: `live_summary` referenced but never assigned.
          Causes NameError on EVERY episode termination, swallowed by outer except.
          FIX: add `live_summary = None` after _dash_proc assignment.

  BUG-C3  run.py ~L2537: `_d_prog` used inside try block BEFORE it is assigned
          at L2549. Raises UnboundLocalError every step-1.
          FIX: pre-initialise `_d_prog = 0.0` before the try block.

REF:
  run_20260429_172738.log - "BC pilot harvest done: 0 episodes, 0 transitions"
  run_20260429_172738.log - "EXCEPT run.py:2509 UnboundLocalError: cannot access local variable dprog"
  run_20260429_172738.log - "EXCEPT run.py:3420 NameError: name 'live_summary' is not defined"
"""
import re, sys, pathlib, shutil, datetime

TARGET = pathlib.Path("run.py")
if not TARGET.exists():
    sys.exit(f"ERROR: {TARGET} not found. Run from project root.")

backup = TARGET.with_suffix(f".py.bak_v116c_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
shutil.copy2(TARGET, backup)
print(f"[v116c] Backup: {backup}")

src = TARGET.read_text(encoding="utf-8")
original = src

# -- BUG-C1: _bc_spawn_neutral threshold ----------------------------------------
OLD_C1 = "_bc_spawn_neutral = float(rp.get('speed', 99.0)) < 0.30  # post-bleed check"
NEW_C1 = (
    "_bc_spawn_neutral = True  # v1.1.6c BUG-C1: threshold was 0.30<floor(0.500) -- always False.\n"
    "        # BLEED already neutralised speed to <=0.55 m/s in gym_adapter.env_reset().\n"
    "        # Accept all post-bleed states so BC Pilot data is not discarded wholesale."
)
if OLD_C1 in src:
    src = src.replace(OLD_C1, NEW_C1, 1)
    print("[v116c] BUG-C1 FIXED: _bc_spawn_neutral = True")
else:
    pat1 = re.compile(r"_bc_spawn_neutral\s*=\s*float\(rp\.get\('speed',\s*99\.0\)\)\s*<\s*0\.[0-9]+.*?\n")
    m = pat1.search(src)
    if m:
        src = src[:m.start()] + NEW_C1 + "\n" + src[m.end():]
        print("[v116c] BUG-C1 FIXED (regex): _bc_spawn_neutral = True")
    else:
        print("[v116c] WARNING BUG-C1: pattern not found -- manual fix required at _bc_spawn_neutral")

# -- BUG-C2: live_summary not defined -------------------------------------------
# Try inserting after _dash_proc = None
OLD_C2 = "        _dash_proc = None\n\n    # --- BSTS metrics CSV"
NEW_C2 = "        _dash_proc = None\n    live_summary = None  # v1.1.6c BUG-C2: was never defined; NameError every episode\n\n    # --- BSTS metrics CSV"
if OLD_C2 in src:
    src = src.replace(OLD_C2, NEW_C2, 1)
    print("[v116c] BUG-C2 FIXED: live_summary = None after _dash_proc = None")
else:
    OLD_C2b = "    # --- BSTS metrics CSV for reward weight history ---"
    if OLD_C2b in src:
        src = src.replace(OLD_C2b,
            "    live_summary = None  # v1.1.6c BUG-C2: NameError guard\n    # --- BSTS metrics CSV for reward weight history ---",
            1)
        print("[v116c] BUG-C2 FIXED (alt): live_summary = None before BSTS CSV block")
    else:
        print("[v116c] WARNING BUG-C2: could not find anchor -- manual fix required")

# -- BUG-C3: _d_prog used before assignment -------------------------------------
OLD_C3 = "                try:\n                    _sac_ex = td3sac.exploration_bonus(None, None, log_prob=None)"
NEW_C3 = (
    "                _d_prog = 0.0  # v1.1.6c BUG-C3: pre-init; UnboundLocalError at step-1\n"
    "                try:\n"
    "                    _sac_ex = td3sac.exploration_bonus(None, None, log_prob=None)"
)
if OLD_C3 in src:
    src = src.replace(OLD_C3, NEW_C3, 1)
    print("[v116c] BUG-C3 FIXED: _d_prog = 0.0 pre-init inserted")
else:
    for indent in ["            ", "                    ", "        "]:
        OLD_C3x = f"{indent}try:\n{indent}    _sac_ex = td3sac.exploration_bonus(None, None, log_prob=None)"
        if OLD_C3x in src:
            src = src.replace(OLD_C3x,
                f"{indent}_d_prog = 0.0  # v1.1.6c BUG-C3\n{OLD_C3x}", 1)
            print(f"[v116c] BUG-C3 FIXED (indent={repr(indent)})")
            break
    else:
        print("[v116c] WARNING BUG-C3: exploration_bonus try block not found -- manual fix required")

if src == original:
    print("[v116c] WARNING: No changes made! Check patterns above.")
else:
    TARGET.write_text(src, encoding="utf-8")
    changed = sum(1 for a, b in zip(original.splitlines(), src.splitlines()) if a != b)
    print(f"[v116c] run.py patched: ~{changed} lines changed. Backup at {backup}")

print("[v116c] Done. Verify: grep -n '_bc_spawn_neutral\\|live_summary\\|_d_prog' run.py | head -20")
