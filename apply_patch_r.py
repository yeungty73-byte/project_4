#!/usr/bin/env python3
"""
CS7642 Project 4 v1.1.5b PATCH-R applicator
Run this script from the project_4 directory:
    python apply_patch_r.py

Applies PATCH-R: BSTS-gated vperp-scaled off-track penalty to run.py
"""
import sys, os, shutil, datetime

BACKUP = f"run.py.bak_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

OLD = """                if _offtrack:
                    _vperp_val = float(_v_perp_barrier) if "_v_perp_barrier" in dir() else 1.0
                    _offtrack_attn = 1.0 / (1.0 + max(0.0, _vperp_val) * 0.40)
                    reward = reward * _offtrack_attn  # never subtracts; just scales earned reward down
                _is_stuck = (_speed < 0.3) or _offtrack_stuck or _is_rev  # v39: uses grace"""

NEW = """                if _offtrack:
                    _vperp_val = float(_v_perp_barrier) if "_v_perp_barrier" in dir() else 1.0
                    _offtrack_attn = 1.0 / (1.0 + max(0.0, _vperp_val) * 0.40)
                    reward = reward * _offtrack_attn  # multiplicative attenuation (always active)
                    # v1.1.5b PATCH-R: vperp-scaled additive off-track penalty, BSTS-gated.
                    # Off-track attenuation alone is too weak: at vperp=3.6 it cuts reward
                    # by 59%, but the absolute cut on a small progress reward is trivial.
                    # The agent perceives off-track as "slightly less good", not "bad".
                    # Design constraints (Ng et al. 1999 safe):
                    #   1. penalty is vperp-scaled: creeping back costs less than blasting off
                    #   2. max(0.005, reward) floor: net step reward stays positive at speed>=1 m/s
                    #      -> freeze trap architecturally impossible (sim min-speed 2.25 m/s)
                    #   3. BSTS-gated: only fires once avg_track_progress EMA > 0.30
                    #      -> Phase A agent gets pure positive signal first (Bengio 2009 curriculum)
                    # REF: Ng, A.Y., Harada, D., Russell, S. (1999) ICML Policy Invariance.
                    # REF: Bengio, Y. et al. (2009) ICML Curriculum Learning.
                    _avg_tp_step = float(
                        bsts_feedback.ema.get('avg_track_progress', 0.0)
                        if hasattr(bsts_feedback, 'ema') else 0.0
                    )
                    # Fallback: use current episode pct if EMA not yet populated
                    if _avg_tp_step == 0.0 and '_prog' in dir() and float(_prog) > 0:
                        _avg_tp_step = float(_prog) / 100.0
                    if _avg_tp_step > 0.30:
                        _otp_scale = 0.15  # at vperp=3.6 -> penalty=-0.54; net positive at speed>=1 m/s
                        _otp = -_otp_scale * max(0.0, _vperp_val)
                        reward += _otp
                        reward = max(0.005, reward)  # floor: preserves ordinal signal (bad > terminal)
                _is_stuck = (_speed < 0.3) or _offtrack_stuck or _is_rev  # v39: uses grace"""

with open("run.py", "r") as f:
    content = f.read()

if OLD not in content:
    print("ERROR: Could not find target block. Is run.py the right version?")
    print("Looking for: '...reward = reward * _offtrack_attn  # never subtracts...'")
    sys.exit(1)

if "PATCH-R" in content:
    print("PATCH-R already applied.")
    sys.exit(0)

shutil.copy("run.py", BACKUP)
print(f"Backup: {BACKUP}")

content = content.replace(OLD, NEW, 1)

with open("run.py", "w") as f:
    f.write(content)

import py_compile, tempfile
tmp = tempfile.mktemp(suffix=".py")
with open(tmp, "w") as f:
    f.write(content)
try:
    py_compile.compile(tmp, doraise=True)
    print("Syntax: OK")
    os.unlink(tmp)
except py_compile.PyCompileError as e:
    print(f"SYNTAX ERROR: {e}")
    shutil.copy(BACKUP, "run.py")
    print("Restored backup.")
    sys.exit(1)

print("PATCH-R applied successfully.")
print("\nVerification:")
for i, line in enumerate(content.splitlines(), 1):
    if any(k in line for k in ["PATCH-R", "_otp_scale", "_avg_tp_step", "max(0.005"]):
        print(f"  L{i}: {line.strip()[:80]}")
