#!/usr/bin/env python3
"""
CS7642 Project 4 v1.1.5b — PATCH-R (3-site) applicator
Run from project_4 directory: python apply_patch_r.py

Patches applied:
  SITE-B  off-track vperp attenuation  → + BSTS-gated additive penalty   [PRIMARY]
  SITE-A  bootstrap no-progress atten  → strengthen factor 0.88 → 0.78   [SECONDARY]
  SITE-C  reversal attenuation         → + flat additive penalty -0.10    [SECONDARY]

All sites verified against run.py v1.1.5b (232402 chars).
"""
import sys, os, shutil, datetime, py_compile, tempfile

BACKUP = f"run.py.bak_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

# SITE-B: OFF-TRACK vperp attenuation
# Anchor: unique comment "never subtracts; just scales earned reward down"
SITE_B_OLD = \
'                    reward = reward * _offtrack_attn  # never subtracts; just scales earned reward down'

SITE_B_NEW = \
'''                    reward = reward * _offtrack_attn  # multiplicative attenuation (always active)
                    # v1.1.5b PATCH-R SITE-B: BSTS-gated vperp-scaled additive off-track penalty.
                    # Problem: multiplicative-only at vperp=3.6 cuts reward by 59%, but if the
                    # progress reward earned is small (early episode), 41% of small ≈ small.
                    # The agent perceives off-track as "slightly less good" — not "bad".
                    # Fix: add -0.15*vperp once BSTS avg_track_progress EMA > 0.30 (Phase B gate).
                    # Safety guarantees (Ng, Harada, Russell 1999 ICML):
                    #   1. vperp-scaled: creeping back to track costs less than blasting off it
                    #   2. max(0.005, reward) floor: net step reward never goes negative
                    #      Freeze trap impossible: sim min-speed = 2.25 m/s enforced by engine
                    #   3. BSTS-gated at avg_tp > 0.30: Phase A agent earns pure +signal first
                    # REF: Ng, Harada, Russell (1999) ICML — policy invariance under shaping
                    # REF: Bengio et al. (2009) ICML — curriculum learning, easy-first phases
                    _avg_tp_step = float(
                        bsts_feedback.ema.get('avg_track_progress', 0.0)
                        if hasattr(bsts_feedback, 'ema') else 0.0
                    )
                    if _avg_tp_step == 0.0 and '_prog' in dir() and float(_prog) > 0:
                        _avg_tp_step = float(_prog) / 100.0  # fallback before EMA populated
                    if _avg_tp_step > 0.30:
                        # At vperp=3.6 → penalty=-0.54; net reward still >0 at speed>=1 m/s
                        _otp = -0.15 * max(0.0, _vperp_val)
                        reward += _otp
                        reward = max(0.005, reward)  # ordinal floor: bad(0.005) > terminal(0)'''

# SITE-A: Bootstrap no-progress attenuation 0.88 → 0.78
# Anchor: exact comment text from file:143
SITE_A_OLD = \
'                    reward = reward * 0.88  # v1.1.0 no-progress ATTENUATION during bootstrap — 12% cut per step, no additive penalty'

SITE_A_NEW = \
'                    reward = reward * 0.78  # v1.1.5b PATCH-R SITE-A: 0.88→0.78 (22% cut). Phase A only. alive_bonus +0.05 keeps net reward >0.'

# SITE-C: Reversal attenuation
# Anchor: two-line block verified from file:143 search output
SITE_C_OLD = \
'''                if _is_rev:
                    reward = reward * 0.40  # v1.1.2 was 0.80 — 60% cut makes reversal strictly dominated
                # 5: Reverse is attenuated but NOT additively penalized v1.1.0 multiplicative'''

SITE_C_NEW = \
'''                if _is_rev:
                    reward = reward * 0.40  # v1.1.2 60% cut (always active)
                    # v1.1.5b PATCH-R SITE-C: additive reversal penalty.
                    # progr is already gated to 0.0 when reversed (v1.4.1 gate), so after
                    # reward*0.40 the signal is dominated by speed/center bonuses only.
                    # A flat -0.10 makes per-step reversal signal unambiguously negative.
                    # Freeze trap: impossible (sim min-speed 2.25 m/s; car always moving).
                    # spawn_penalty handles episode-level reversal; this handles per-step.
                    reward -= 0.10
                    reward = max(0.002, reward)  # ordinal floor: reversed(>0) > terminal(0)
                # 5: Reversal now has multiplicative attenuation + additive per-step penalty'''


def try_patch(content, old, new, label):
    if old not in content:
        print(f"  \u2717 {label}: anchor NOT FOUND")
        return content, False
    sentinel = new.strip().splitlines()[1].strip()[:40]
    if sentinel in content:
        print(f"  ~ {label}: already applied — skipping")
        return content, True
    result = content.replace(old, new, 1)
    print(f"  \u2713 {label}: applied")
    return result, True


with open("run.py", "r") as f:
    content = f.read()

print(f"run.py loaded: {len(content):,} chars, {content.count(chr(10))+1} lines")

if "PATCH-R SITE-B" in content and "PATCH-R SITE-A" in content and "PATCH-R SITE-C" in content:
    print("All three PATCH-R sites already present. Nothing to do.")
    sys.exit(0)

shutil.copy("run.py", BACKUP)
print(f"Backup written: {BACKUP}\n")

content, ok_b = try_patch(content, SITE_B_OLD, SITE_B_NEW, "SITE-B off-track penalty")
content, ok_a = try_patch(content, SITE_A_OLD, SITE_A_NEW, "SITE-A bootstrap atten 0.88\u21920.78")
content, ok_c = try_patch(content, SITE_C_OLD, SITE_C_NEW, "SITE-C reversal additive -0.10")

print()
tmp = tempfile.mktemp(suffix=".py")
with open(tmp, "w") as f:
    f.write(content)
try:
    py_compile.compile(tmp, doraise=True)
    print("Syntax check: OK")
    os.unlink(tmp)
except py_compile.PyCompileError as e:
    print(f"SYNTAX ERROR: {e}")
    shutil.copy(BACKUP, "run.py")
    print("Restored backup — no changes written.")
    sys.exit(1)

with open("run.py", "w") as f:
    f.write(content)

print(f"run.py written: {len(content):,} chars\n")
print("Applied patches:")
for site, ok in [("B", ok_b), ("A", ok_a), ("C", ok_c)]:
    print(f"  SITE-{site}: {'\u2713' if ok else '\u2717 NOT FOUND — check run.py version'}")

print("\nVerification (key lines):")
for i, line in enumerate(content.splitlines(), 1):
    if any(k in line for k in ["PATCH-R SITE", "_otp =", "reward -= 0.10", "0.78  # v1.1.5b"]):
        print(f"  L{i:4d}: {line.strip()[:95]}")
