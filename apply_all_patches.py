#!/usr/bin/env python3
"""
CS7642 Project 4 v1.1.5b — FULL REWARD AUDIT PATCH (4 sites)
Run from project_4 directory:  python apply_all_patches.py

SITES:
  A   Bootstrap no-progress  0.88 → 0.78  (anchor confirmed file:145)
  C   Reversal *0.40         + additive -0.10  (anchor confirmed file:145)
  V7  on-track speed bonus   add off-track + reversed gate
  SPD incompatible-behavior  `reward += 0.08 * min(spd,4.0)`  add reversed gate

All anchors verified from live file:145 search output (234210 chars).
"""
import sys, os, shutil, datetime, py_compile, tempfile

BACKUP = f"run.py.bak_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

# SITE-A  Bootstrap no-progress attenuation  0.88 -> 0.78
SITE_A_OLD = "reward = reward * 0.88  # v1.1.0 no-progress ATTENUATION during bootstrap"
SITE_A_NEW = "reward = reward * 0.78  # v1.1.5b SITE-A: 0.88\u21920.78 (22% cut). Phase A only. alive_bonus +0.05 keeps net reward >0."

# SITE-C  Reversal multiplicative -> + additive -0.10
SITE_C_OLD = """\
                if _is_rev:
                    reward = reward * 0.40  # v1.1.2 was 0.80 \u2014 60% cut makes reversal strictly dominated
                # 5: Reverse is attenuated but NOT additively penalized v1.1.0 multiplicative"""

SITE_C_NEW = """\
                if _is_rev:
                    reward = reward * 0.40  # v1.1.2 60% cut (always active)
                    # v1.1.5b SITE-C: additive reversal penalty per step.
                    # progr already gated to 0.0 for reversed (v1.4.1).
                    # After *0.40 the remaining signal = speed/center bonuses only \u2192 not negative.
                    # Flat -0.10 makes reversed step unambiguously below on-track path.
                    reward -= 0.10
                    reward = max(0.002, reward)  # ordinal floor: reversed(>0.002) > terminal(0)
                # 5: Reversal: multiplicative *0.40 + additive -0.10 per step"""

# SITE-V7  On-track speed bonus: gate to NOT off-track, NOT reversed
# PROBLEM: min(speed*4.0,1.0) is always 1.0 at speed>0.25 -> flat +0.5 bonus for any speed>1.
# No off-track gate. No reversed gate. Encourages "gas it" regardless.
SITE_V7_OLD = """\
            if rp_v7.get("speed", 0) > 1.0:
                reward += 0.5 * min(speed * 4.0, 1.0)"""

SITE_V7_NEW = """\
            # v1.1.5b SITE-V7: gate speed bonus to on-track + not-reversed.
            # PROBLEM: min(speed*4.0,1.0)=1.0 at speed>0.25 -> flat +0.5 at any speed>1 m/s.
            # This rewarded off-track speed and reversed speed equally. Now gated.
            _v7_offtrack = bool(rp_v7.get("is_offtrack", offtrack))
            _v7_reversed = bool(rp_v7.get("is_reversed", False))
            if rp_v7.get("speed", 0) > 1.0 and not _v7_offtrack and not _v7_reversed:
                reward += 0.5 * min(speed * 4.0, 1.0)  # on-track, forward, speed>1 m/s"""

# SITE-SPD  Incompatible-behavior block speed reward: add reversed gate
SITE_SPD_OLD = "                reward += 0.08 * min(spd, 4.0)  # caps at 3 m/s to avoid pure speed hacking"

SITE_SPD_NEW = """\
                # v1.1.5b SITE-SPD: gate anti-creep speed reward to not-reversed.
                # Reversed car near spawn can have positive dprog briefly.
                if not _is_rev:
                    reward += 0.08 * min(spd, 4.0)  # anti-creep: caps at 4 m/s, forward only"""


def do_patch(content, old, new, label):
    if old in content:
        sentinel = new.strip().splitlines()[0].strip()[:50]
        if sentinel in content:
            print(f"  ~ {label}: already applied")
            return content, "skip"
        result = content.replace(old, new, 1)
        print(f"  \u2713 {label}: applied")
        return result, "ok"
    else:
        print(f"  \u2717 {label}: NOT FOUND")
        key = old.strip().splitlines()[0].strip()[:60]
        for i, line in enumerate(content.splitlines(), 1):
            if key[:30] in line:
                print(f"    partial match L{i}: {line.strip()[:80]}")
                break
        return content, "fail"


with open("run.py", "r") as f:
    content = f.read()

print(f"run.py loaded: {len(content):,} chars, {content.count(chr(10))+1} lines")
shutil.copy("run.py", BACKUP)
print(f"Backup: {BACKUP}\n")

content, ra   = do_patch(content, SITE_A_OLD,   SITE_A_NEW,   "SITE-A  bootstrap 0.88\u21920.78")
content, rc   = do_patch(content, SITE_C_OLD,   SITE_C_NEW,   "SITE-C  reversal additive -0.10")
content, rv7  = do_patch(content, SITE_V7_OLD,  SITE_V7_NEW,  "SITE-V7 on-track speed gate")
content, rspd = do_patch(content, SITE_SPD_OLD, SITE_SPD_NEW, "SITE-SPD incompatible-block reversed gate")

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
    print("Restored backup \u2014 NO changes written.")
    sys.exit(1)

with open("run.py", "w") as f:
    f.write(content)

print(f"run.py written: {len(content):,} chars\n")
print("Results:")
for label, r in [("A", ra), ("C", rc), ("V7", rv7), ("SPD", rspd)]:
    icon = {"ok": "\u2713", "skip": "~", "fail": "\u2717"}[r]
    print(f"  {icon} SITE-{label}")

print("\nVerification lines:")
for i, line in enumerate(content.splitlines(), 1):
    if any(k in line for k in ["SITE-A:", "SITE-C:", "SITE-V7:", "SITE-SPD:",
                                 "_v7_offtrack", "_v7_reversed", "reward -= 0.10",
                                 "reward * 0.78", "not _is_rev"]):
        print(f"  L{i:4d}: {line.strip()[:95]}")
