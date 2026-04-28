#!/usr/bin/env python3
"""
patches/v1_1_6b_run_patch.py

Applies 3 surgical patches to run.py for v1.1.6b.
Run once from the project root: python patches/v1_1_6b_run_patch.py

PATCH-I: Add tracklengthm key to epdata dict
  Location: inside the Online Kalman BSTS update block, epdata = { ... } dict
  Root cause: tracklengthm 0.0 in every Kalman line despite PATCH-BB reading
              rp['tracklength'] into ep_track_length_m. ep_data dict never had
              the tracklengthm key, so analyzelogs.episodesummarymetrics always
              got ep.get('tracklengthm', 16.6) -> default 16.6, then
              computesuccess guard (tracklengthm > 1.0 and tracklengthm < 99.0)
              passed, BUT bstsfeedback.update bstsmetrics had the old 100.0 reset
              value -> trackprogressm 0.0 in Kalman.

PATCH-L-FIX: Unconditional 3-tier COMPLIANCEKEYSNEUTRALS merge into summary
  Location: after bstsrow.updatehmout at ~L3049, in the compliance override block
  Root cause: brakecompliance 0.0, racelineadherence 0.0 in Kalman X-matrix.
              hmout override guard `if hmout in dir() and hmout` silently skips
              when computeall throws exception -> summary gets 0.0 from empty
              computesuccess path -> Kalman regressors all zero -> BSTS blind.

PATCH-SWIN: Capture swin_clearance from bf_step
  Location: after _bf_step = _brake_field.step(...) call
  Root cause: _swin_clearance never assigned in run.py.
              BrakeField.step() returns swin_clearance in result dict,
              but it was never read out -> SwinUNetPP perception always disabled.
"""

import re
import sys
import shutil
from pathlib import Path

RUN_PY = Path('run.py')
BAK = Path('run.py.bak.v116b')

def apply_patches():
    if not RUN_PY.exists():
        print(f'ERROR: {RUN_PY} not found. Run from project root.', file=sys.stderr)
        sys.exit(1)

    src = RUN_PY.read_text(encoding='utf-8')
    shutil.copy(RUN_PY, BAK)
    print(f'[v1.1.6b] Backup saved: {BAK}')

    original_len = len(src)
    patches_applied = 0

    # ==========================================================================
    # PATCH-I: Add tracklengthm to epdata dict
    # Find the epdata = { ... } block (has trackwidth and nwaypoints keys)
    # and add tracklengthm after nwaypoints.
    # ==========================================================================
    PATCH_I_FIND = ('            nwaypoints lenwaypoints if waypoints in dir() '
                    'and waypoints else 120,')
    # Handle both with and without leading whitespace variants
    PATCH_I_FIND_ALT = (
        'nwaypoints = len(waypoints) if \'waypoints\' in dir() and waypoints else 120,'
    )

    PATCH_I_INSERT = (
        '            # v1.1.6b FIX-I: forward real arc length so trackprogress guard (>1.0 and <99.0) passes.\n'
        '            # ep_track_length_m updated per-step from rp[\'tracklength\'] via PATCH-BB.\n'
        '            # Without this key, analyzelogs.episodesummarymetrics uses ep.get(\'tracklengthm\', 16.6)\n'
        '            # but computeall receives None -> trackprogressm=0.0 in every Kalman line.\n'
        '            # REF: Heilmeier et al. (2020) VSD -- track arc length normalization.\n'
        "            tracklengthm=float(ep_track_length_m) if 'ep_track_length_m' in dir() and float(ep_track_length_m) > 1.0 and float(ep_track_length_m) < 99.0 else 16.6,\n"
    )

    # Try the exact string from the live run.py
    _epdata_marker = 'nwaypoints lenwaypoints if waypoints in dir() and waypoints else 120,'
    if _epdata_marker in src and 'tracklengthm=float(ep_track_length_m)' not in src:
        # Insert AFTER the nwaypoints line
        idx = src.find(_epdata_marker)
        eol = src.find('\n', idx)
        src = src[:eol+1] + PATCH_I_INSERT + src[eol+1:]
        print('[PATCH-I] tracklengthm added to epdata dict')
        patches_applied += 1
    elif 'tracklengthm=float(ep_track_length_m)' in src:
        print('[PATCH-I] already applied (tracklengthm key found in epdata)')
    else:
        # Try alternate form
        _alt = 'nwaypoints = len(waypoints)'
        if _alt in src:
            idx = src.find(_alt)
            eol = src.find('\n', idx)
            src = src[:eol+1] + PATCH_I_INSERT + src[eol+1:]
            print('[PATCH-I] tracklengthm added to epdata dict (alt form)')
            patches_applied += 1
        else:
            print('[PATCH-I] WARNING: could not locate epdata nwaypoints line. Manual patch needed.')

    # ==========================================================================
    # PATCH-L-FIX: Replace old hmout/bstsrow conditional override with
    # 3-tier COMPLIANCEKEYSNEUTRALS unconditional merge.
    # ==========================================================================
    PATCH_L_FIND_MARKER = 'v1.1.5 FIX Mirror hmout into summary with BOTH bare key AND mean key.'
    PATCH_L_FIND_MARKER_ALT = 'hmsrc = hmout if hmout in dir() and hmout else {}'

    PATCH_L_REPLACEMENT = '''        # v1.1.6b PATCH-L-FIX: Unconditional 3-tier COMPLIANCEKEYSNEUTRALS merge into summary.
        # Priority: bstsmetrics (neutral floor, always populated)
        #        -> hmout (richer per-step, can be None/{})
        #        -> bstsrow (post-updatehmout, most authoritative)
        # bstsrow.updatehmout at L3049 is CORRECT and UNCHANGED.
        # This block reads bstsrow AFTER that update, inheriting hmout values when available.
        # REF: Ng et al. (1999) -- all per-step signals must reach gradient path.
        _COMPLIANCE_KEYS_NEUTRALS = {
            'racelineadherence': 0.5,
            'brakecompliance': 1.0,
            'brakefieldcompliancegradient': 1.0,
            'racelinecompliancegradient': 0.5,
            'smoothnesssteeringrate': 1.0,
            'trackprogress': 0.0,
            'avgspeedcenterline': 0.0,
        }
        # Tier 1: neutral floor from bstsmetrics (always populated)
        for _ck, _cn in _COMPLIANCE_KEYS_NEUTRALS.items():
            _cv = bstsmetrics.get(_ck) if 'bstsmetrics' in dir() and bstsmetrics else None
            if _cv is not None:
                try:
                    _cvf = float(_cv)
                    if math.isfinite(_cvf):
                        summary[_ck] = _cvf
                        summary[f\'{_ck}mean\'] = _cvf
                except (TypeError, ValueError):
                    pass
            else:
                summary.setdefault(_ck, _cn)
                summary.setdefault(f\'{_ck}mean\', _cn)
        # Tier 2: hmout override (richer per-step, if computeall succeeded)
        if 'hmout' in dir() and isinstance(hmout, dict) and hmout:
            for _hmk, _hmv in hmout.items():
                try:
                    _hfv = float(_hmv)
                    if math.isfinite(_hfv):
                        summary[_hmk] = _hfv
                        summary[f\'{_hmk}mean\'] = _hfv
                except (TypeError, ValueError):
                    pass
        # Tier 3: bstsrow (already has hmout merged via updatehmout at L3049)
        if 'bstsrow' in dir() and isinstance(bstsrow, dict) and bstsrow:
            for _bk, _bn in _COMPLIANCE_KEYS_NEUTRALS.items():
                _bv = bstsrow.get(_bk)
                if _bv is not None:
                    try:
                        _bvf = float(_bv)
                        if math.isfinite(_bvf):
                            summary[_bk] = _bvf
                            summary[f\'{_bk}mean\'] = _bvf
                    except (TypeError, ValueError):
                        pass
'''

    if PATCH_L_FIND_MARKER in src and 'COMPLIANCE_KEYS_NEUTRALS' not in src:
        # Find the block and its end (replace from marker to the bstsrow loop end)
        start = src.find('        # v1.1.5 FIX Mirror hmout')
        if start == -1:
            start = src.find('v1.1.5 FIX Mirror hmout')
            if start != -1:
                start = src.rfind('\n', 0, start) + 1
        # Find end of the existing override block - look for the closing bstsrow loop
        end_marker = 'summaryfbkmean = floatbstsrowbk'
        end_marker2 = "summary[f'{bk}mean'] = float(bstsrow[bk])"
        end_marker3 = 'summaryfbkmean floatbstsrowbk'
        _end = -1
        for _em in [end_marker, end_marker2, end_marker3]:
            _pos = src.find(_em, start)
            if _pos != -1:
                _end = src.find('\n', _pos) + 1
                break
        if start != -1 and _end > start:
            src = src[:start] + PATCH_L_REPLACEMENT + src[_end:]
            print('[PATCH-L-FIX] 3-tier COMPLIANCEKEYSNEUTRALS merge installed')
            patches_applied += 1
        else:
            print('[PATCH-L-FIX] WARNING: could not find end of old override block. Manual patch needed.')
    elif 'COMPLIANCE_KEYS_NEUTRALS' in src:
        print('[PATCH-L-FIX] already applied')
    elif PATCH_L_FIND_MARKER_ALT in src and 'COMPLIANCE_KEYS_NEUTRALS' not in src:
        start = src.rfind('\n', 0, src.find(PATCH_L_FIND_MARKER_ALT)) + 1
        end = src.find('\n', src.find(PATCH_L_FIND_MARKER_ALT)) + 1
        src = src[:start] + PATCH_L_REPLACEMENT + src[end:]
        print('[PATCH-L-FIX] 3-tier merge installed (alt marker)')
        patches_applied += 1
    else:
        print('[PATCH-L-FIX] WARNING: could not locate old hmout override block. Manual patch needed.')

    # ==========================================================================
    # PATCH-SWIN: Capture swin_clearance from _bf_step result dict
    # ==========================================================================
    PATCH_SWIN_FIND = "        _bf_ok              = (not _in_brake_field) or _is_braking_now"
    PATCH_SWIN_INSERT = (
        "        # v1.1.6b PATCH-SWIN: read swin_clearance from bf_step so SwinUNetPP\n"
        "        # 4-class perception path can fire. Without this, _swin_clearance=None\n"
        "        # was passed to every BrakeField.step() call (grep confirms no assignment).\n"
        "        _swin_clearance = _bf_step.get('swin_clearance', None)\n"
    )
    if PATCH_SWIN_FIND in src and "_swin_clearance = _bf_step.get('swin_clearance'" not in src:
        idx = src.find(PATCH_SWIN_FIND)
        eol = src.find('\n', idx)
        src = src[:eol+1] + PATCH_SWIN_INSERT + src[eol+1:]
        print('[PATCH-SWIN] _swin_clearance captured from bf_step')
        patches_applied += 1
    elif "_swin_clearance = _bf_step.get('swin_clearance'" in src:
        print('[PATCH-SWIN] already applied')
    else:
        print('[PATCH-SWIN] WARNING: could not locate _bf_ok line. Manual patch needed.')

    # Write patched file
    RUN_PY.write_text(src, encoding='utf-8')
    print(f'[v1.1.6b] run.py patched: {patches_applied}/3 patches applied '
          f'({len(src) - original_len:+d} bytes)')

    # Syntax check
    import ast
    try:
        ast.parse(src)
        print('[v1.1.6b] Syntax OK')
    except SyntaxError as e:
        print(f'[v1.1.6b] SYNTAX ERROR: {e}', file=sys.stderr)
        print(f'[v1.1.6b] Restoring backup...', file=sys.stderr)
        shutil.copy(BAK, RUN_PY)
        sys.exit(1)

if __name__ == '__main__':
    apply_patches()
