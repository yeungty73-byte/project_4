#!/usr/bin/env python3
"""Crash Hotspot Analyzer - identifies track waypoints with high crash rates."""
import json, re, sys, os, collections, math, glob
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / 'results'

def parse_crash_forensics(log_path):
    crashes = []
    pattern = re.compile(r'CRASH_FORENSIC.*?x=([\d.\-]+).*?y=([\d.\-]+).*?wp=(\d+).*?spd=([\d.]+).*?heading=([\d.\-]+)')
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                crashes.append({'x': float(m.group(1)), 'y': float(m.group(2)),
                    'waypoint': int(m.group(3)), 'speed': float(m.group(4)),
                    'heading': float(m.group(5))})
    return crashes

def analyze_hotspots(crashes, top_n=10):
    wp_counts = collections.Counter(c['waypoint'] for c in crashes)
    hotspots = []
    for wp, count in wp_counts.most_common(top_n):
        wp_crashes = [c for c in crashes if c['waypoint'] == wp]
        hotspots.append({
            'waypoint': wp, 'count': count,
            'pct': round(count/len(crashes)*100, 1),
            'avg_speed': round(sum(c['speed'] for c in wp_crashes)/len(wp_crashes), 2),
            'avg_heading': round(sum(c['heading'] for c in wp_crashes)/len(wp_crashes), 1),
            'avg_x': round(sum(c['x'] for c in wp_crashes)/len(wp_crashes), 2),
            'avg_y': round(sum(c['y'] for c in wp_crashes)/len(wp_crashes), 2),
        })
    return hotspots

def generate_report(hotspots, total_crashes):
    print(f'=== CRSSH HOTSPOT REPORT ({total_crashes} total crashes) ===')
    for i, h in enumerate(hotspots):
        print(f"  #{i+1} WP{h['waypoint']}: "
              f"{h['count']} crashes ({h['pct']}%) "
              f"avg_spd={h['avg_speed']} avg_hdg={h['avg_heading']} "
              f"pos=({h['avg_x']}, {h['avg_y']})")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / 'crash_hotspot_report.json'
    with open(report_path, 'w') as f:
        json.dump({'total_crashes': total_crashes, 'hotspots': hotspots}, f, indent=2)
    print(f'Report saved to {report_path}')

def main():
    logs = sorted(glob.glob('training_*.log'))
    if not logs:
        logs = sorted(glob.glob(str(RESULTS_DIR.parent / 'training_*.log')))
    all_crashes = []
    for log in logs:
        crashes = parse_crash_forensics(log)
        print(f'{log}: {len(crashes)} crashes')
        all_crashes.extend(crashes)
    if not all_crashes:
        print('No crash data found'); return
    hotspots = analyze_hotspots(all_crashes)
    generate_report(hotspots, len(all_crashes))

if __name__ == '__main__':
    main()
