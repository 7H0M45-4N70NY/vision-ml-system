import sys
import os
import argparse

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vision_ml.analytics.analytics_db import AnalyticsDB


def _print_events(events):
    if not events:
        print('[Pipeline] No training events found.')
        return

    df = pd.DataFrame(events)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp', ascending=False)

    cols = [c for c in ['event_id', 'timestamp', 'trigger_type', 'dataset_size', 'drift_score', 'status', 'model_version'] if c in df.columns]
    print(df[cols].to_string(index=False))


def _detect_drift(db: AnalyticsDB, threshold: float, lookback: int):
    runs = db.get_inference_runs(limit=lookback)
    if len(runs) < 2:
        print('[Pipeline] Need at least 2 inference runs for drift check.')
        return False, 0.0

    df = pd.DataFrame(runs)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    # Use confidence-based drift score stored per run
    latest = df.iloc[-1]
    drift_score = float(latest.get('drift_score', 0) or 0)
    avg_conf = float(latest.get('avg_confidence', 0) or 0)

    print(f"[Pipeline] Avg confidence: {avg_conf:.4f}  Drift score: {drift_score:.4f}  (threshold: {threshold:.4f})")
    return drift_score > threshold, drift_score


def main():
    parser = argparse.ArgumentParser(description='Training Pipeline CLI')
    parser.add_argument('--action', required=True, choices=['status', 'drift-check', 'trigger'])
    parser.add_argument('--limit', type=int, default=20, help='Limit for status action')
    parser.add_argument('--threshold', type=float, default=0.2, help='Drift threshold for drift-check/trigger')
    parser.add_argument('--lookback', type=int, default=50, help='Inference lookback for drift-check/trigger')
    parser.add_argument('--model-version', type=str, default='v1', help='Model version for trigger action')
    parser.add_argument('--force', action='store_true', help='Force trigger event even if drift is below threshold')
    args = parser.parse_args()

    db = AnalyticsDB()

    if args.action == 'status':
        events = db.get_training_events(limit=args.limit)
        _print_events(events)
        return

    drift_detected, drift_score = _detect_drift(db, args.threshold, args.lookback)

    if args.action == 'drift-check':
        if drift_detected:
            print('[Pipeline] Drift detected.')
            return
        print('[Pipeline] No drift detected.')
        return

    if args.action == 'trigger':
        if drift_detected or args.force:
            event_id = db.save_training_event({
                'trigger_type': 'drift' if drift_detected else 'manual',
                'dataset_size': 0,
                'drift_score': drift_score,
                'model_version': args.model_version,
            })
            print(f"[Pipeline] Training event created: {event_id}")
            return

        print('[Pipeline] Trigger skipped (no drift; use --force to override).')


if __name__ == '__main__':
    main()
