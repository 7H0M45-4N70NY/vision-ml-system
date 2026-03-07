import sys
import os
import argparse
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vision_ml.analytics.analytics_db import AnalyticsDB


def main():
    parser = argparse.ArgumentParser(description='Analytics CLI')
    parser.add_argument('--action', required=True, choices=['summary', 'inference-runs', 'training-events', 'labeling-events'])
    parser.add_argument('--limit', type=int, default=20)
    args = parser.parse_args()

    db = AnalyticsDB()

    if args.action == 'summary':
        summary = db.get_analytics_summary()
        print('[Analytics] Summary')
        for key, value in summary.items():
            print(f'  {key}: {value}')
        return

    if args.action == 'inference-runs':
        rows = db.get_inference_runs(limit=args.limit)
    elif args.action == 'training-events':
        rows = db.get_training_events(limit=args.limit)
    else:
        rows = db.get_labeling_events(limit=args.limit)

    if not rows:
        print('[Analytics] No rows found.')
        return

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
