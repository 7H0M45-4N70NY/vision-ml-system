import sys
import os
import argparse
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vision_ml.mlflow_integration import MLflowManager


def _get_registered_models(client):
    if hasattr(client, 'search_registered_models'):
        return list(client.search_registered_models())
    return list(client.list_registered_models())


def list_models(manager: MLflowManager):
    models = _get_registered_models(manager.client)
    if not models:
        print('[Registry] No registered models found.')
        return

    print(f"[Registry] Found {len(models)} models")
    print('-' * 100)
    print(f"{'MODEL':40} {'VERSIONS':10} {'LAST UPDATED':20}")
    print('-' * 100)
    for model in models:
        updated = datetime.fromtimestamp(model.last_updated_timestamp / 1000).strftime('%Y-%m-%d %H:%M')
        print(f"{model.name[:40]:40} {len(model.latest_versions):10} {updated:20}")


def compare_model(manager: MLflowManager, model_name: str, metric_name: str):
    comparison = manager.compare_models(model_name=model_name, metric_name=metric_name)
    if not comparison:
        print(f"[Registry] No staged models found for '{model_name}'.")
        return

    print(f"[Registry] Comparison for '{model_name}' on metric '{metric_name}'")
    print('-' * 100)
    for stage in ('Production', 'Staging'):
        info = comparison.get(stage)
        if not info:
            print(f"{stage:12}: N/A")
            continue
        print(
            f"{stage:12}: version={info.get('version')} "
            f"run_id={info.get('run_id')} "
            f"{metric_name}={info.get(metric_name)}"
        )


def promote_model(manager: MLflowManager, model_name: str, from_stage: str, to_stage: str):
    manager.promote_model(model_name=model_name, from_stage=from_stage, to_stage=to_stage)
    print(f"[Registry] Promotion requested: {model_name} {from_stage} -> {to_stage}")


def main():
    parser = argparse.ArgumentParser(description='Model Registry CLI (MLflow + DagsHub)')
    parser.add_argument('--action', required=True, choices=['list', 'compare', 'promote'])
    parser.add_argument('--model-name', type=str, default=None, help='Model name for compare/promote')
    parser.add_argument('--metric-name', type=str, default='val_loss', help='Metric for compare action')
    parser.add_argument('--from-stage', type=str, default='Staging', help='Source stage for promote')
    parser.add_argument('--to-stage', type=str, default='Production', help='Target stage for promote')
    args = parser.parse_args()

    manager = MLflowManager()

    if args.action == 'list':
        list_models(manager)
    elif args.action == 'compare':
        if not args.model_name:
            raise ValueError('--model-name is required for --action compare')
        compare_model(manager, args.model_name, args.metric_name)
    elif args.action == 'promote':
        if not args.model_name:
            raise ValueError('--model-name is required for --action promote')
        promote_model(manager, args.model_name, args.from_stage, args.to_stage)


if __name__ == '__main__':
    main()
