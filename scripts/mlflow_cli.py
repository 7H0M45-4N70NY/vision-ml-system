import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vision_ml.mlflow_integration import MLflowManager


def _fmt_float(value):
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _print_experiments(manager: MLflowManager):
    experiments = manager.client.search_experiments()
    if not experiments:
        print("[MLflow] No experiments found.")
        return

    print(f"[MLflow] Found {len(experiments)} experiments")
    print("-" * 90)
    print(f"{'EXPERIMENT NAME':45} {'ID':10} {'LIFECYCLE':15}")
    print("-" * 90)
    for exp in experiments:
        print(f"{exp.name[:45]:45} {exp.experiment_id:10} {exp.lifecycle_stage:15}")


def _print_runs(manager: MLflowManager, experiment_name: str, limit: int, order_by: str):
    runs = manager.get_experiment_runs(experiment_name=experiment_name, limit=limit, order_by=order_by)
    if not runs:
        print(f"[MLflow] No runs found for experiment '{experiment_name}'.")
        return

    print(f"[MLflow] Top {len(runs)} runs in '{experiment_name}'")
    print("-" * 120)
    print(f"{'RUN ID':36} {'STATUS':10} {'VAL_LOSS':10} {'TRAIN_LOSS':10} {'ACCURACY':10}")
    print("-" * 120)
    for run in runs:
        metrics = run.get('metrics', {})
        print(
            f"{run.get('run_id', ''):36} "
            f"{run.get('status', ''):10} "
            f"{_fmt_float(metrics.get('val_loss')):10} "
            f"{_fmt_float(metrics.get('train_loss')):10} "
            f"{_fmt_float(metrics.get('accuracy')):10}"
        )


def _print_best_run(manager: MLflowManager, experiment_name: str, metric_name: str, mode: str):
    best = manager.get_best_run(experiment_name=experiment_name, metric_name=metric_name, mode=mode)
    if not best:
        print(f"[MLflow] No best run found for experiment '{experiment_name}'.")
        return

    print(f"[MLflow] Best run for experiment '{experiment_name}' ({metric_name}, mode={mode})")
    print("-" * 90)
    print(f"Run ID : {best.get('run_id')}")
    print(f"Status : {best.get('status')}")
    print(f"Metric : {_fmt_float(best.get('metrics', {}).get(metric_name))}")


def main():
    parser = argparse.ArgumentParser(description="MLflow + DagsHub CLI")
    parser.add_argument(
        '--action',
        required=True,
        choices=['experiments', 'runs', 'best'],
        help='Action to perform',
    )
    parser.add_argument('--experiment-name', type=str, default=None, help='MLflow experiment name for runs/best')
    parser.add_argument('--limit', type=int, default=20, help='Max runs to show')
    parser.add_argument('--order-by', type=str, default='metrics.val_loss ASC', help='Order clause for runs action')
    parser.add_argument('--metric-name', type=str, default='val_loss', help='Metric name for best action')
    parser.add_argument('--mode', type=str, default='min', choices=['min', 'max'], help='Optimization mode for best action')
    args = parser.parse_args()

    manager = MLflowManager()

    if args.action == 'experiments':
        _print_experiments(manager)
    elif args.action == 'runs':
        if not args.experiment_name:
            raise ValueError("--experiment-name is required for --action runs")
        _print_runs(manager, args.experiment_name, args.limit, args.order_by)
    elif args.action == 'best':
        if not args.experiment_name:
            raise ValueError("--experiment-name is required for --action best")
        _print_best_run(manager, args.experiment_name, args.metric_name, args.mode)


if __name__ == '__main__':
    main()
