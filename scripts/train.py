import sys
import os
import argparse
import subprocess
from dotenv import load_dotenv

from datetime import datetime

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vision_ml.training.trainer import Trainer
from vision_ml.analytics.analytics_db import AnalyticsDB
from vision_ml.utils.config import load_config
from vision_ml.events.publishers import get_pipeline_event_publisher


def main():
    parser = argparse.ArgumentParser(description="Vision ML Training Pipeline")
    parser.add_argument('--config', type=str, default='config/training/base.yaml')
    parser.add_argument('--run-name', type=str, default=None, help='MLflow run name override')
    parser.add_argument('--trigger', type=str, default='manual',
                        choices=['manual', 'drift', 'auto'], help='Training trigger type')
    parser.add_argument('--dataset-yaml', type=str, default=None, help='Override data.dataset_yaml')
    parser.add_argument('--device', type=str, default=None, help='Override training.device (cpu/cuda)')
    parser.add_argument('--epochs', type=int, default=None, help='Override training.epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Override training.batch_size')
    parser.add_argument('--learning-rate', type=float, default=None, help='Override training.learning_rate')
    parser.add_argument('--disable-mlflow', action='store_true', help='Disable MLflow logging/registration for this run')
    parser.add_argument('--enable-dvc', action='store_true', help='Enable DVC hook commands around training')
    parser.add_argument('--dvc-pull', action='store_true', help='Run `dvc pull` before training (requires --enable-dvc)')
    parser.add_argument('--dvc-add', action='store_true', help='Run `dvc add` on targets after training (requires --enable-dvc)')
    parser.add_argument('--dvc-push', action='store_true', help='Run `dvc push` after training (requires --enable-dvc)')
    parser.add_argument('--dvc-targets', type=str, default='runs/train',
                        help='Comma-separated targets for `dvc add` when --dvc-add is enabled')
    parser.add_argument('--skip-event-log', action='store_true', help='Skip logging training event in analytics.db')
    parser.add_argument('--event-id', type=str, default=None,
                        help='Reuse an existing training event ID instead of creating a new one')
    args = parser.parse_args()

    config = load_config(args.config)
    publisher = get_pipeline_event_publisher(config)

    # CLI overrides
    if args.dataset_yaml:
        config.setdefault('data', {})['dataset_yaml'] = args.dataset_yaml
    if args.device:
        config.setdefault('training', {})['device'] = args.device
    if args.epochs is not None:
        config.setdefault('training', {})['epochs'] = args.epochs
    if args.batch_size is not None:
        config.setdefault('training', {})['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        config.setdefault('training', {})['learning_rate'] = args.learning_rate

    if args.disable_mlflow:
        mlflow_cfg = config.setdefault('mlflow', {})
        mlflow_cfg['log_params'] = False
        mlflow_cfg['log_metrics'] = False
        mlflow_cfg['log_model'] = False
        mlflow_cfg['register_model'] = False
        print("[Script] MLflow logging disabled for this run (--disable-mlflow).")

    effective_trigger = args.trigger
    if args.trigger == 'auto':
        schedule_mode = config.get('schedule', {}).get('mode', 'manual')
        effective_trigger = 'drift' if schedule_mode == 'on_drift' else 'manual'
        print(f"[Script] Auto trigger resolved via schedule.mode='{schedule_mode}' -> '{effective_trigger}'")

    event_id = None
    db = AnalyticsDB()
    if args.event_id:
        event_id = args.event_id
        db.update_training_event_status(
            event_id,
            'pending',
            {
                'trigger_type': effective_trigger,
                'model_version': config.get('mlflow', {}).get('model_name', 'v1'),
            },
        )
        print(f"[Script] Reusing training event: {event_id}")
    elif not args.skip_event_log:
        event_id = db.save_training_event({
            'trigger_type': effective_trigger,
            'dataset_size': 0,
            'drift_score': 0.0,
            'model_version': config.get('mlflow', {}).get('model_name', 'v1'),
            'status': 'pending',
        })
        print(f"[Script] Logged training event: {event_id}")

    def _run_dvc(cmd) -> None:
        print(f"[DVC] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    if args.enable_dvc and args.dvc_pull:
        _run_dvc(['dvc', 'pull'])

    trainer = Trainer(config)

    started_at = datetime.now()
    if event_id:
        db.mark_training_event_running(event_id)

    publisher.publish('training.started', {
        'event_id': event_id,
        'trigger': effective_trigger,
        'run_name': args.run_name,
    })

    try:
        if effective_trigger == 'drift':
            print("[Script] Running drift-triggered retraining...")
            trainer.train_on_drift(run_name=args.run_name)
        else:
            print("[Script] Running manual training...")
            trainer.train(run_name=args.run_name)
    except Exception:
        if event_id:
            db.mark_training_event_failed(event_id)
        publisher.publish('training.failed', {
            'event_id': event_id,
            'trigger': effective_trigger,
            'run_name': args.run_name,
        })
        raise
    else:
        if event_id:
            db.mark_training_event_completed(event_id)
        publisher.publish('training.completed', {
            'event_id': event_id,
            'trigger': effective_trigger,
            'run_name': args.run_name,
        })

    if args.enable_dvc and args.dvc_add:
        targets = [target.strip() for target in args.dvc_targets.split(',') if target.strip()]
        for target in targets:
            _run_dvc(['dvc', 'add', target])

    if args.enable_dvc and args.dvc_push:
        _run_dvc(['dvc', 'push'])

    elapsed = datetime.now() - started_at
    print(f"[Script] Training completed in {elapsed}.")
    if event_id:
        print(f"[Script] Event tracked as: {event_id}")


if __name__ == '__main__':
    main()
