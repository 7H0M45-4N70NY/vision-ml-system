import argparse
import subprocess


def _run(cmd):
    print(f"[DVC] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description='DVC CLI helper for Vision ML System')
    parser.add_argument('--action', required=True, choices=['pull', 'status', 'add', 'push'])
    parser.add_argument('--targets', type=str, default='runs/train', help='Comma-separated paths for add action')
    args = parser.parse_args()

    if args.action == 'pull':
        _run(['dvc', 'pull'])
    elif args.action == 'status':
        _run(['dvc', 'status'])
    elif args.action == 'add':
        targets = [t.strip() for t in args.targets.split(',') if t.strip()]
        for target in targets:
            _run(['dvc', 'add', target])
    elif args.action == 'push':
        _run(['dvc', 'push'])


if __name__ == '__main__':
    main()
