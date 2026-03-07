import subprocess
import sys


def main():
    """Launch Vision ML multi-page Streamlit app."""
    print("🚀 Launching Vision ML System...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "home.py"])


if __name__ == "__main__":
    main()
