from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import app, main as root_main  # noqa: E402


def main():
    root_main()


if __name__ == "__main__":
    main()
