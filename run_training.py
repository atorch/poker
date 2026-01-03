#!/usr/bin/env python3
"""
Wrapper script to run poker training with output logged to both terminal and file.

Usage:
    uv run python run_training.py

Output will be displayed in terminal AND saved to:
    training_output_YYYY_MM_DD_<description>.txt
"""

import subprocess
import sys
from datetime import datetime


def main():
    # Generate timestamped filename with hour/minute to avoid overwrites
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M")
    description = "full_curriculum_128x128_batch8"
    output_file = f"training_output_{timestamp}_{description}.txt"

    print(f"Starting training...")
    print(f"Output will be saved to: {output_file}")
    print(f"=" * 70)
    print()

    # Run the training command
    cmd = ["uv", "run", "python", "poker/play.py"]

    # Use subprocess.Popen to capture output in real-time
    with open(output_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1  # Line buffered
        )

        # Read and write output line by line
        for line in process.stdout:
            # Write to terminal
            sys.stdout.write(line)
            sys.stdout.flush()
            # Write to file
            f.write(line)
            f.flush()

        process.wait()

    print()
    print(f"=" * 70)
    print(f"Training complete! Output saved to: {output_file}")
    print(f"=" * 70)

    return process.returncode


if __name__ == "__main__":
    sys.exit(main())
