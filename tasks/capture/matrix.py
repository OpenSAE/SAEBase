"""Public entry point for running or reading a capture matrix."""

from tasks.capture.outputs.matrix import CaptureMatrix as CaptureMatrix
from tasks.capture.outputs.matrix import main


if __name__ == "__main__":
    raise SystemExit(main())
