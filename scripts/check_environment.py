"""Print interpreter, core dependency, and optional PyTorch availability."""
import importlib
import importlib.util
import platform
import sys
from importlib.metadata import version, PackageNotFoundError


def main():
    print(f"Python: {platform.python_version()}")
    print(f"Interpreter: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    for distribution in ("numpy", "pandas", "scikit-learn", "matplotlib", "tqdm"):
        try:
            print(f"{distribution}: {version(distribution)}")
        except PackageNotFoundError:
            print(f"{distribution}: not installed")
    if importlib.util.find_spec("torch") is None:
        print("PyTorch: not installed (install after selecting CPU/CUDA build)")
        return
    torch = importlib.import_module("torch")
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


if __name__ == "__main__":
    main()
