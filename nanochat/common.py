"""
Common utilities for nanochat (TPU edition).
"""

import os
import re
import logging
import urllib.request
import torch
import torch.distributed as dist
from filelock import FileLock


def tpu_is_available() -> bool:
    """True when TorchTPU is installed and a TPU device is visible.

    Importing torch_tpu registers the ``tpu`` backend with PyTorch. A missing
    package or a machine without a chip both report unavailable.
    """
    try:
        import torch_tpu  # noqa: F401  registers torch.tpu
    except ImportError:
        return False
    accelerator = torch.accelerator.current_accelerator()
    if accelerator is None or accelerator.type != "tpu":
        return False
    return torch.accelerator.is_available()


# The dtype used for compute (matmuls, activations). Master weights stay fp32 for optimizer precision.
# Linear layers cast their weights to this dtype in forward.
# Override with NANOCHAT_DTYPE env var: "bfloat16", "float32"
_DTYPE_MAP = {"bfloat16": torch.bfloat16, "float32": torch.float32}
def _detect_compute_dtype():
    env = os.environ.get("NANOCHAT_DTYPE")
    if env is not None:
        return _DTYPE_MAP[env], f"set via NANOCHAT_DTYPE={env}"
    return torch.bfloat16, "TPU (bf16 MXU)"
COMPUTE_DTYPE, COMPUTE_DTYPE_REASON = _detect_compute_dtype()


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages."""
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
        message = super().format(record)
        if levelname == 'INFO':
            message = re.sub(r'(\d+\.?\d*\s*(?:GB|MB|%|docs))', rf'{self.BOLD}\1{self.RESET}', message)
            message = re.sub(r'(Shard \d+)', rf'{self.COLORS["INFO"]}{self.BOLD}\1{self.RESET}', message)
        return message


def setup_default_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler]
    )


setup_default_logging()
logger = logging.getLogger(__name__)


def get_base_dir():
    if os.environ.get("NANOCHAT_BASE_DIR"):
        nanochat_dir = os.environ.get("NANOCHAT_BASE_DIR")
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        nanochat_dir = os.path.join(cache_dir, "nanochat")
    os.makedirs(nanochat_dir, exist_ok=True)
    return nanochat_dir


def download_file_with_lock(url, filename, postprocess_fn=None):
    """
    Downloads a file from a URL to a local path in the base directory.
    Uses a lock file to prevent concurrent downloads among multiple ranks.
    """
    base_dir = get_base_dir()
    file_path = os.path.join(base_dir, filename)
    lock_path = file_path + ".lock"

    if os.path.exists(file_path):
        return file_path

    with FileLock(lock_path):
        if os.path.exists(file_path):
            return file_path

        print(f"Downloading {url}...")
        with urllib.request.urlopen(url) as response:
            content = response.read()

        with open(file_path, 'wb') as f:
            f.write(content)
        print(f"Downloaded to {file_path}")

        if postprocess_fn is not None:
            postprocess_fn(file_path)

    return file_path


def print0(s="", **kwargs):
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)


def print_banner():
    banner = """
                                                       █████                █████
                                                      ░░███                ░░███
     ████████    ██████   ████████    ██████   ██████  ░███████    ██████  ███████
    ░░███░░███  ░░░░░███ ░░███░░███  ███░░███ ███░░███ ░███░░███  ░░░░░███░░░███░
     ░███ ░███   ███████  ░███ ░███ ░███ ░███░███ ░░░  ░███ ░███   ███████  ░███
     ░███ ░███  ███░░███  ░███ ░███ ░███ ░███░███  ███ ░███ ░███  ███░░███  ░███ ███
     ████ █████░░████████ ████ █████░░██████ ░░██████  ████ █████░░███████  ░░█████
    ░░░░ ░░░░░  ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░░░   ░░░░░
    """
    print0(banner)


def is_ddp_requested() -> bool:
    """True if launched with distributed env vars (RANK, LOCAL_RANK, WORLD_SIZE)."""
    return all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))


def is_ddp_initialized() -> bool:
    """True if torch.distributed is available and the process group is initialized."""
    return dist.is_available() and dist.is_initialized()


def get_dist_info():
    if is_ddp_requested():
        assert all(var in os.environ for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1


def autodetect_device_type():
    device_type = "tpu" if tpu_is_available() else "cpu"
    print0(f"Autodetected device type: {device_type}")
    return device_type


def compute_init(device_type="tpu"):
    """Initialize TPU device, RNG seeds, and optional multi-chip process group."""
    assert device_type in ["tpu", "cpu"], f"Unsupported device type: {device_type}"
    if device_type == "tpu":
        import torch_tpu  # noqa: F401
        assert tpu_is_available(), "TorchTPU is installed but no TPU is visible. device_type is 'tpu'"

    torch.manual_seed(42)
    if device_type == "tpu":
        torch.tpu.manual_seed(42)

    # Launch a multi-chip TPU job with one process per chip, after:
    #   eval $(python -m torch.tpu.distributed | sed 's/^/export /')
    ddp_requested, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if ddp_requested and device_type == "tpu":
        device = torch.device("tpu")
        dist.init_process_group(backend="tpu_dist")
        dist.barrier()
    elif device_type == "tpu":
        device = torch.device("tpu")
    else:
        device = torch.device("cpu")

    if ddp_rank == 0:
        logger.info(f"Distributed world size: {ddp_world_size}")

    return ddp_requested, ddp_rank, ddp_local_rank, ddp_world_size, device


def compile_model(model, *, dynamic=False, **kwargs):
    """torch.compile a module using the TorchTPU backend on TPU."""
    try:
        device_type = next(model.parameters()).device.type
    except StopIteration:
        device_type = "cpu"
    if device_type == "tpu":
        kwargs.setdefault("backend", "tpu")
    return torch.compile(model, dynamic=dynamic, **kwargs)


def device_synchronize(device_type="tpu"):
    """Device-wide sync callable."""
    if device_type == "tpu":
        return torch.accelerator.synchronize
    return lambda *args, **kwargs: None


def device_peak_memory(device_type="tpu"):
    """Peak allocator bytes on TPU."""
    if device_type == "tpu":
        return torch.accelerator.max_memory_allocated
    return lambda *args, **kwargs: 0


def get_device_name(device_type: str = "tpu") -> str:
    """Human-readable TPU accelerator name for MFU tables."""
    if device_type == "tpu":
        return torch.tpu.get_device_properties().name
    return device_type


def compute_cleanup():
    """Destroy process group before script exit."""
    if is_ddp_initialized():
        dist.destroy_process_group()


class DummyWandb:
    """No-op WandB replacement when logging is disabled."""
    def __init__(self):
        pass
    def log(self, *args, **kwargs):
        pass
    def finish(self):
        pass


def get_peak_flops(device_name: str) -> float:
    """Dense BF16 peak FLOPs per Google Cloud TPU chip."""
    name = device_name.lower()
    _PEAK_FLOPS_TABLE = (
        (["ironwood"], 4.614e15),
        (["tpu7"], 4.614e15),
        (["v7"], 4.614e15),
        (["trillium"], 918e12),
        (["v6e"], 918e12),
        (["v5e"], 197e12),
        (["v5litepod"], 197e12),
        (["v5p"], 459e12),
        (["v4"], 275e12),
    )
    for patterns, flops in _PEAK_FLOPS_TABLE:
        if all(p in name for p in patterns):
            return flops

    logger.warning(f"Peak flops undefined for: {device_name}, MFU will show as 0%")
    return float('inf')


def get_peak_bandwidth(device_name: str) -> float:
    """Peak HBM memory bandwidth in bytes/sec per Google Cloud TPU chip."""
    name = device_name.lower()
    _PEAK_BANDWIDTH_TABLE = (
        (["ironwood"], 7.37e12),
        (["tpu7"], 7.37e12),
        (["v7"], 7.37e12),
        (["trillium"], 1.64e12),
        (["v6e"], 1.64e12),
        (["v5p"], 2.765e12),
        (["v5e"], 819e9),
        (["v5litepod"], 819e9),
        (["v4"], 1.2e12),
    )
    for patterns, bandwidth in _PEAK_BANDWIDTH_TABLE:
        if all(p in name for p in patterns):
            return bandwidth

    logger.warning(f"Peak bandwidth undefined for: {device_name}, MBU will show as 0%")
    return float('inf')
