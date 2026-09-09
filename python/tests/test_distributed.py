import os
from pathlib import Path
import subprocess
import sys
import pytest


@pytest.mark.distributed
def test_persistent_backends_graph_and_symmetric_leases():
    worker = Path(__file__).with_name("distributed_worker.py")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(worker),
        ],
        check=True,
        timeout=1800,
        env=os.environ.copy(),
    )


@pytest.mark.distributed
def test_persistent_tensor_operations_and_fp32_statistics():
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(Path(__file__).with_name("distributed_tensor_worker.py")),
        ],
        check=True,
        timeout=1800,
        env=os.environ.copy(),
    )
