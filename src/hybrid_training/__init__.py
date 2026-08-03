"""build-your-own-hybrid-distributed-training.

A from-scratch, correctness-first implementation of the parallelism strategies
used to train large models:

* :mod:`hybrid_training.parallel.ddp` -- bucketed distributed data parallelism
* :mod:`hybrid_training.parallel.fsdp` -- parameter/gradient/optimizer sharding
* :mod:`hybrid_training.parallel.tensor_parallel` -- partitioned linear layers
* :mod:`hybrid_training.parallel.sequence_parallel` -- activation sharding
* :mod:`hybrid_training.parallel.hybrid` -- composition of all of the above
* :mod:`hybrid_training.checkpoint` -- manifest-based distributed checkpoints

Nothing here wraps ``torch.nn.parallel.DistributedDataParallel``,
``torch.distributed.fsdp``, DTensor or ``torch.distributed.checkpoint``.  The
only PyTorch distributed APIs used are the raw collectives.
"""

from .config import ExperimentConfig, TopologyConfig, load_experiment_config
from .errors import HybridTrainingError
from .logging import configure_logging, get_logger

__version__ = "0.1.0"

__all__ = [
    "ExperimentConfig",
    "HybridTrainingError",
    "TopologyConfig",
    "__version__",
    "configure_logging",
    "get_logger",
    "load_experiment_config",
]
