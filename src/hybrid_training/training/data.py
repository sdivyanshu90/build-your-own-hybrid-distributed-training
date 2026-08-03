"""Deterministic synthetic data and a topology-aware sampler.

Why synthetic data
==================
Every correctness claim in this project has the shape "distributed run ==
single-process run".  That comparison is only meaningful if both runs see
*exactly* the same samples in the same order, which means the dataset must be:

* reproducible from a seed alone (no files, no downloads, no shuffling that
  depends on world size);
* **indexable**, so that sample ``i`` is the same tensor no matter which rank
  asks for it or how many ranks there are.

Both datasets here therefore generate sample ``i`` from a generator seeded by
``(seed, i)``.  Changing the world size changes *who* reads a sample, never
*what* the sample is.

The data is also *learnable*: the MLP targets come from a fixed random teacher
network and the token sequences follow a deterministic recurrence with noise.
That matters because several end-to-end tests assert that the loss actually
decreases, which a purely random target would make impossible.

Who reads what
==============
The sampler indexes by the rank's coordinate in the ``dp_shard`` group -- the
set of ranks that process *different* data.  Ranks in the same
``tensor_sequence`` group get the **same** indices, because they are
collaborating on one batch: rank 0 computes attention heads 0-3 of a sample and
rank 1 computes heads 4-7 of the *same* sample.  Feeding them different samples
produces a loss curve that looks plausible and is wrong.

Uneven batches are not allowed.  If the dataset does not divide evenly, the
tail is dropped rather than producing a short batch on one rank: DDP's
gradient average assumes equal per-rank sample counts, and a short batch would
silently up-weight the ranks that got a full one.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch

from ..config import DataConfig, ModelConfig
from ..distributed.groups import GroupHandle
from ..errors import ConfigurationError, format_error
from ..logging import get_logger
from ..utils.reproducibility import derive_seed

__all__ = [
    "Batch",
    "DistributedBatchSampler",
    "SyntheticDataset",
    "SyntheticMLPDataset",
    "SyntheticTokenDataset",
    "build_dataset",
]

_LOGGER = get_logger(__name__)


@dataclass
class Batch:
    """One training batch.

    Attributes:
        inputs: ``(batch, features)`` floats for the MLP, or ``(batch,
            sequence)`` integer token ids for the transformer.
        targets: ``(batch, outputs)`` floats, or ``(batch, sequence)`` next-token
            ids.
        indices: Global dataset indices of the samples, kept so tests can prove
            which rank read which sample.
    """

    inputs: torch.Tensor
    targets: torch.Tensor
    indices: tuple[int, ...]

    def to(self, device: torch.device) -> Batch:
        """Move the batch to ``device``.

        Args:
            device: Destination device.

        Returns:
            A new :class:`Batch` on ``device``.
        """
        return Batch(
            inputs=self.inputs.to(device, non_blocking=True),
            targets=self.targets.to(device, non_blocking=True),
            indices=self.indices,
        )

    @property
    def size(self) -> int:
        """Number of samples in the batch."""
        return int(self.inputs.shape[0])


class SyntheticDataset:
    """Base class for the indexable synthetic datasets.

    Args:
        num_samples: Number of samples.
        seed: Master seed; sample ``i`` uses ``derive_seed(seed, "sample", i)``.

    Raises:
        ConfigurationError: If ``num_samples`` is not positive.
    """

    def __init__(self, num_samples: int, seed: int) -> None:
        if num_samples < 1:
            raise ConfigurationError(
                format_error(
                    "data.SyntheticDataset",
                    "a dataset needs at least one sample",
                    expected=">= 1",
                    observed=num_samples,
                    resolution="raise num_train_samples / num_eval_samples",
                )
            )
        self._num_samples = num_samples
        self._seed = seed

    def __len__(self) -> int:
        """Number of samples."""
        return self._num_samples

    def _generator(self, index: int) -> torch.Generator:
        """Return a CPU generator seeded for one sample."""
        generator = torch.Generator()
        generator.manual_seed(derive_seed(self._seed, "sample", index))
        return generator

    def get(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(inputs, targets)`` for one sample.

        Args:
            index: Sample index in ``[0, len(self))``.

        Returns:
            The sample.

        Raises:
            NotImplementedError: In the base class.
        """
        raise NotImplementedError

    def batch(self, indices: tuple[int, ...]) -> Batch:
        """Assemble a batch from sample indices.

        Args:
            indices: Sample indices.

        Returns:
            The stacked batch.
        """
        samples = [self.get(index) for index in indices]
        return Batch(
            inputs=torch.stack([s[0] for s in samples]),
            targets=torch.stack([s[1] for s in samples]),
            indices=tuple(indices),
        )


class SyntheticMLPDataset(SyntheticDataset):
    """Regression data from a fixed random teacher network.

    ``y = tanh(x A) B + eps`` with ``A`` and ``B`` drawn once from a seed
    derived from ``seed``, and ``eps`` a small per-sample noise term.  A model
    with the same architecture can fit this, so the loss genuinely decreases,
    while the teacher's fixed weights keep the target reproducible.

    Args:
        num_samples: Number of samples.
        seed: Master seed.
        input_size: Input width.
        output_size: Target width.
        hidden_size: Teacher hidden width.
        noise_std: Standard deviation of the additive noise.
    """

    def __init__(
        self,
        num_samples: int,
        seed: int,
        *,
        input_size: int,
        output_size: int,
        hidden_size: int = 32,
        noise_std: float = 0.01,
    ) -> None:
        super().__init__(num_samples, seed)
        self.input_size = input_size
        self.output_size = output_size
        self.noise_std = noise_std
        teacher = torch.Generator()
        teacher.manual_seed(derive_seed(seed, "teacher"))
        self._a = torch.randn(input_size, hidden_size, generator=teacher) / (input_size**0.5)
        self._b = torch.randn(hidden_size, output_size, generator=teacher) / (hidden_size**0.5)

    def get(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one ``(features, target)`` pair.

        Args:
            index: Sample index.

        Returns:
            ``(input_size,)`` and ``(output_size,)`` float tensors.
        """
        generator = self._generator(index)
        x = torch.randn(self.input_size, generator=generator)
        y = torch.tanh(x @ self._a) @ self._b
        y = y + self.noise_std * torch.randn(self.output_size, generator=generator)
        return x, y


class SyntheticTokenDataset(SyntheticDataset):
    """Next-token prediction over a deterministic recurrence with noise.

    Token ``t+1`` is ``(a * token_t + b) mod vocab`` most of the time, and a
    uniform random token otherwise.  The rule is learnable, so a transformer's
    loss drops well below ``log(vocab)``, while the noise keeps the task from
    being trivially memorisable in two steps.

    Args:
        num_samples: Number of sequences.
        seed: Master seed.
        vocab_size: Vocabulary size.
        sequence_length: Tokens per sample.  The target is the input shifted by
            one, with the final target position sampled from the recurrence.
        noise_probability: Probability that a token is drawn uniformly instead
            of following the recurrence.

    Raises:
        ConfigurationError: If ``sequence_length`` is less than 2.
    """

    def __init__(
        self,
        num_samples: int,
        seed: int,
        *,
        vocab_size: int,
        sequence_length: int,
        noise_probability: float = 0.1,
    ) -> None:
        super().__init__(num_samples, seed)
        if sequence_length < 2:
            raise ConfigurationError(
                format_error(
                    "data.SyntheticTokenDataset",
                    "next-token prediction needs at least two positions",
                    expected=">= 2",
                    observed=sequence_length,
                    resolution="raise DataConfig.sequence_length",
                )
            )
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.noise_probability = noise_probability
        # Coprime with most vocabularies, so the recurrence has a long period.
        self._multiplier = 7
        self._offset = 3

    def get(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one ``(tokens, next_tokens)`` pair.

        Args:
            index: Sample index.

        Returns:
            Two ``(sequence_length,)`` int64 tensors.
        """
        generator = self._generator(index)
        length = self.sequence_length + 1
        tokens = torch.empty(length, dtype=torch.long)
        tokens[0] = int(torch.randint(0, self.vocab_size, (1,), generator=generator).item())
        noise = torch.rand(length, generator=generator)
        random_tokens = torch.randint(0, self.vocab_size, (length,), generator=generator)
        for position in range(1, length):
            if float(noise[position]) < self.noise_probability:
                tokens[position] = random_tokens[position]
            else:
                tokens[position] = (
                    self._multiplier * int(tokens[position - 1]) + self._offset
                ) % self.vocab_size
        return tokens[:-1].contiguous(), tokens[1:].contiguous()


def build_dataset(
    model_config: ModelConfig, data_config: DataConfig, *, split: str = "train"
) -> SyntheticDataset:
    """Construct the dataset matching a model kind.

    Args:
        model_config: Model configuration; selects the dataset type.
        data_config: Dataset sizes and seed.
        split: ``"train"`` or ``"eval"``.  The two splits use different derived
            seeds, so evaluation samples never appear in training.

    Returns:
        The dataset.

    Raises:
        ConfigurationError: If ``split`` is unknown.
    """
    if split not in {"train", "eval"}:
        raise ConfigurationError(
            format_error(
                "data.build_dataset",
                "unknown split",
                expected=["train", "eval"],
                observed=split,
                resolution="use 'train' or 'eval'",
            )
        )
    num_samples = (
        data_config.num_train_samples if split == "train" else data_config.num_eval_samples
    )
    seed = derive_seed(data_config.seed, f"dataset-{split}")
    if model_config.kind == "transformer":
        return SyntheticTokenDataset(
            max(num_samples, 1),
            seed,
            vocab_size=model_config.vocab_size,
            sequence_length=data_config.sequence_length,
        )
    return SyntheticMLPDataset(
        max(num_samples, 1),
        seed,
        input_size=model_config.input_size,
        output_size=model_config.output_size,
    )


class DistributedBatchSampler:
    """Deterministic, topology-aware batch sampler.

    Args:
        dataset_size: Number of samples.
        micro_batch_size: Samples per data-parallel rank per micro-step.
        data_group: Group whose coordinate selects the data slice -- the
            ``dp_shard`` group.  **Required**: using the world group here would
            give tensor-parallel peers different samples.
        shuffle: Shuffle the sample order each epoch.
        seed: Shuffle seed.  All ranks use the same seed and the same
            permutation, then take disjoint slices of it, so the union of the
            ranks' batches is exactly the global batch.
        drop_last: Discard a trailing partial global batch.  Forced ``True``;
            see the module docstring.

    Raises:
        ConfigurationError: If the dataset is too small for one global batch.
    """

    def __init__(
        self,
        dataset_size: int,
        micro_batch_size: int,
        data_group: GroupHandle,
        *,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        self.dataset_size = dataset_size
        self.micro_batch_size = micro_batch_size
        self.data_group = data_group
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.global_batch_size = micro_batch_size * data_group.size
        if dataset_size < self.global_batch_size:
            raise ConfigurationError(
                format_error(
                    "data.DistributedBatchSampler",
                    "the dataset is smaller than one global batch, so no rank could be "
                    "given a full micro-batch",
                    rank=data_group.global_rank,
                    expected=f">= {self.global_batch_size} samples",
                    observed=dataset_size,
                    resolution=(
                        "raise num_train_samples, or lower micro_batch_size / the "
                        "data-parallel size"
                    ),
                )
            )

    @property
    def batches_per_epoch(self) -> int:
        """Number of micro-batches this rank yields per epoch."""
        return self.dataset_size // self.global_batch_size

    def epoch_order(self, epoch: int) -> torch.Tensor:
        """Return the global sample order for one epoch.

        Identical on every rank, which is what makes the per-rank slices
        disjoint and their union the global batch.

        Args:
            epoch: Epoch index.

        Returns:
            A permutation (or ``arange``) of the dataset indices.
        """
        if not self.shuffle:
            return torch.arange(self.dataset_size)
        generator = torch.Generator()
        generator.manual_seed(derive_seed(self.seed, "shuffle", epoch))
        return torch.randperm(self.dataset_size, generator=generator)

    def __iter__(self) -> Iterator[tuple[int, ...]]:
        """Yield this rank's micro-batch indices for epoch 0."""
        return self.iter_epoch(0)

    def iter_epoch(self, epoch: int) -> Iterator[tuple[int, ...]]:
        """Yield this rank's micro-batch indices for a given epoch.

        Args:
            epoch: Epoch index.

        Yields:
            Tuples of ``micro_batch_size`` global sample indices.
        """
        order = self.epoch_order(epoch)
        local = self.data_group.local_rank
        for batch_index in range(self.batches_per_epoch):
            start = batch_index * self.global_batch_size + local * self.micro_batch_size
            yield tuple(int(i) for i in order[start : start + self.micro_batch_size])

    def global_batch_indices(self, epoch: int, batch_index: int) -> tuple[int, ...]:
        """Return the indices of the whole global batch.

        Used by the single-process reference in the equivalence tests, which
        must consume exactly the union of what the distributed ranks consumed.

        Args:
            epoch: Epoch index.
            batch_index: Index of the batch within the epoch.

        Returns:
            ``global_batch_size`` sample indices.
        """
        order = self.epoch_order(epoch)
        start = batch_index * self.global_batch_size
        return tuple(int(i) for i in order[start : start + self.global_batch_size])


class SyntheticDataLoader:
    """Iterates a :class:`SyntheticDataset` using a :class:`DistributedBatchSampler`.

    Args:
        dataset: Source dataset.
        sampler: Index source.
        device: Device batches are moved to.

    Example:
        >>> # doctest: +SKIP
        >>> loader = SyntheticDataLoader(dataset, sampler, device)
        >>> for batch in loader.iter_epoch(0):
        ...     loss = criterion(model(batch.inputs), batch.targets)
    """

    def __init__(
        self,
        dataset: SyntheticDataset,
        sampler: DistributedBatchSampler,
        device: torch.device,
    ) -> None:
        self.dataset = dataset
        self.sampler = sampler
        self.device = device

    def __len__(self) -> int:
        """Micro-batches per epoch."""
        return self.sampler.batches_per_epoch

    def iter_epoch(self, epoch: int) -> Iterator[Batch]:
        """Yield this rank's batches for an epoch.

        Args:
            epoch: Epoch index.

        Yields:
            Batches already moved to the loader's device.
        """
        for indices in self.sampler.iter_epoch(epoch):
            yield self.dataset.batch(indices).to(self.device)

    def global_batch(self, epoch: int, batch_index: int) -> Batch:
        """Return the full global batch, for reference comparisons.

        Args:
            epoch: Epoch index.
            batch_index: Batch index within the epoch.

        Returns:
            The global batch on the loader's device.
        """
        indices = self.sampler.global_batch_indices(epoch, batch_index)
        return self.dataset.batch(indices).to(self.device)
