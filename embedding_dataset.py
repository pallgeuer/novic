# Generic dataset of embeddings and target nouns

# Imports
from __future__ import annotations
import warnings
import itertools
import contextlib
import collections
import dataclasses
from typing import ContextManager, Iterable, Optional, Union, Any
import torch.utils.data
from logger import log
import embedders

# Ignore warnings: In order to provide ultra-fast zero-copy torch tensors backed by a memory-mapped cache file in some dataset implementations (using torch.frombuffer()), we need to be responsible and NEVER attempt to write to the constructed PyTorch tensors (otherwise a SIGSEGV occurs and the program crashes, but the cache file will never be touched so it is at least 'safe' in that respect)
warnings.filterwarnings("ignore", message="The given buffer is not writable, and PyTorch does not support non-writable tensors.", category=UserWarning)

# Embedding dataset data configuration class
@dataclasses.dataclass(frozen=True)
class DataConfig:             # Important: The first target of every embedding must be non-fully-padded and have non-zero weight, every fully-padded target must have zero weight, fully-padded targets must come after non-fully-padded targets, and weights must be non-negative values in descending order (implies order of targets is strictly: non-fully-padded non-zero-weighted -> non-fully-padded zero-weighted -> fully-padded zero-weighted)

	use_weights: bool         # Whether target weight tensors should be included in addition to the provided target tokenizations, or whether the target weights should always just be None (should be False if not using targets)
	unit_weights: bool        # If use_weights (otherwise True), whether the target weights for each embedding must sum to 1 (fixed_multi_length=False is not a problem as it can only omit zero-weighted targets as all fully-padded targets must have zero weight)
	multi_target: bool        # Whether multiple targets per embedding should be provided (this adds a multiple targets dimension to the target tokenizations and weights, should be False if not using targets, first target per embedding must always be non-fully-padded and have non-zero weight)
	multi_first: bool         # If multi_target (otherwise False), whether the added multiple targets dimension comes before or after the batch dimension
	full_targets: bool        # If multi_target (otherwise True), whether every embedding has the same full number of non-fully-padded targets as given by multi_length, i.e. none of the multi-targets are ever fully padded
	fixed_multi_length: bool  # If multi_target (otherwise True), whether the multiple targets dimension should always have a fixed value for all data (given by multi_length, some targets can be fully-padded/weights can be zero) or whether it can be any value from 1 to multi_length (trailing fully-padded and/or zero-weighted targets can be excluded)
	multi_length: int         # If multi_target (otherwise 1), the maximum number of multiple targets per embedding

	@staticmethod
	def create(data_config_dict: dict[str, Union[bool, int]], use_targets: bool = True) -> DataConfig:
		data_config_dict = data_config_dict.copy()
		if not use_targets:
			data_config_dict.update(use_weights=False, multi_target=False)
		if not data_config_dict['use_weights']:
			data_config_dict.update(unit_weights=True)
		if not data_config_dict['multi_target']:
			data_config_dict.update(multi_first=False, full_targets=True, fixed_multi_length=True, multi_length=1)
		data_config = DataConfig(**data_config_dict)
		if data_config.multi_length < 1:
			raise ValueError(f"Number of multi-targets needs to be positive: {data_config.multi_length}")
		return data_config

# Embedding dataset loader information class
@dataclasses.dataclass(frozen=True)
class LoaderInfo:
	num_workers: int        # Number of worker processes (can be 0 for data loading in main process)
	prefetch_factor: int    # Prefetch factor (number of batches loaded in advance per worker, always 0 if no worker processes)
	pin_memory: bool        # Whether all returned tensors are pinned CPU tensors (True if device is not CPU and no outputs are tensors)
	on_device: bool         # Whether the returned batches are already on the correct device (e.g. always True if device is CPU, and False if device is not CPU and no outputs are tensors)
	batch_size: int         # Batch size of all batches except possibly the last batch
	batch_size_last: int    # Size of the incomplete last batch (0 = Incomplete batch not permitted or necessary, 1 to batch_size-1 = Incomplete batch size)
	complete_batches: int   # Number of complete batches per epoch
	incomplete_batch: bool  # Whether a trailing incomplete batch is part of the epoch (Note: bool can be added numerically to ints for logic)
	epoch_batches: int      # Number of batches per epoch = complete_batches + incomplete_batch
	epoch_samples: int      # Number of samples per epoch = complete_batches * batch_size + incomplete_batch * batch_size_last
	available_samples: int  # Number of available samples from which the samples for each epoch are chosen

# Embedding dataset class
class EmbeddingDataset(torch.utils.data.Dataset):

	def __init__(
		self,
		embedder: embedders.Embedder,               # Embedder that is to be used in the context of the embedding dataset (any existing target_config is ignored for now)
		nominal_data_config: DataConfig,            # Nominal data configuration to use for the dataset
		strict_data_config_fields: set[str],        # Fields in the nominal data configuration that are non-negotiable and cannot change due to dataset limitations
		num_items: int,                             # Number of items in the dataset (see __getitem__)
		num_embeds: Optional[int] = None,           # Number of samples in the dataset (may be different to num_items as each item may already be a batch of samples, taken as equal to num_items if None)
		targets: Optional[tuple[str, ...]] = None,  # Tuple of exactly all the target nouns that exist in the dataset (if empty tuple or None then dataset is considered not to have targets, i.e. dataset of embedding vectors only)
		num_invalid_targets: int = 0,               # How many of the FIRST/LEADING target nouns in 'targets' are invalid, in the sense of representing a padding/placeholder target noun (e.g. for multi-target datasets)
		use_targets: bool = True,                   # Whether target information should be returned by the data loader or whether these outputs (i.e. target token IDs, target token padding masks, target weights) should always be None (cannot be True if not targets)
	):

		self.embedder = embedder
		self.nominal_data_config = nominal_data_config
		self.strict_data_config_fields = strict_data_config_fields
		self.num_items = num_items
		self.num_embeds = num_embeds if num_embeds is not None else self.num_items
		self.targets = targets
		self.have_targets = bool(self.targets)
		self.num_invalid_targets = 0 if not self.have_targets else num_invalid_targets
		self.num_valid_targets = 0 if not self.have_targets else len(self.targets) - self.num_invalid_targets
		self.use_targets = use_targets

		if self.strict_data_config_fields.difference(field.name for field in dataclasses.fields(DataConfig)):
			raise ValueError("Invalid DataConfig field(s) were specified to be strict")
		if self.num_items < 1 or self.num_embeds < 1 or self.num_items > self.num_embeds:
			raise ValueError("Empty or invalid embedding dataset")
		if self.have_targets and (self.num_valid_targets > len(self.targets) or self.num_valid_targets < 1):
			raise ValueError(f"Number of valid targets must be positive and cannot exceed the number of targets provided: {self.num_valid_targets}")
		if self.use_targets:
			if not self.have_targets:
				raise ValueError("Cannot use targets if dataset has none")
			targets_count = collections.Counter(self.targets)
			if any(count > 1 for target, count in targets_count.items() if target) or targets_count[''] > 2:
				raise ValueError(f"There are duplicates in the dataset: {sorted(item for item in targets_count.items() if item[1] > (1 if item[0] else 2))}")

		self.translation: Optional[embedders.TargetConfig] = None
		self.data_config: Optional[DataConfig] = None

	def __len__(self) -> int:
		# Return the length of the dataset in terms of items and __getitem__
		return self.num_items

	def set_translation(self, target_config: Optional[embedders.TargetConfig]):
		# Set a target configuration translation to occur (the fixed_token_length, token_length and use_masks fields are not considered as part of translation)
		# If using targets, the embedder must have a valid target_config before this method is called
		if target_config is not None:
			if self.use_targets:
				for field in ('fixed_token_length', 'token_length', 'use_masks'):
					if getattr(target_config, field) != getattr(self.embedder.target_config, field):
						log.warning(f"Translation target config {field} has value mismatch: Dataset {getattr(self.embedder.target_config, field)} vs Translation {getattr(target_config, field)} (returned data will respect the former)")
				if target_config.compact_ids != self.embedder.target_config.compact_ids:
					raise ValueError("Translation cannot change whether target tokenizations are compact")
				if (target_config.start_token_id is None) != (self.embedder.target_config.start_token_id is None) or (target_config.end_token_id is None) != (self.embedder.target_config.end_token_id is None):
					raise ValueError("Translation cannot change whether start and/or end tokens are present")
				target_config = dataclasses.replace(target_config, fixed_token_length=self.embedder.target_config.fixed_token_length, token_length=self.embedder.target_config.token_length, use_masks=self.embedder.target_config.use_masks)
			else:
				raise ValueError("Cannot set a non-None translation for an embedding dataset without targets")
		self.translation = target_config

	def resolve_data_config(self, **data_kwargs) -> DataConfig:
		# Return a resolved data configuration based on what is externally desired (data_kwargs), internally available (strict_data_config_fields) and/or internally preferred (nominal_data_config)
		# The possible data_kwargs are exactly the fields of DataConfig, where a missing value is equivalent to None, and a value of None means don't care (embedding dataset can decide based on what data it has on offer)

		data_config_dict = {}
		nominal_data_config_dict = dataclasses.asdict(self.nominal_data_config)
		for field_name, nominal_value in nominal_data_config_dict.items():
			kwarg_value = data_kwargs.pop(field_name, None)
			data_config_dict[field_name] = kwarg_value if kwarg_value is not None else nominal_value
		if data_kwargs:
			raise ValueError(f"Cannot resolve invalid data config fields: {sorted(data_kwargs.keys())}")

		strict_data_config_fields = self.strict_data_config_fields.copy()
		for field_name in self.strict_data_config_fields:
			if data_config_dict[field_name] == nominal_data_config_dict[field_name]:
				strict_data_config_fields.discard(field_name)

		data_config = DataConfig.create(data_config_dict=data_config_dict, use_targets=self.use_targets)
		if data_config.multi_length > self.nominal_data_config.multi_length:
			raise ValueError(f"This embedding dataset does not support a number of multi-targets above {self.nominal_data_config.multi_length}: {data_config.multi_length}")

		for field_name in self.strict_data_config_fields:
			if getattr(data_config, field_name) == nominal_data_config_dict[field_name]:
				strict_data_config_fields.discard(field_name)
		if strict_data_config_fields:
			raise ValueError(f"Incompatibility between embedding dataset and requested data config in fields: {sorted(strict_data_config_fields)}")

		return data_config

	def configure_data(self, data_config: DataConfig):
		# If using targets, then the embedder must have a valid target_config
		self.data_config = data_config
		if not self.data_config.use_weights and self.nominal_data_config.use_weights:
			log.warning("Information is being lost: Specified data configuration will ignore weights even though they are non-trivial")
		if self.data_config.multi_length < self.nominal_data_config.multi_length:
			log.warning(f"Information is being lost: Specified data configuration will ignore some targets actually available in the dataset due to a reduced multi-target length ({self.data_config.multi_length} < {self.nominal_data_config.multi_length})")
		if self.use_targets and not self.embedder.target_config.use_masks and not self.data_config.use_weights and not self.data_config.full_targets:
			raise RuntimeError("When using non-full targets without padding masks and without weights, there is no robust way of being able to tell which targets are supposed to be ignored")

	# noinspection PyMethodMayBeStatic
	def loaded(self) -> ContextManager:
		# Return a context manager that must be in the entered state whenever and covering the complete duration that __getitem__ is called in user code
		# If using targets, then the embedder must have a valid target_config by the time the context manager returned by this method is entered
		# A valid data configuration must have been configured by the time the context manager returned by this method is entered
		return contextlib.nullcontext()

	def __getitem__(self, index) -> Any:
		# Get an item by index from the dataset (subclass implementations should provide a more specific output type annotation)
		# The exact output type annotation is not so important as long as the loader based on this dataset returns standardised embedding dataset batched data (see create_loader)
		# The implementation must raise IndexError for an invalid index
		raise NotImplementedError

	def create_loader(self, batch_size: int, num_workers: int, training: bool, device: torch.device, patch: bool = True) -> tuple[torch.utils.data.DataLoader, LoaderInfo]:
		# batch_size = Batch size (B) to use for the data returned by the loader
		# num_workers = Number of worker processes to use for data loading
		# training = Whether the data loader should be in training data mode (e.g. requires data to be returned in a statistically non-systematic order, and all batches must be of nominal full size)
		# device = Device that the returned tensor batches should be on (requires patching)
		# patch = Whether the data loader should be patched (if even required) in order to satisfy the required behaviour (e.g. device), or whether the data loader should remain intentionally unpatched for raw custom handling of the data and moving thereof to the correct device
		#
		# This method should return a data loader instance (or subclass thereof) that when iterated produces a tuple of the following data:
		#  - Embedding vectors: Tensor of shape BxF
		#  - Optional (valid if use_targets=True) target token IDs:           Tensor of shape BxC (multi_target=False) or BxMxC (multi_first=False) or MxBxC (multi_first=True)
		#  - Optional (valid if use_targets=True) target token padding masks: Tensor of shape BxC (multi_target=False) or BxMxC (multi_first=False) or MxBxC (multi_first=True)
		#  - Optional (valid if use_targets=use_weights=True) target weights: Tensor of shape B   (multi_target=False) or BxM   (multi_first=False) or MxB   (multi_first=True)
		# Note: B is the batch size, F is the fixed embedding dimension, M can vary from returned batch to batch (<= multi_length) if not fixed_multi_length, C can vary from returned batch to batch if not fixed_token_length
		# If full_targets=True then no target anywhere may be pure padding (Note: The first target for each embedding can never be pure padding or zero-weighted anyway)
		# If unit_weights=True then the target weights must sum to 1 for each embedding
		# As a second return value should be a LoaderInfo instance describing the properties/sizes of the loader
		#
		# If using targets, the embedder must have a valid target_config before this method is called
		# A valid data configuration must have been configured before this method is called
		# When the created data loader is iterated it will use __getitem__ to retrieve data from the dataset, meaning that by that time loaded() must have been entered
		# This method in general will be called before loaded() is called and/or entered
		raise NotImplementedError

# Embedding dataset loader gradient accumulation wrapper
class GradAccum:

	batch_size: int          # Batch size of the raw data loader
	accum_batch_size: int    # Effective meta-batch size after accumulation
	complete_steps: int      # Number of accumulated optimizer steps (i.e. meta-batches) per epoch that are complete
	complete_batches: int    # Total number of raw batches that are iterated as part of complete meta-batches
	complete_samples: int    # Total number of samples that are iterated as part of complete meta-batches
	incomplete_step: bool    # Whether there is a trailing accumulated optimizer step (i.e. meta-batch) that is incomplete
	incomplete_batches: int  # Total number of raw batches that are iterated as part of the incomplete meta-batch (0 if no incomplete meta-batch)
	incomplete_samples: int  # Total number of samples that are iterated as part of the incomplete meta-batch (0 if no incomplete meta-batch)
	loader_steps: int        # Number of accumulated optimizer steps per epoch = complete_steps + incomplete_step
	loader_batches: int      # Number of raw batches that are used in each epoch by the iterator returned from loader() (i.e. the length of the iterator) = complete_batches + incomplete_batches
	loader_samples: int      # Number of samples that are used in each epoch by the iterator returned from loader() = complete_samples + incomplete_samples

	def __init__(self, loader: torch.utils.data.DataLoader, loader_info: LoaderInfo, accum_size: int, drop_last: bool):
		# loader = Data loader created with embedding_dataset.create_loader()
		# loader_info = Corresponding embedding dataset loader information
		# accum_size = Gradient accumulation size, i.e. the number of batches that lead to a single optimizer step
		# drop_last = Whether to drop trailing batches that are insufficient to form a complete accumulation size
		# If accum_size is 1, drop_last is False, and the loader actually has an incomplete batch, then note that the loss for that one batch will be rescaled proportional to how incomplete the batch is (difference to using raw loader)
		# While this accumulates GRADIENTS across multiple batches 'perfectly', it does NOT accumulate the statistics inside e.g. batch norms or similar, if these are present in the model.
		# As such, it is not necessarily absolutely identical to training with a larger batch size, but if the smaller batch size is still "large enough" the difference might not be measurable.

		self.raw_loader = loader
		self.raw_loader_info = loader_info
		self.accum_size = accum_size
		self.drop_last = drop_last

		if self.accum_size < 1:
			raise ValueError(f"Accumulation size must be at least 1: {self.accum_size}")
		assert self.raw_loader_info.epoch_batches == len(self.raw_loader)

		self.batch_size = self.raw_loader_info.batch_size
		self.accum_batch_size = self.batch_size * self.accum_size
		self.complete_steps = self.raw_loader_info.complete_batches // self.accum_size
		self.complete_batches = self.complete_steps * self.accum_size
		self.complete_samples = self.complete_batches * self.batch_size

		if self.drop_last:
			self.loader_batches = self.complete_batches
			self.loader_samples = self.complete_samples
			self.incomplete_batches = 0
			self.incomplete_samples = 0
			self.incomplete_step = False
		else:
			self.loader_batches = self.raw_loader_info.epoch_batches
			self.loader_samples = self.raw_loader_info.epoch_samples
			self.incomplete_batches = self.loader_batches - self.complete_batches
			self.incomplete_samples = self.loader_samples - self.complete_samples
			assert self.incomplete_batches >= 0 and self.incomplete_samples >= 0 and (self.incomplete_batches > 0) == (self.incomplete_samples > 0)
			self.incomplete_step = (self.incomplete_samples > 0)

		self.loader_steps = self.complete_steps + self.incomplete_step
		self.batch_num = 0

		log.info(f"Gradient accumulation factor {self.accum_size} results in {self.complete_steps}+{int(self.incomplete_step)} = {self.loader_steps} meta-batches of size {self.accum_batch_size}+{self.incomplete_samples}")
		log.info(f"Gradient accumulation is using {self.loader_batches}/{self.raw_loader_info.epoch_batches} available batches and {self.loader_samples}/{self.raw_loader_info.epoch_samples} available samples")

	def loader(self) -> Iterable[tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]]:
		# Note: The length of the returned iterable is self.loader_batches
		self.batch_num = 0
		return itertools.islice(self.raw_loader, self.loader_batches) if self.drop_last and self.loader_batches < self.raw_loader_info.epoch_batches else self.raw_loader

	def accum_loss(self, mean_batch_loss: torch.Tensor, num_in_batch: int) -> tuple[torch.Tensor, bool]:
		# mean_batch_loss = Mean loss for the current raw batch
		# num_in_batch = Number of samples in the current raw batch
		# Returns the mean batch loss scale-adjusted for gradient accumulation, and whether it is time for an optimizer step after this batch

		self.batch_num += 1
		if self.batch_num <= self.complete_batches:
			mean_accum_batch_loss = mean_batch_loss / self.accum_size
		else:
			mean_accum_batch_loss = mean_batch_loss * (num_in_batch / self.incomplete_samples)

		optimizer_step = (self.batch_num % self.accum_size == 0 or self.batch_num == self.raw_loader_info.epoch_batches)
		return mean_accum_batch_loss, optimizer_step
# EOF
