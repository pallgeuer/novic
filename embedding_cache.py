# Cached dataset of embeddings and target nouns

# Imports
from __future__ import annotations
import os
import mmap
import struct
import random
import itertools
import functools
import contextlib
import dataclasses
from typing import Optional, Sequence, BinaryIO, ContextManager, Union
import torch
import torch.utils.data
from logger import log
import embedders
import embedding_dataset

#
# General
#

# Embedding cache file format (only Header and Embedding vectors if use_targets=False):
#  - Header (see Header class)
#  - Target noun strings (R null-separated strings encoded in UTF-8, first one, i.e. token ID 0, must be the empty string and correspond to a fully-padded tokenization)
#  - Target token IDs (RxC array of int)
#  - Target token padding mask (RxC array of bool, always there even if embedder doesn't use mask padding, exactly only the first row must be fully padded)
#  - Embedding target noun IDs (NxM array of int, a value of 0 means unknown or unclassified or ignored, all non-zero values must precede all zero values, first column cannot contain zeros)
#  - Embedding target noun weights (NxM descending array of non-negative float that are same dtype as embedding vectors, a value of 0 should result in the corresponding target being essentially ignored, first column cannot contain zeros)
#  - Embedding vectors (NxF array of float unit vectors)

# Embedding cache file header class
@dataclasses.dataclass(frozen=True)
class Header:

	VERSION = 1
	MAGIC_SIZE = 32
	MAGIC_BYTES = b'\xa9\xfdK\x14*\x9a\xb8\x13m\x157\xca\xe8+\xef\x82B\x19\xdbJ\xb8\x93\xb2&\xa0\x1a=\xe4\xadR\xb1\x99'
	INIT_MAGIC_BYTES = b'\x00' * MAGIC_SIZE
	STRUCT_FORMAT = f'<{MAGIC_SIZE}sB?????32s32sLLHHHLHHHH'
	STRUCT_FACTORY = struct.Struct(format=STRUCT_FORMAT)
	assert len(MAGIC_BYTES) == len(INIT_MAGIC_BYTES) == MAGIC_SIZE and STRUCT_FACTORY.size == 128

	TARGET_EXCLUDE = {'fixed_token_length'}  # Set of field names in TargetConfig that do not affect cache generation and reuse
	assert not TARGET_EXCLUDE.difference(field.name for field in dataclasses.fields(embedders.TargetConfig))

	INT_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64)
	INT_DTYPE_ID = {dtype: dtype_id for dtype_id, dtype in enumerate(INT_DTYPES)}
	BOOL_DTYPES = (torch.bool,)
	BOOL_DTYPE_ID = {dtype: dtype_id for dtype_id, dtype in enumerate(BOOL_DTYPES)}
	FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
	FLOAT_DTYPE_ID = {dtype: dtype_id for dtype_id, dtype in enumerate(FLOAT_DTYPES)}

	magic_bytes: bytes           # {MAGIC_SIZE}s = Magic bytes signifying that cache file is intact (must come first)
	version: int                 # B = Cache file version
	use_targets: bool            # ? = Whether the cache file contains targets
	full_targets: bool           # ? = Whether all targets are valid and none can be fully padded (i.e. target noun ID 0 representing unknown/unclassified/ignored)
	default_weights: bool        # ? = Whether the cache file contains only default weights
	unit_weights: bool           # ? = Whether the weights for each embedding must sum to 1
	embedder_strict: bool        # ? = Whether the embedder in use needs to match more than just dimension and dtype
	embedder_hash: bytes         # 32s = SHA-256 of the embedder excluding any target configuration
	target_config_hash: bytes    # 32s = SHA-256 of the target configuration excluding fields in TARGET_EXCLUDE
	target_nouns_num: int        # L = Number of available target nouns (R)
	target_nouns_size: int       # L = Number of bytes taken up by the target noun strings
	target_dim: int              # H = Target token IDs dimension (C)
	target_dtype_id: int         # H = Target token IDs dtype ID (see INT_DTYPES)
	target_mask_dtype_id: int    # H = Target token padding mask dtype ID (see BOOL_DTYPES)
	embed_num: int               # L = Number of embeddings (N)
	embed_targets_dim: int       # H = Number of targets per embedding (M)
	embed_targets_dtype_id: int  # H = Embedding target noun IDs dtype ID (see INT_DTYPES)
	embed_dim: int               # H = Embedding dimension (F)
	embed_dtype_id: int          # H = Embedding dtype ID (see FLOAT_DTYPES)

# Embedding cache file metadata class
@dataclasses.dataclass(frozen=True)
class Meta:

	target_dtype: torch.dtype
	target_mask_dtype: torch.dtype
	embed_targets_dtype: torch.dtype
	embed_dtype: torch.dtype
	embed_eps: float

	target_dtype_bytes: int
	target_mask_dtype_bytes: int
	embed_targets_dtype_bytes: int
	embed_dtype_bytes: int

	target_stride: int
	target_mask_stride: int
	embed_targets_stride: int
	embed_target_weights_stride: int
	embed_stride: int

	header_size: int
	target_nouns_size: int
	target_size: int
	target_mask_size: int
	embed_targets_size: int
	embed_target_weights_size: int
	embed_size: int

	header_offset: int
	target_nouns_offset: int
	target_offset: int
	target_mask_offset: int
	embed_targets_offset: int
	embed_target_weights_offset: int
	embed_offset: int

	total_size: int

	@classmethod
	def from_header(cls, header: Header) -> Meta:

		return Meta(

			target_dtype=(target_dtype := Header.INT_DTYPES[header.target_dtype_id]),
			target_mask_dtype=(target_mask_dtype := Header.BOOL_DTYPES[header.target_mask_dtype_id]),
			embed_targets_dtype=(embed_targets_dtype := Header.INT_DTYPES[header.embed_targets_dtype_id]),
			embed_dtype=(embed_dtype := Header.FLOAT_DTYPES[header.embed_dtype_id]),
			embed_eps=torch.finfo(embed_dtype).eps,

			target_dtype_bytes=(target_dtype_bytes := torch.tensor((), dtype=target_dtype).element_size()),
			target_mask_dtype_bytes=(target_mask_dtype_bytes := torch.tensor((), dtype=target_mask_dtype).element_size()),
			embed_targets_dtype_bytes=(embed_targets_dtype_bytes := torch.tensor((), dtype=embed_targets_dtype).element_size()),
			embed_dtype_bytes=(embed_dtype_bytes := torch.tensor((), dtype=embed_dtype).element_size()),

			target_stride=(target_stride := header.target_dim * target_dtype_bytes),
			target_mask_stride=(target_mask_stride := header.target_dim * target_mask_dtype_bytes),
			embed_targets_stride=(embed_targets_stride := header.embed_targets_dim * embed_targets_dtype_bytes),
			embed_target_weights_stride=(embed_target_weights_stride := header.embed_targets_dim * embed_dtype_bytes),
			embed_stride=(embed_stride := header.embed_dim * embed_dtype_bytes),

			header_size=(header_size := Header.STRUCT_FACTORY.size),
			target_nouns_size=(target_nouns_size := header.target_nouns_size),
			target_size=(target_size := header.target_nouns_num * target_stride),
			target_mask_size=(target_mask_size := header.target_nouns_num * target_mask_stride),
			embed_targets_size=(embed_targets_size := header.embed_num * embed_targets_stride),
			embed_target_weights_size=(embed_target_weights_size := header.embed_num * embed_target_weights_stride),
			embed_size=(embed_size := header.embed_num * embed_stride),

			header_offset=(header_offset := 0),
			target_nouns_offset=(target_nouns_offset := header_offset + header_size),
			target_offset=(target_offset := target_nouns_offset + target_nouns_size),
			target_mask_offset=(target_mask_offset := target_offset + target_size),
			embed_targets_offset=(embed_targets_offset := target_mask_offset + target_mask_size),
			embed_target_weights_offset=(embed_target_weights_offset := embed_targets_offset + embed_targets_size),
			embed_offset=(embed_offset := embed_target_weights_offset + embed_target_weights_size),

			total_size=embed_offset + embed_size,

		)

#
# Writing
#

# Embedding cache writer class
class EmbeddingCacheWriter:

	def __init__(
		self,
		cache_path: str,                               # Path of the embedding cache file to write
		embedder: embedders.Embedder,                  # Embedder that must be perfectly matched to the data to be written to the cache file (always needed as it specifies dtypes and embedding dimension and such, target_config is required if use_targets=True otherwise it is ignored)
		num_embed: int,                                # Number of embeddings that will be provided and written to the cache (N, must be >= 1)
		shuffle: bool = True,                          # Whether incoming embeddings should be shuffled and written in a random order to the cache file
		use_targets: bool = True,                      # Whether to use targets in the cache file at all (if False, implies that no embed_targets or embed_target_weights may be passed to the write method)
		full_targets: bool = True,                     # If using targets, whether the number of targets per embedding must always be the full amount specified by num_embed_targets without ANY target IDs of 0
		target_nouns: Optional[Sequence[str]] = None,  # If using targets, sequence of the targets nouns to be used in the cache (length R-1, must be provided if use_targets is True, an empty string is internally prepended to represent the fully-padded target ID 0)
		num_embed_targets: int = 1,                    # If using targets, number of targets per embedding (M, ignored if use_targets is False, must be >= 1)
		default_weights: bool = False,                 # If using targets, whether to always use uniform weights for the embedding target noun weights (only non-zero target IDs receive non-zero weights, implies that no embed_target_weights may be passed to the write method)
		unit_weights: bool = True,                     # If using targets, whether the weights per embedding must always sum to 1 (always True if using default targets)
		embedder_strict: bool = True,                  # Whether any use of the embedding cache file to write must be with a perfectly matched embedder (e.g. random unit vectors without targets does not require this)
	):

		self.use_targets = use_targets
		self.cache_path = os.path.abspath(cache_path)
		self.embedder = embedder
		self.num_embed = num_embed
		self.shuffle = shuffle
		self.num_embed_targets = num_embed_targets if self.use_targets else 0
		self.full_targets = full_targets or not self.use_targets or self.num_embed_targets <= 1
		self.default_weights = default_weights or not self.use_targets
		self.unit_weights = unit_weights or self.default_weights
		self.embedder_strict = embedder_strict

		if self.use_targets and not self.embedder_strict:
			log.warning("Cache writer is using targets but is not strict => Verify this is not a mistake")

		if not self.use_targets:
			self.target_nouns = ()
		elif target_nouns is None:
			raise ValueError("Target nouns must be provided if use_targets=True")
		else:
			self.target_nouns = ('',) + (target_nouns if isinstance(target_nouns, tuple) else tuple(target_nouns))

		self.num_target_nouns = len(self.target_nouns)  # R
		self.target_noun_map = {target_noun: i for i, target_noun in enumerate(self.target_nouns)}  # Note: This works correctly even if there is an empty string in the target nouns
		if len(self.target_noun_map) != (self.num_target_nouns - 1 if '' in itertools.islice(self.target_nouns, 1, None) else self.num_target_nouns):
			raise ValueError("There are duplicate non-empty target nouns")
		self.target_nouns_bytes = '\x00'.join(self.target_nouns).encode('utf-8')
		self.embed_targets_dtype = torch.int32  # Note: This must be at least 32-bit for PyTorch index-by-tensor to work and doesn't need to be more than 32-bit because target_nouns_num only has 4 bytes of storage in the header anyway

		self.header = Header(
			magic_bytes=Header.INIT_MAGIC_BYTES,
			version=Header.VERSION,
			use_targets=self.use_targets,
			full_targets=self.full_targets,
			default_weights=self.default_weights,
			unit_weights=self.unit_weights,
			embedder_strict=self.embedder_strict,
			embedder_hash=self.embedder.get_configuration_hash(main_config=True, target_config=False, hexdigest=False, algorithm='sha256') if self.embedder_strict else b'\x00' * 32,
			target_config_hash=self.embedder.get_configuration_hash(main_config=False, target_config=True, target_exclude=Header.TARGET_EXCLUDE, hexdigest=False, algorithm='sha256') if self.use_targets and self.embedder_strict else b'\x00' * 32,  # Note: A difference of fixed_token_length (see TARGET_EXCLUDE) is not a compatibility issue of whether a cache can be used
			target_nouns_num=self.num_target_nouns,
			target_nouns_size=len(self.target_nouns_bytes),
			target_dim=self.embedder.target_config.token_length if self.use_targets else 0,
			target_dtype_id=Header.INT_DTYPE_ID[self.embedder.token_dtype],
			target_mask_dtype_id=Header.BOOL_DTYPE_ID[self.embedder.target_config.mask_dtype if self.use_targets else torch.bool],
			embed_num=self.num_embed,
			embed_targets_dim=self.num_embed_targets,
			embed_targets_dtype_id=Header.INT_DTYPE_ID[self.embed_targets_dtype],
			embed_dim=self.embedder.embed_dim,
			embed_dtype_id=Header.FLOAT_DTYPE_ID[self.embedder.embed_dtype],
		)

		if self.header.embed_num < 1:
			raise ValueError(f"Cache file must have a positive number of embeddings: {self.header.embed_num}")
		if self.use_targets:
			if self.header.target_dim < 1:
				raise ValueError(f"Cache file must have a positive target token IDs dimension: {self.header.target_dim}")
			if self.header.embed_targets_dim < 1:
				raise ValueError(f"Cache file must have a positive number of targets per embedding: {self.header.embed_targets_dim}")

		self.header_bytes = Header.STRUCT_FACTORY.pack(*dataclasses.asdict(self.header).values())
		self.meta = Meta.from_header(header=self.header)

		self.embed_written = 0
		self.bytes_written = 0
		self.shuffle_perm = None

		self.target_token_ids = None
		self.target_mask = None
		self.cache_fd = None
		self.default_weights_tensor = None

	def tensorize_embed_targets(self, embed_targets_str: Sequence[Union[str, Sequence[str]]]) -> torch.Tensor:
		# embed_targets_str: Sequence of target nouns (or sequence of sequence of target nouns in multi-target case) to convert to a tensor of target noun IDs (the index of the target noun in the sequence of target nouns configured for this cache)
		# Returns the equivalent zero-padded BxM target noun IDs tensor, where B is the length of embed_targets_str, and no element of embed_targets_str can be a sequence longer than M (otherwise there will be an indexing error)
		# If there is a target noun (other than ID 0) that is the empty string, then this will be the output target noun ID for an input empty string, otherwise it will be ID 0

		if not self.use_targets:
			raise ValueError("Cannot tensorize embedding target noun IDs if not using targets")

		embed_targets = torch.zeros(size=(len(embed_targets_str), self.header.embed_targets_dim), dtype=self.meta.embed_targets_dtype)
		for i, targets in enumerate(embed_targets_str):
			if isinstance(targets, str):
				embed_targets[i, 0] = self.target_noun_map[targets]
			else:
				for j, target in enumerate(targets):
					embed_targets[i, j] = self.target_noun_map[target]

		return embed_targets

	def __enter__(self) -> EmbeddingCacheWriter:

		log.info(f"Writing {self.meta.total_size / (1 << 30):.3f}GiB embedding cache: {self.cache_path}")

		self.embed_written = 0
		self.bytes_written = 0
		self.shuffle_perm = torch.randperm(n=self.header.embed_num, dtype=torch.int32) if self.shuffle else None

		self.target_token_ids = None
		self.target_mask = None
		self.cache_fd = None
		self.default_weights_tensor = None

		def abort_cache():
			self.target_token_ids = None
			self.target_mask = None
			self.cache_fd = None
			self.default_weights_tensor = None

		with contextlib.ExitStack() as stack:

			stack.callback(abort_cache)

			if self.use_targets:
				target_token_ids, target_mask = self.embedder.tokenize_target(text=self.target_nouns)
				if target_mask is None:
					target_mask = torch.zeros_like(target_token_ids, dtype=self.meta.target_mask_dtype)
				target_token_ids[0, :].fill_(value=self.embedder.target_config.pad_token_id)
				target_mask[0, :].fill_(value=True)
				assert target_token_ids.stride() == target_mask.stride() == (self.header.target_dim, 1)  # noqa
				if not target_token_ids.shape[1] == target_mask.shape[1] == self.header.target_dim:
					raise ValueError(f"Mismatch in target tokenization lengths: {target_token_ids.shape[1]} vs {target_mask.shape[1]} vs {self.header.target_dim}")
				if target_token_ids.dtype != self.meta.target_dtype or target_token_ids.shape != (self.header.target_nouns_num, self.header.target_dim):
					raise ValueError(f"Unexpected target token IDs tensor: Shape {tuple(target_token_ids.shape)}, DType {target_token_ids.dtype}")
				if target_mask.dtype != self.meta.target_mask_dtype or target_mask.shape != (self.header.target_nouns_num, self.header.target_dim):
					raise ValueError(f"Unexpected target token padding mask tensor: Shape {tuple(target_mask.shape)}, DType {target_mask.dtype}")
				self.target_token_ids = target_token_ids
				self.target_mask = target_mask

			cache_fd = os.open(self.cache_path, os.O_RDWR | os.O_CREAT)
			stack.callback(os.close, cache_fd)

			self.cache_fd = cache_fd

			os.ftruncate(self.cache_fd, 0)
			os.ftruncate(self.cache_fd, self.meta.total_size)
			stack.callback(self.remove)

			self._write(buffer=self.header_bytes, offset=self.meta.header_offset, expected_size=self.meta.header_size)
			if self.use_targets:
				self._write(buffer=self.target_nouns_bytes, offset=self.meta.target_nouns_offset, expected_size=self.meta.target_nouns_size)
				self._write(buffer=memoryview(self.target_token_ids.numpy()), offset=self.meta.target_offset, expected_size=self.meta.target_size)
				self._write(buffer=memoryview(self.target_mask.numpy()), offset=self.meta.target_mask_offset, expected_size=self.meta.target_mask_size)
				if self.default_weights:
					if self.full_targets:
						self._write(buffer=memoryview(torch.full(size=(self.header.embed_num, self.header.embed_targets_dim), fill_value=1 / self.header.embed_targets_dim, dtype=self.meta.embed_dtype).numpy()), offset=self.meta.embed_target_weights_offset, expected_size=self.meta.embed_target_weights_size)
					else:
						self.default_weights_tensor = torch.tril(torch.ones(self.header.embed_targets_dim, self.header.embed_targets_dim)) / torch.arange(start=1, end=self.header.embed_targets_dim + 1).unsqueeze(dim=1)

			stack.pop_all()

		return self

	def write(self, embeds: torch.Tensor, embed_targets: Optional[torch.Tensor] = None, embed_target_weights: Optional[torch.Tensor] = None):
		# embeds = BxF batch of embeddings (B >= 1, Careful: Must be CPU tensor like all inputs to this method)
		# embed_targets = BxM batch of embedding target noun IDs (must be provided iff use_targets)
		# embed_target_weights = BxM batch of embedding target noun weights (must be provided iff use_targets and not default_weights)
		# The target noun IDs must have all non-zero IDs before any zero IDs
		# The weights (if given) must be non-negative and must for each embedding be in descending order (obviously the order of embed_targets must then also match the order of the weights)
		# The weights corresponding to all zero target noun IDs must also be zero
		# The first target for each embedding cannot have neither a zero target noun ID nor a zero weight
		# Either the data must be provided in statistically non-systematic order or shuffle must be True (it's the responsibility of the writer to make sure of this)

		batch_size = embeds.shape[0]
		if (embed_targets is not None) != self.use_targets:
			raise ValueError("Embedding target noun IDs were provided although none were expected, or vice versa")
		if (embed_target_weights is None) != self.default_weights:
			raise ValueError("Embedding target noun weights were provided although none were expected, or vice versa")
		if embeds.ndim != 2 or batch_size < 1 or embeds.shape[1] != self.header.embed_dim or embeds.dtype != self.meta.embed_dtype:
			raise ValueError(f"Unexpected embeddings tensor: Shape {tuple(embeds.shape)}, DType {embeds.dtype}")

		embed_index = self.embed_written
		self.embed_written += batch_size
		if self.embed_written > self.header.embed_num:
			raise ValueError(f"Invalid embedding index {embed_index} to write {batch_size} samples to due to the total number of embeddings only being {self.header.embed_num}")

		if torch.any((torch.linalg.vector_norm(embeds, dim=1) - 1).abs() > 4 * self.meta.embed_eps):
			raise ValueError("Embeddings must always be unit vectors")

		buffer = memoryview(embeds.numpy())
		if self.shuffle:
			embed_indices = self.shuffle_perm[embed_index:self.embed_written].tolist()
			assert len(embed_indices) == batch_size
			for i, index in enumerate(embed_indices):
				self._write(buffer=buffer[i:i+1], offset=self.meta.embed_offset + index * self.meta.embed_stride, expected_size=self.meta.embed_stride)
		else:
			self._write(buffer=buffer, offset=self.meta.embed_offset + embed_index * self.meta.embed_stride, expected_size=batch_size * self.meta.embed_stride)
			embed_indices = None

		if embed_targets is not None:

			if embed_targets.shape != (batch_size, self.header.embed_targets_dim) or embed_targets.dtype != self.meta.embed_targets_dtype:
				raise ValueError(f"Unexpected embedding target noun IDs tensor: Shape {tuple(embed_targets.shape)}, DType {embed_targets.dtype}")
			min_id, max_id = torch.aminmax(embed_targets)
			if min_id < 0 or max_id >= self.num_target_nouns:
				raise ValueError(f"Target noun IDs tensor has values outside the expected range: IDs {min_id.item()} to {max_id.item()} seen given {self.num_target_nouns} target nouns")
			if self.full_targets:
				if min_id <= 0:
					raise ValueError("Embedding target cannot have any zeros if full targets is specified")
			elif embed_targets[:, 0].min() <= 0:
				raise ValueError("First target must always be non-zero even if not using full targets")
			embed_targets_nonzero = embed_targets.bool()
			if embed_targets.shape[1] > 1 and not torch.equal(embed_targets_nonzero.cummin(dim=1)[0], embed_targets_nonzero):
				raise ValueError("All non-zero target noun IDs must come before any trailing zeros")

			buffer = memoryview(embed_targets.numpy())
			if self.shuffle:
				for i, index in enumerate(embed_indices):
					self._write(buffer=buffer[i:i+1], offset=self.meta.embed_targets_offset + index * self.meta.embed_targets_stride, expected_size=self.meta.embed_targets_stride)
			else:
				self._write(buffer=buffer, offset=self.meta.embed_targets_offset + embed_index * self.meta.embed_targets_stride, expected_size=batch_size * self.meta.embed_targets_stride)

			if embed_target_weights is None and not self.full_targets:
				embed_target_weights = self.default_weights_tensor[embed_targets_nonzero[:, 1:].sum(dim=1)]

		if embed_target_weights is not None:

			if embed_target_weights.shape != (batch_size, self.header.embed_targets_dim) or embed_target_weights.dtype != self.meta.embed_dtype:
				raise ValueError(f"Unexpected embedding target noun weights tensor: Shape {tuple(embed_target_weights.shape)}, DType {embed_target_weights.dtype}")
			if torch.any(embed_target_weights < 0):
				raise ValueError("Embedding target noun weights must be non-negative")
			if embed_target_weights[:, 0].min() <= 0:
				raise ValueError("First target weight must always be non-zero")
			if self.header.embed_targets_dim > 1 and torch.any(embed_target_weights[:, 1:] - embed_target_weights[:, :-1] > 4 * self.meta.embed_eps):
				raise ValueError("Embedding target noun weights must be in descending order")
			embed_target_weights_nonzero = embed_target_weights.bool()
			if (embed_targets == 0).logical_and_(embed_target_weights_nonzero).any():  # noqa
				raise ValueError("Zero target noun IDs must have zero weight")
			if embed_target_weights.shape[1] > 1 and not torch.equal(embed_target_weights_nonzero.cummin(dim=1)[0], embed_target_weights_nonzero):
				raise ValueError("All non-zero target noun weights must come before any trailing zeros")
			if self.unit_weights and torch.any((embed_target_weights.sum(dim=1) - 1).abs() > 4 * self.meta.embed_eps):
				raise ValueError("As unit weights was specified, the target noun weights are expected to sum to 1 for each embedding")

			buffer = memoryview(embed_target_weights.numpy())
			if self.shuffle:
				for i, index in enumerate(embed_indices):
					self._write(buffer=buffer[i:i+1], offset=self.meta.embed_target_weights_offset + index * self.meta.embed_target_weights_stride, expected_size=self.meta.embed_target_weights_stride)
			else:
				self._write(buffer=buffer, offset=self.meta.embed_target_weights_offset + embed_index * self.meta.embed_target_weights_stride, expected_size=batch_size * self.meta.embed_target_weights_stride)

	def _write(self, buffer: Union[bytes, memoryview], offset: int, expected_size: int):
		# Write a buffer to the cache, starting at a particular offset relative to the start of the file, and check that all bytes were written successfully (also that as many as expected were written)
		buffer_bytes = buffer.nbytes if isinstance(buffer, memoryview) else len(buffer)
		bytes_written = os.pwrite(self.cache_fd, buffer, offset)
		self.bytes_written += bytes_written
		if bytes_written != buffer_bytes:
			raise OSError(f"Failed to write all bytes in the buffer: {bytes_written} vs {buffer_bytes}")
		if bytes_written != expected_size:
			raise OSError(f"Written buffer was not of the expected size: {bytes_written} vs {expected_size}")

	def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
		valid_cache = False
		try:
			if exc_type is None and self.embed_written == self.header.embed_num and self.bytes_written == self.meta.total_size:
				self._write(buffer=Header.MAGIC_BYTES, offset=self.meta.header_offset, expected_size=Header.MAGIC_SIZE)
				log.info("Flushing pending writes to output embedding cache file...")
				os.fsync(self.cache_fd)
				assert os.pread(self.cache_fd, Header.MAGIC_SIZE, self.meta.header_offset) == Header.MAGIC_BYTES and os.fstat(self.cache_fd).st_size == self.meta.total_size
				valid_cache = True
		finally:
			try:
				cache_fd = self.cache_fd
				self.embed_written = 0
				self.bytes_written = 0
				self.shuffle_perm = None
				self.target_token_ids = None
				self.target_mask = None
				self.cache_fd = None
				self.default_weights_tensor = None
				os.close(cache_fd)
			finally:
				if valid_cache:
					log.info(f"Finished writing {self.meta.total_size / (1 << 30):.3f}GiB embedding cache")
				else:
					self.remove()
					raise RuntimeError("Failed to write embedding cache")
		return False

	def remove(self):
		with contextlib.suppress(FileNotFoundError):
			os.remove(self.cache_path)
			log.warning(f"Removed cache file: {self.cache_path}")

#
# Reading
#

# Open embedding cache file class
@dataclasses.dataclass(frozen=True)
class OpenCacheFile:
	file: Optional[BinaryIO]    # Opened binary cache file object
	mmap: Optional[mmap.mmap]   # Memory map of the cache file
	view: Optional[memoryview]  # Bytes-like view of the memory-mapped data

# Embedding cache class
class EmbeddingCache:

	def __init__(self, cache_path: str, embedder: embedders.Embedder, use_targets: Optional[bool] = None, strict_embedder: bool = True):
		# cache_path = Path of the embedding cache file to open
		# embedder = Embedder that must be perfectly matched to the cache file (otherwise an exception is raised, any existing target config is ignored for now and only considered as of __enter__)
		# use_targets = Whether targets should be used if they are present in the embedding cache file (None = Use targets if they are present in the cache file otherwise not)
		# strict_embedder = Whether the embedder must perfectly match the embedder used to generate the cache file (True) or only the embedding dimension and dtype have to match (False) => Use with caution!

		self.cache_path = os.path.abspath(cache_path)
		self.embedder = embedder
		self.use_targets = use_targets          # Note: This can later change from None to True or False based on the loaded cache file
		self.strict_embedder = strict_embedder  # Note: Even if this is True, a strict embedder check is skipped if the cache file specifies it does not need a strict embedder check
		log.info(f"Using embedding cache: {self.cache_path}")

		with open(self.cache_path, 'rb') as file:

			self.cache_stat = os.fstat(file.fileno())

			self.header_bytes = file.read(Header.STRUCT_FACTORY.size)
			if len(self.header_bytes) != Header.STRUCT_FACTORY.size:
				raise ValueError(f"Cache file too short for header: {len(self.header_bytes)} bytes read but {Header.STRUCT_FACTORY.size} needed")
			self.header = Header(*Header.STRUCT_FACTORY.unpack(self.header_bytes))

			if self.header.magic_bytes != Header.MAGIC_BYTES:
				raise ValueError(f"Cache file has invalid magic bytes: {self.header.magic_bytes} vs {Header.MAGIC_BYTES}")
			if self.header.version > Header.VERSION or self.header.version < 1:
				raise ValueError(f"Cache file version is unsupported: {self.header.version} vs supported {Header.VERSION}")
			log.info(f"Loaded cache header information of version {self.header.version}")

			if self.strict_embedder and self.header.embedder_strict:
				embedder_hash = self.embedder.get_configuration_hash(main_config=True, target_config=False, hexdigest=False, algorithm='sha256')
				if embedder_hash != self.header.embedder_hash:
					raise ValueError("Cache file embedder hash does not match embedder hash => Incompatible")

			if self.use_targets is None:
				self.use_targets = self.header.use_targets
			if self.use_targets:
				if not self.header.use_targets:
					raise ValueError("Embedding cache class requires targets but the loaded cache file has none")
				if self.header.target_nouns_num < 1:
					raise ValueError("Cache file needs to have at least one target noun")
				self.target_nouns_bytes = file.read(self.header.target_nouns_size)
				if len(self.target_nouns_bytes) != self.header.target_nouns_size:
					raise ValueError(f"Cache file too short for target nouns: {len(self.target_nouns_bytes)} bytes read but {self.header.target_nouns_size} needed")
				self.target_nouns = tuple(self.target_nouns_bytes.decode('utf-8').split('\x00'))
				if len(self.target_nouns) != self.header.target_nouns_num:
					raise ValueError(f"Cache file has an inconsistent number of target nouns: {len(self.target_nouns)} vs {self.header.target_nouns_num}")
				if self.target_nouns[0] != '':
					raise ValueError("First target noun in cache file must always be the empty string (which signifies 'unknown/no classification')")
				log.info(f"Loaded {self.header.target_nouns_num} target nouns from cache")
			else:
				self.target_nouns_bytes = None
				self.target_nouns = None

			file.seek(0, os.SEEK_END)
			self.cache_size = file.tell()
			assert self.cache_size == self.cache_stat.st_size
			log.info(f"Cache size is {self.cache_size} bytes = {self.cache_size / (1 << 30):.3f}GiB")

		self.meta = Meta.from_header(header=self.header)

		if self.header.embed_num < 1:
			raise ValueError(f"Cache file must have a positive number of embeddings: {self.header.embed_num}")
		if self.header.embed_dim != self.embedder.embed_dim:
			raise ValueError(f"Cache file has embedding dimension mismatch: {self.header.embed_dim} vs {self.embedder.embed_dim}")
		if self.meta.embed_dtype != self.embedder.embed_dtype:
			raise ValueError(f"Cache file has embedding dtype mismatch: {self.meta.embed_dtype} vs {self.embedder.embed_dtype}")
		if self.cache_size != self.meta.total_size:
			raise ValueError(f"Cache file has an unexpected actual size: {self.cache_size} vs {self.meta.total_size}")

		if self.use_targets:
			if self.header.target_dim < 1:
				raise ValueError(f"Cache file must have a positive target token IDs dimension: {self.header.target_dim}")
			if self.header.embed_targets_dim < 1:
				raise ValueError(f"Cache file must have a positive number of targets per embedding: {self.header.embed_targets_dim}")
			if self.meta.target_dtype != self.embedder.token_dtype:
				raise ValueError(f"Cache file has target token IDs dtype mismatch: {self.meta.target_dtype} vs {self.embedder.token_dtype}")
			if self.header.target_nouns_num - 1 > torch.iinfo(self.meta.embed_targets_dtype).max:
				raise ValueError(f"Cache file embedding target noun IDs dtype is not big enough for the number of target nouns: {self.header.target_nouns_num}")

		self.enter_count = 0
		self.cache: Optional[OpenCacheFile] = None

		self.target_token_ids: Optional[torch.Tensor] = None
		self.target_mask: Optional[torch.Tensor] = None
		self.embed_targets: Optional[torch.Tensor] = None
		self.embed_target_weights: Optional[torch.Tensor] = None

		self.empty_slice_embed: Optional[torch.Tensor] = None
		self.empty_slice_target_ids: Optional[torch.Tensor] = None
		self.empty_slice_target: Optional[torch.Tensor] = None
		self.empty_slice_mask: Optional[torch.Tensor] = None
		self.empty_slice_weight: Optional[torch.Tensor] = None

		self.translation: Optional[embedders.TargetConfig] = None

	def __len__(self) -> int:
		return self.header.embed_num

	def __enter__(self) -> EmbeddingCache:
		# Note: If using targets, the embedder must have had the required target config set by the time this method is called

		if self.use_targets:

			if self.embedder.target_config is None:
				raise ValueError("Cannot enter embedding cache that uses targets without a target configuration")

			if self.strict_embedder and self.header.embedder_strict:
				target_config_hash = self.embedder.get_configuration_hash(main_config=False, target_config=True, target_exclude=Header.TARGET_EXCLUDE, hexdigest=False, algorithm='sha256')  # Note: A difference of fixed_token_length (see TARGET_EXCLUDE) is not a compatibility issue of the cache itself
				if target_config_hash != self.header.target_config_hash:
					target_config_hash = self.embedder.get_configuration_hash(main_config=False, target_config=True, target_exclude=Header.TARGET_EXCLUDE, target_override={'use_masks': True}, hexdigest=False, algorithm='sha256')  # Note: If a cache was generated with the only difference being that masks were used, but now we don't want them, then that's not a compatibility issue
					if target_config_hash != self.header.target_config_hash:
						raise ValueError("Cache file target config hash does not match target config hash => Incompatible")

			if self.header.target_dim != self.embedder.target_config.token_length:
				raise ValueError(f"Cache file has target token IDs dimension mismatch: {self.header.target_dim} vs {self.embedder.target_config.token_length}")
			if self.meta.target_dtype != self.embedder.target_config.token_dtype:
				raise ValueError(f"Cache file has target token ID dtype mismatch: {self.meta.target_dtype} vs {self.embedder.target_config.token_dtype}")
			if self.meta.target_mask_dtype != self.embedder.target_config.mask_dtype:
				raise ValueError(f"Cache file has target token padding mask dtype mismatch: {self.meta.target_mask_dtype} vs {self.embedder.target_config.mask_dtype}")

		if not self.cache:

			with contextlib.ExitStack() as stack:

				stack.callback(log.info, "Unloaded and un-memory-mapped cache")
				cache_file = stack.enter_context(open(self.cache_path, 'r+b'))
				cache_stat = os.fstat(cache_file.fileno())
				if cache_stat.st_ino != self.cache_stat.st_ino or cache_stat.st_dev != self.cache_stat.st_dev or cache_stat.st_size != self.cache_stat.st_size or cache_stat.st_mtime_ns != self.cache_stat.st_mtime_ns:
					raise ValueError("Cache file has externally changed since it was first opened")

				cache_mmap = stack.enter_context(mmap.mmap(cache_file.fileno(), length=0, flags=mmap.MAP_SHARED, access=mmap.ACCESS_READ))
				cache_mmap.madvise(mmap.MADV_RANDOM)
				cache_mmap.madvise(mmap.MADV_WILLNEED)

				def abort_cache():

					self.empty_slice_embed = None
					self.empty_slice_target_ids = None
					self.empty_slice_target = None
					self.empty_slice_mask = None
					self.empty_slice_weight = None

					self.target_token_ids = None
					self.target_mask = None
					self.embed_targets = None
					self.embed_target_weights = None

					self.cache = None

					nonlocal cache_view
					cache_view.release()

				cache_view = memoryview(cache_mmap).toreadonly()
				stack.callback(abort_cache)

				if cache_view[self.meta.header_offset:self.meta.target_nouns_offset] != self.header_bytes:
					raise ValueError("Cache file header bytes have changed since cache file was first opened")
				if self.use_targets and cache_view[self.meta.target_nouns_offset:self.meta.target_offset] != self.target_nouns_bytes:
					raise ValueError("Cache file target nouns have changed since cache file was first opened")
				assert cache_view.nbytes == cache_stat.st_size == self.cache_size

				log.info("Loaded memory-mapped cache ready for use")
				self.cache = OpenCacheFile(file=cache_file, mmap=cache_mmap, view=cache_view)

				if self.use_targets:
					# Note: These preloaded tensors only result in more shared memory if they are directly returned from the loader, which only applies to the embed_target_weights => By sharing this memory explicitly now we reduce the shared memory usage by a factor of the number of data loader workers (although it means more shared memory if use_weights is False, as this class is not aware whether a future Dataset created on top of it will be using weights)
					#       You can check shared memory usage with df -h by looking at /dev/shm, and running htop should also show that the worker processes are almost purely shared memory slowly incrementing with time

					self.target_token_ids = torch.frombuffer(self.cache.view, dtype=self.meta.target_dtype, count=self.header.target_nouns_num * self.header.target_dim, offset=self.meta.target_offset).view(self.header.target_nouns_num, self.header.target_dim)  # RxC
					assert self.target_token_ids.stride() == (self.header.target_dim, 1)
					if self.embedder.target_config.use_masks:
						self.target_mask = torch.frombuffer(self.cache.view, dtype=self.meta.target_mask_dtype, count=self.header.target_nouns_num * self.header.target_dim, offset=self.meta.target_mask_offset).view(self.header.target_nouns_num, self.header.target_dim)  # RxC
						assert self.target_mask.stride() == (self.header.target_dim, 1)
					else:
						self.target_mask = None
					log.info(f"Preloaded all target noun tokenizations from cache: {self.target_token_ids.shape[0]}\xD7{self.target_token_ids.shape[1]} of {self.target_token_ids.dtype}{f'/{self.target_mask.dtype}' if self.target_mask is not None else ''}")

					self.embed_targets = torch.frombuffer(self.cache.view, dtype=self.meta.embed_targets_dtype, count=self.header.embed_num * self.header.embed_targets_dim, offset=self.meta.embed_targets_offset).view(self.header.embed_num, self.header.embed_targets_dim)  # NxM
					assert self.embed_targets.stride() == (self.header.embed_targets_dim, 1)
					self.embed_target_weights = torch.frombuffer(self.cache.view, dtype=self.meta.embed_dtype, count=self.header.embed_num * self.header.embed_targets_dim, offset=self.meta.embed_target_weights_offset).view(self.header.embed_num, self.header.embed_targets_dim).share_memory_()  # NxM
					assert self.embed_target_weights.stride() == (self.header.embed_targets_dim, 1)
					log.info(f"Preloaded all target noun IDs and weights from cache: {self.embed_targets.shape[0]}\xD7{self.embed_targets.shape[1]} of {self.embed_targets.dtype}/{self.embed_target_weights.dtype}")

				else:

					self.target_token_ids = None
					self.target_mask = None
					self.embed_targets = None
					self.embed_target_weights = None

				self.empty_slice_embed = torch.empty(size=(0, self.header.embed_dim), dtype=self.meta.embed_dtype)
				self.empty_slice_target_ids = torch.empty(size=(0, self.header.embed_targets_dim), dtype=self.meta.embed_targets_dtype) if self.use_targets else None
				self.empty_slice_target = torch.empty(size=(0, self.header.embed_targets_dim, self.header.target_dim), dtype=self.meta.target_dtype) if self.use_targets else None
				self.empty_slice_mask = torch.empty(size=(0, self.header.embed_targets_dim, self.header.target_dim), dtype=self.meta.target_mask_dtype) if self.target_mask is not None else None
				self.empty_slice_weight = torch.empty(size=(0, self.header.embed_targets_dim), dtype=self.meta.embed_dtype) if self.use_targets else None

				if self.translation is not None and self.translation != self.embedder.target_config:
					log.info(f"Applying TRANSLATION to embedding cache to go from a vocab size of {self.embedder.target_config.vocab_size} (cache file) to {self.translation.vocab_size} (returned samples)")
					assert self.use_targets
					if self.embedder.target_config.compact_ids:
						assert self.target_token_ids.min() >= 0 and self.target_token_ids.max() < self.embedder.target_config.vocab_size
						self.target_token_ids = self.embedder.target_config.compact_unmap[self.target_token_ids]
					self.target_token_ids = self.target_token_ids.to(dtype=self.translation.token_dtype)
					self.empty_slice_target = self.empty_slice_target.to(dtype=self.translation.token_dtype)
					assert self.target_token_ids.min() >= 0 and self.target_token_ids.max() < self.embedder.vocab_size
					if self.translation.compact_ids:
						self.target_token_ids = self.translation.compact_map[self.target_token_ids]
					if not (self.target_token_ids.min() >= 0 and self.target_token_ids.max() < self.translation.vocab_size):
						raise ValueError("Translation was not successful => Is the translation vocab a superset of the dataset vocab?")
					if self.target_mask is not None:
						self.target_mask = self.target_mask.to(dtype=self.translation.mask_dtype)
						self.empty_slice_mask = self.empty_slice_mask.to(dtype=self.translation.mask_dtype)

				stack.pop_all()

		self.enter_count += 1
		return self

	def __getitem__(self, index: slice) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
		# Note: Only slices with non-negative indices and default/unit steps are supported for indexing, and thus iterating this class directly with iter() is not supported
		start, stop, step = index
		if not (step is None or step == 1):
			raise IndexError("Indexed slices must have a unit step")
		start = start or 0
		stop = self.header.embed_num if stop is None else stop
		return self.get_samples(start=start, stop=stop)

	def get_samples(self, start: int, stop: int, use_weights: bool = True) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
		# start, stop = Non-negative indices defining the slice of samples to retrieve (start is inclusive, stop is non-inclusive)
		# use_weights = Assuming targets are in use, whether to retrieve and return weights
		# Returns embed = BxF, target_ids = Optional[BxM], target = Optional[BxMxC], mask = Optional[BxMxC], weight = Optional[BxM] where B can be 0 for empty slices

		if not self.cache:
			raise RuntimeError("Cache must be entered before data can be accessed")
		if start < 0 or stop < 0:
			raise IndexError("Negative indices are not supported")

		stop = min(stop, self.header.embed_num)
		batch_size = stop - start
		if batch_size <= 0:
			return self.empty_slice_embed, self.empty_slice_target_ids, self.empty_slice_target, self.empty_slice_mask, (self.empty_slice_weight if use_weights else None)

		embed = torch.frombuffer(self.cache.view, dtype=self.meta.embed_dtype, count=batch_size * self.header.embed_dim, offset=self.meta.embed_offset + start * self.meta.embed_stride).view(batch_size, self.header.embed_dim)  # BxF
		if self.use_targets:
			target_ids = self.embed_targets[start:stop, :]  # BxM
			target = self.target_token_ids[target_ids, :]  # BxMxC
			mask = None if self.target_mask is None else self.target_mask[target_ids, :]  # BxMxC
			weight = self.embed_target_weights[start:stop, :] if use_weights else None  # BxM
		else:
			target_ids = target = mask = weight = None

		return embed, target_ids, target, mask, weight

	def __exit__(self, exc_type, exc_val, exc_tb) -> bool:

		self.enter_count -= 1
		if self.enter_count <= 0:
			self.enter_count = 0

			if self.cache:

				self.empty_slice_embed = None
				self.empty_slice_target_ids = None
				self.empty_slice_target = None
				self.empty_slice_mask = None
				self.empty_slice_weight = None

				self.target_token_ids = None
				self.target_mask = None
				self.embed_targets = None
				self.embed_target_weights = None

				try:
					self.cache.view.release()
				finally:
					try:
						self.cache.mmap.close()
					finally:
						try:
							self.cache.file.close()
						finally:
							self.cache = None
							log.info("Unloaded and un-memory-mapped cache")

		return False

	class Dataset(embedding_dataset.EmbeddingDataset):

		def __init__(self, embed_cache: EmbeddingCache, batch_size: int, training: bool):
			# embed_cache = Embedding cache to wrap as a dataset that directly returns batches instead of individual samples
			# batch_size = Batch size to use for the dataset
			# training = Whether the dataset should be in training data mode (i.e. shuffle batches and drop incomplete batches)

			self.embed_cache = embed_cache
			self.header = self.embed_cache.header
			self.meta = self.embed_cache.meta
			self.batch_size = batch_size
			self.training = training

			if self.batch_size < 1:
				raise ValueError(f"Batch size must be a positive integer: {self.batch_size}")
			if self.batch_size > self.header.embed_num:
				raise ValueError(f"Batch size cannot be larger than the number of embeddings in the cache: {self.batch_size} > {self.header.embed_num}")

			num_embeds = self.header.embed_num
			complete_batches, incomplete_samples = divmod(num_embeds, self.batch_size)
			incomplete_batch = (incomplete_samples > 0)
			num_items = complete_batches

			if self.training:
				if incomplete_batch:
					num_embeds -= incomplete_samples
					incomplete_batch = False
					incomplete_samples = 0
			else:
				num_items += incomplete_batch

			self.epoch_index_offset = 0  # Note: Used by data loaders to manage per-epoch index offsets for the returned batches

			super().__init__(
				embedder=self.embed_cache.embedder,
				nominal_data_config=embedding_dataset.DataConfig(
					use_weights=not (self.header.default_weights and self.header.full_targets),
					unit_weights=self.header.unit_weights,
					multi_target=(self.header.embed_targets_dim > 1),
					multi_first=False,
					full_targets=self.header.full_targets,
					fixed_multi_length=False,
					multi_length=self.header.embed_targets_dim or 1,
				),
				strict_data_config_fields=set() if self.header.full_targets else {'full_targets'},
				num_items=num_items,
				num_embeds=num_embeds,
				targets=self.embed_cache.target_nouns,
				num_invalid_targets=1 if self.embed_cache.target_nouns else 0,
				use_targets=self.embed_cache.use_targets,
			)

			self.loader_info_data = dict(
				batch_size=self.batch_size,
				batch_size_last=incomplete_samples,
				complete_batches=complete_batches,
				incomplete_batch=incomplete_batch,
				epoch_batches=self.num_items,
				epoch_samples=self.num_embeds,
				available_samples=self.num_embeds,
			)

		def set_translation(self, target_config: Optional[embedders.TargetConfig]):
			super().set_translation(target_config=target_config)
			self.embed_cache.translation = self.translation  # Note: All Datasets that point to the same EmbeddingCache must have equal translations configured (even if not concurrent)

		def loaded(self) -> ContextManager:
			return self.embed_cache

		def __getitem__(self, index) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:

			if index < 0 or index >= self.num_items:
				raise IndexError("Index out of range")

			if self.epoch_index_offset == 0 or not self.training:
				start = index * self.batch_size
				embed, target_ids, target, mask, weight = self.embed_cache.get_samples(start=start, stop=start + self.batch_size, use_weights=self.data_config.use_weights)  # Note: Due to slicing, any last incomplete batch is handled correctly despite the way stop is calculated here
			else:
				start = (index * self.batch_size + self.epoch_index_offset) % self.header.embed_num  # 0 <= start < embed_num, so there is at least one embed after slice-index start before the slice-index wrapping point
				stop = (start + self.batch_size - 1) % self.header.embed_num + 1  # 0 < stop <= embed_num (self.batch_size >= 1 so 0 wrapping to embed_num - 1 is never an incorrect thing to do, e.g. would have been a problem if start = stop = 0)
				if start < stop:
					embed, target_ids, target, mask, weight = self.embed_cache.get_samples(start=start, stop=stop, use_weights=self.data_config.use_weights)
				else:
					embed, target_ids, target, mask, weight = (None if tensors[0] is None else torch.concat(tensors=tensors, dim=0) for tensors in zip(self.embed_cache.get_samples(start=start, stop=self.header.embed_num, use_weights=self.data_config.use_weights), self.embed_cache.get_samples(start=0, stop=stop, use_weights=self.data_config.use_weights)))

			if target_ids is not None:

				if self.data_config.multi_target:

					if trimmed_multi_length := (self.data_config.multi_length < target.shape[1]):
						target = target[:, :self.data_config.multi_length, :]
						if mask is not None:
							mask = mask[:, :self.data_config.multi_length, :]
						if weight is None:
							target_ids = target_ids[:, :self.data_config.multi_length]
						else:
							weight = weight[:, :self.data_config.multi_length]

					if not self.data_config.fixed_multi_length and target.shape[1] > 1:
						col_nonzero, col_index = (target_ids if weight is None else weight).any(dim=0).min(dim=0)
						if not col_nonzero:
							target = target[:, :col_index, :]
							if mask is not None:
								mask = mask[:, :col_index, :]
							if weight is not None:
								weight = weight[:, :col_index]

					if weight is not None and self.data_config.unit_weights and (not self.header.unit_weights or trimmed_multi_length):
						if weight.shape[1] == 1:
							weight.fill_(value=1)
						else:
							torch.nn.functional.normalize(weight, p=1, dim=1, out=weight)

				else:

					target = target[:, 0, :]
					if mask is not None:
						mask = mask[:, 0, :]
					if weight is not None:
						weight_length = weight.shape[1]
						weight = weight[:, 0]
						if self.data_config.unit_weights and (not self.header.unit_weights or weight_length > 1):
							weight.fill_(value=1)

				if not self.embedder.target_config.fixed_token_length and mask is not None:
					col_masked, col_index = (mask.all(dim=0) if mask.ndim > 2 else mask).all(dim=0).max(dim=0)
					if col_masked:
						target = target[..., :col_index]
						mask = mask[..., :col_index]

				if self.data_config.multi_target and self.data_config.multi_first:
					target = target.transpose(0, 1)
					if mask is not None:
						mask = mask.transpose(0, 1)
					if weight is not None:
						weight = weight.transpose(0, 1)

			return embed, target, mask, weight

		def create_loader(self, batch_size: int, num_workers: int, training: bool, device: torch.device, patch: bool = True) -> tuple[torch.utils.data.DataLoader, embedding_dataset.LoaderInfo]:

			if batch_size != self.batch_size or training != self.training:
				raise ValueError("Batch size and training mode must match between dataset constructor and call to create_loader()")

			device_is_cpu = (device.type == 'cpu')
			loader = EmbeddingCache.DataLoader(dataset=self, batch_size=None, shuffle=self.training, num_workers=num_workers, pin_memory=not device_is_cpu, patch_device=device if patch and not device_is_cpu else None)
			on_device = device_is_cpu or patch

			loader_info = embedding_dataset.LoaderInfo(
				num_workers=loader.num_workers,
				prefetch_factor=0 if loader.prefetch_factor is None else loader.prefetch_factor,
				pin_memory=not on_device,
				on_device=on_device,
				**self.loader_info_data,
			)

			return loader, loader_info

	create_dataset = functools.partialmethod(Dataset)  # Note: Creates a dataset from an embedding cache while automatically providing self as the embed_cache argument, i.e. use as `dataset = embed_cache.create_dataset(batch_size=1024, training=True)

	class DataLoader(torch.utils.data.DataLoader):

		dataset: EmbeddingCache.Dataset

		def __init__(self, dataset: EmbeddingCache.Dataset, *args, patch_device: Optional[torch.device], **kwargs):
			super().__init__(dataset, *args, **kwargs)
			self.patch_device = patch_device

		def __iter__(self) -> Iter:
			# Note: We return an EmbeddingCache.DataLoader.Iter instead of a torch.utils.data.dataloader._BaseDataLoaderIter (which the base class is annotated to return) but this is not an issue
			return self.Iter(loader=self, patch_device=self.patch_device)

		class Iter:

			def __init__(self, loader: EmbeddingCache.DataLoader, patch_device: Optional[torch.device]):
				self.loader = loader
				self.dataset = self.loader.dataset
				self.patch_device = patch_device
				assert self.patch_device is None or self.patch_device.type != 'cpu'
				self.epoch_index_offset = random.randrange(self.dataset.num_embeds) if self.dataset.training else 0
				self.dataset.epoch_index_offset = self.epoch_index_offset  # Note: In case data loader workers will be used, we need to set the epoch index offset now so that when the dataset is copied to the worker processes it will already have the right value
				self.raw_iter = super(EmbeddingCache.DataLoader, self.loader).__iter__()  # Note: Data loader worker processes (if any) are created here (must be non-persistent)

			def __iter__(self) -> EmbeddingCache.DataLoader.Iter:
				return self

			def __next__(self) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:

				self.dataset.epoch_index_offset = self.epoch_index_offset  # Note: In case data loader workers are not in use, we need to set the epoch index offset to ensure the right value will be used
				embed, target, mask, weight = next(self.raw_iter)  # Note: The epoch index offset is used inside this call (within this process, or using the fixed epoch index offset that was configured when the workers were created)

				if self.patch_device is not None:
					embed = embed.to(device=self.patch_device, non_blocking=True)
					if target is not None:
						target = target.to(device=self.patch_device, non_blocking=True)
					if mask is not None:
						mask = mask.to(device=self.patch_device, non_blocking=True)
					if weight is not None:
						weight = weight.to(device=self.patch_device, non_blocking=True)

				return embed, target, mask, weight
# EOF
