# Image/text embedder model classes

# Imports
from __future__ import annotations
import os
import json
import hashlib
import warnings
import itertools
import contextlib
import dataclasses
from typing import Union, Optional, Sequence, Callable, Any
import tqdm
import PIL.Image
import torch
try:
	import clip  # noqa
except ModuleNotFoundError:
	clip = None
try:
	import open_clip  # noqa
except ModuleNotFoundError:
	open_clip = None
try:
	import transformers  # noqa
except ModuleNotFoundError:
	transformers = None
try:
	import optimum.bettertransformer  # noqa
except ModuleNotFoundError:
	optimum = None
from logger import log
import utils

# Ignore warnings: PyTorch compile outputs some spam warnings
warnings.filterwarnings("ignore", message="'has_cuda' is deprecated", category=UserWarning)
warnings.filterwarnings("ignore", message="'has_cudnn' is deprecated", category=UserWarning)
warnings.filterwarnings("ignore", message="'has_mps' is deprecated", category=UserWarning)
warnings.filterwarnings("ignore", message="'has_mkldnn' is deprecated", category=UserWarning)

# Target configuration class
@dataclasses.dataclass(frozen=True)
class TargetConfig:

	vocab_size: int                        # Must be number of available compacted token IDs if compact, otherwise the vocab size of the tokenizer
	token_dtype: torch.dtype               # Required PyTorch tensor data type of the target token ID tensors
	mask_dtype: torch.dtype                # Required PyTorch tensor data type of the target padding mask tensors
	start_token_id: Optional[int]          # Start token ID (None = No start tokens in target tokenizations, must be None or same as tokenizer start token ID if not compact, MUST be None or 1 if compact)
	end_token_id: Optional[int]            # End token ID (None = No end tokens in target tokenizations, must be None or same as tokenizer end token ID if not compact, may be identical to pad_token_id, MUST be None or 0 if compact)
	pad_token_id: int                      # Pad token ID (must be same as tokenizer pad token ID if not compact, MUST be 0 if compact)
	compact_ids: bool                      # Whether target tokenizations should use compact IDs (sequential renumbering of only the token IDs that are actually used, pad token = 0, end token = 0 or None, start token = 1 or None)
	compact_map: Optional[torch.Tensor]    # If compact, 1D index list mapping token IDs to compact token IDs (sparse with fill value -1, must have correct token dtype)
	compact_unmap: Optional[torch.Tensor]  # If compact, 1D index list mapping compact token IDs to token IDs (dense, must have correct token dtype, start_token_id (i.e. 1) maps to -1 if tokenizer start token ID is None)
	fixed_token_length: bool               # Whether target tokenizations should always have a fixed length (if not, each batch is only padded to the longest sequence in that batch)
	token_length: int                      # The required fixed token length if fixed_token_length, or a nominal token length that is assumed never to be dynamically exceeded (ideally the smallest such value, but could be the full context length)
	use_masks: bool                        # Whether in addition to the target tokenization a mask should be computed by tokenize_target() that is True exactly at each padding location (NOT True at the end token if present, even if it has the same numeric value as the pad token)

	def __eq__(self, other):
		if other.__class__ is self.__class__:
			return (
				(self.vocab_size, self.token_dtype, self.mask_dtype, self.start_token_id, self.end_token_id, self.pad_token_id, self.compact_ids, self.fixed_token_length, self.token_length, self.use_masks) == (other.vocab_size, other.token_dtype, other.mask_dtype, other.start_token_id, other.end_token_id, other.pad_token_id, other.compact_ids, other.fixed_token_length, other.token_length, other.use_masks) and
				(self.compact_map is other.compact_map or (self.compact_map is not None and other.compact_map is not None and self.compact_map.dtype == other.compact_map.dtype and torch.equal(self.compact_map, other.compact_map))) and
				(self.compact_unmap is other.compact_unmap or (self.compact_unmap is not None and other.compact_unmap is not None and self.compact_unmap.dtype == other.compact_unmap.dtype and torch.equal(self.compact_unmap, other.compact_unmap)))
			)
		return NotImplemented

# Embedder class
class Embedder:

	@staticmethod
	def create(
		spec: str,                                       # String specification of which embedder to create and return (format must be 'TYPE:NAME' for a supported TYPE)
		amp: bool = True,                                # Whether to enable automatic mixed precision
		amp_bf16: bool = False,                          # Whether to use torch.bfloat16 for automatic mixed precision
		tokenizer_batch_size: int = 1024,                # See __init__
		inference_batch_size: int = 256,                 # See __init__
		image_batch_size: int = 128,                     # See __init__
		load_model: bool = True,                         # See __init__
		compile_model: bool = False,                     # See __init__
		use_optimum: bool = False,                       # Whether to use the Hugging Face optimum library for the transformers backend
		device: Union[int, str, torch.device] = 'cuda',  # See __init__
		check: bool = False,                             # See __init__
	) -> Embedder:
		if ':' not in spec:
			raise ValueError(f"Embedder spec must be of the format 'TYPE:NAME': {spec}")
		embedder_type, embedder_name = spec.split(':', maxsplit=1)
		if embedder_type == 'openai':
			return OpenAIEmbedder(model_name=embedder_name, tokenizer_batch_size=tokenizer_batch_size, inference_batch_size=inference_batch_size, image_batch_size=image_batch_size, load_model=load_model, compile_model=compile_model, device=device, check=check)
		elif embedder_type == 'openclip':
			return OpenCLIPEmbedder(model_id=embedder_name, amp=amp, amp_bf16=amp_bf16, tokenizer_batch_size=tokenizer_batch_size, inference_batch_size=inference_batch_size, image_batch_size=image_batch_size, load_model=load_model, compile_model=compile_model, device=device, check=check)
		elif embedder_type == 'transformers':
			return TransformersEmbedder(model_id=embedder_name, amp=amp, amp_bf16=amp_bf16, tokenizer_batch_size=tokenizer_batch_size, inference_batch_size=inference_batch_size, image_batch_size=image_batch_size, load_model=load_model, compile_model=compile_model, use_optimum=use_optimum, device=device, check=check)
		else:
			raise ValueError(f"Unsupported embedder type: {embedder_type}")

	def __init__(
		self,
		configuration: dict[str, Any],                   # JSON-ifiable Python dict that contains all configuration parameters that affect the behaviour of the embedder
		context_length: int,                             # Context length of the tokenizer
		vocab_size: int,                                 # Token vocabulary size (i.e. the token IDs range from 0 to vocab_size - 1)
		cased_tokens: bool,                              # Whether the tokenization is case-sensitive
		start_token_id: Optional[int],                   # Token ID that corresponds to the start token (can be None if no start token)
		end_token_id: int,                               # Token ID that corresponds to the end token
		pad_token_id: int,                               # Token ID to pad tokenizations with after the end token (may be same as end token, but NEVER same as any other token)
		token_dtype: torch.dtype,                        # PyTorch tensor data type of the produced token ID tensors
		embed_dtype: torch.dtype,                        # PyTorch tensor data type of the produced embedding vector tensors
		embed_dim: int,                                  # Embedding vector dimension
		amp_mode: Union[bool, torch.dtype] = True,       # Automatic mixed precision inferencing mode (False, True, or a specific mixed precision dtype)
		manual_amp_dtype: Optional[torch.dtype] = None,  # Manual mixed precision data type (None = Manual mixed precision is not being used)
		tokenizer_batch_size: int = 1024,                # Nominal batch size to use for pure tokenization
		inference_batch_size: int = 256,                 # Nominal batch size to use for pure inference or tokenization plus inference
		image_batch_size: int = 128,                     # Nominal batch size to use for inferencing on images
		load_model: bool = True,                         # Whether to initially load the text embedding inference model (or just the tokenizer for now)
		compile_model: bool = False,                     # Whether to compile the inference model using torch.compile
		device: Union[int, str, torch.device] = 'cuda',  # PyTorch device to perform text embedding model inference on (tokenization is always on CPU)
		check: bool = False,                             # Whether to perform any possible sanity checking during computations, even if this is slow
	):

		self.context_length = context_length
		self.vocab_size = vocab_size
		self.cased_tokens = cased_tokens
		log.info(f"Text tokenizer has context length {self.context_length} and {'case-sensitive' if self.cased_tokens else 'case-insensitive'} vocab size {self.vocab_size}")

		self.start_token_id = start_token_id
		self.end_token_id = end_token_id
		self.pad_token_id = pad_token_id
		assert (isinstance(self.start_token_id, int) or self.start_token_id is None) and isinstance(self.end_token_id, int) and isinstance(self.pad_token_id, int)

		self.device = device if isinstance(device, torch.device) else torch.device(device)
		log.info(f"Embedder is using {self.device.type.upper()} device")
		self.amp_mode = amp_mode
		if self.amp_mode and self.device.type != 'cpu':
			self.amp_context = torch.autocast(self.device.type, dtype=self.amp_mode if isinstance(self.amp_mode, torch.dtype) else None)
			self.amp_dtype = self.amp_context.fast_dtype
			assert isinstance(self.amp_dtype, torch.dtype)
			log.info(f"Embedder has AMP enabled with dtype {self.amp_dtype}")
		else:
			self.amp_context = None
			self.amp_dtype = None
			log.info("Embedder has AMP disabled")
		self.amp_context_entered = False
		self.manual_amp_dtype = manual_amp_dtype
		log.info(f"Embedder has manual mixed precision {f'enabled with dtype {self.manual_amp_dtype}' if self.manual_amp_dtype is not None else 'disabled'}")

		self.token_dtype = token_dtype
		self.embed_dtype = embed_dtype
		self.embed_dim = embed_dim
		self.tokenizer_batch_size = tokenizer_batch_size
		self.inference_batch_size = inference_batch_size
		self.image_batch_size = image_batch_size
		log.info(f"Text tokenizer has dtype {self.token_dtype}, start {self.start_token_id} end {self.end_token_id} pad {self.pad_token_id}, and nominal batch size {self.tokenizer_batch_size}")
		log.info(f"Text embedding vector has dim {self.embed_dim}, dtype {self.embed_dtype}, and nominal batch size {self.inference_batch_size}")
		log.info(f"Image component of embedder has nominal batch size {self.image_batch_size}")

		self.configuration = configuration
		self.configuration['class'] = self.__class__.__qualname__
		self.configuration['device_type'] = self.device.type
		self.configuration['amp_dtype'] = format(self.amp_dtype)

		self.target_config: Optional[TargetConfig] = None
		self.target_vocab: Optional[tuple[str, ...]] = None
		self.target_configuration: Optional[dict[str, Any]] = None

		self.check = check
		self.compile_model = compile_model
		if load_model:
			self.load_model()

	def create_target_config(
		self,
		targets: Sequence[str],         # Sequence of exactly all target nouns that will be used with the created target configuration
		*,                              # All arguments beyond here must be provided as kwargs
		with_start_token: bool,         # Whether target tokenizations should include a start token
		with_end_token: bool,           # Whether target tokenizations should include an end token
		compact_ids: bool,              # Whether target tokenizations should use compact IDs (sequential renumbering of only the token IDs that are actually used, pad token = 0, end token = 0 if with end token, start token = 1 if with start token)
		fixed_token_length: bool,       # Whether all batches should use the same fixed token length
		auto_fixed_token_length: bool,  # Whether the fixed token length (if fixed_token_length) should be auto-calculated from targets (True) or taken from the tokenizer context length (False)
		use_masks: bool,                # Whether the padding locations should be masked and thereby not contribute to the training loss
	) -> TargetConfig:

		token_id_set = set()
		max_target_tokens = 0
		max_target_tokens_str = ''
		with tqdm.tqdm(desc='Tokenizing target nouns', total=len(targets), unit='noun', unit_scale=False, dynamic_ncols=True, smoothing=0.08) as progress_bar:
			it = iter(targets)
			while target_nouns := tuple(itertools.islice(it, self.tokenizer_batch_size)):
				max_tokens, max_tokens_str, token_set = self.get_tokenize_details(text=target_nouns, token_id_set=compact_ids)
				if token_set:
					token_id_set.update(token_set)
				if max_tokens > max_target_tokens or not max_target_tokens_str:
					max_target_tokens = max_tokens
					max_target_tokens_str = max_tokens_str
				progress_bar.update(len(target_nouns))
		if compact_ids:
			token_id_set.remove(self.end_token_id)
		if not with_end_token:
			max_target_tokens -= 1
		token_id_set.discard(self.pad_token_id)
		if self.start_token_id is None:
			if with_start_token:
				max_target_tokens += 1
		else:
			if compact_ids:
				token_id_set.remove(self.start_token_id)
			if not with_start_token:
				max_target_tokens -= 1
		log.info(f"Max target tokens {'with' if with_start_token else 'without'} start token {'with' if with_end_token else 'without'} end token is {max_target_tokens} for '{max_target_tokens_str}'")

		if compact_ids:
			pad_token_id = 0
			end_token_id = 0 if with_end_token else None
			compact_list = [self.pad_token_id]
			if with_start_token:
				start_token_id = 1
				compact_list.append(self.start_token_id if self.start_token_id is not None else -1)
			else:
				start_token_id = None
			num_special = len(compact_list)
			compact_list.extend(sorted(token_id_set))
			vocab_size = len(compact_list)
			compact_unmap = torch.tensor(compact_list, dtype=self.token_dtype)
			compact_map = torch.full((self.vocab_size,), fill_value=-1, dtype=self.token_dtype)
			compact_map[compact_unmap[num_special:]] = torch.arange(start=num_special, end=vocab_size, dtype=self.token_dtype)
			compact_map[self.pad_token_id] = 0
			compact_map[self.end_token_id] = 0
			if self.start_token_id is not None and with_start_token:
				compact_map[self.start_token_id] = 1
			log.info(f"Compacting target tokenizations down to a vocab size of {vocab_size} tokens")
		else:
			vocab_size = self.vocab_size
			start_token_id = self.start_token_id if with_start_token else None
			end_token_id = self.end_token_id if with_end_token else None
			pad_token_id = self.pad_token_id
			compact_map = None
			compact_unmap = None
			log.info(f"Not compacting target tokenizations and using full vocab size of {vocab_size} tokens")

		token_length = max_target_tokens if not fixed_token_length or auto_fixed_token_length else self.context_length
		log.info(f"Using target tokenizations of {'fixed' if fixed_token_length else 'variable'} length {token_length} {'with' if use_masks else 'without'} padding masks")

		return TargetConfig(
			vocab_size=vocab_size,
			token_dtype=self.token_dtype,
			mask_dtype=torch.bool,
			start_token_id=start_token_id,
			end_token_id=end_token_id,
			pad_token_id=pad_token_id,
			compact_ids=compact_ids,
			compact_map=compact_map,
			compact_unmap=compact_unmap,
			fixed_token_length=fixed_token_length,
			token_length=token_length,
			use_masks=use_masks,
		)

	def configure_target(self, target_config: TargetConfig, target_vocab: Sequence[str]):
		# Note: The embedder target vocabulary (target_vocab) should strictly only include valid target nouns (e.g. not a leading empty string used to signify invalid targets or such)
		self.target_config = target_config
		self.target_vocab = target_vocab if isinstance(target_vocab, tuple) else tuple(target_vocab)
		self.target_configuration = {key: (value.tolist() if isinstance(value, torch.Tensor) else str(value) if isinstance(value, torch.dtype) else value) for key, value in dataclasses.asdict(target_config).items()}  # noqa

	def get_configuration(self, main_config: bool, target_config: bool, target_exclude: Optional[set[str]] = None, target_override: Optional[dict[str, Any]] = None) -> dict[str, Any]:
		configuration = self.configuration.copy() if main_config else {}
		if target_config:
			if self.target_config is None or self.target_configuration is None:
				raise ValueError("Cannot get configuration including target config because there is none yet")
			if target_exclude is None:
				configuration['target_config'] = self.target_configuration.copy()
			else:
				configuration['target_config'] = {key: value for key, value in self.target_configuration.items() if key not in target_exclude}
			if target_override is not None:
				configuration['target_config'].update(target_override)
		return configuration

	def get_configuration_hash(self, main_config: bool, target_config: bool, target_exclude: Optional[set[str]] = None, target_override: Optional[dict[str, Any]] = None, hexdigest: bool = False, algorithm: str = 'sha256') -> Union[bytes, str]:
		configuration = self.get_configuration(main_config=main_config, target_config=target_config, target_exclude=target_exclude, target_override=target_override)
		configuration_hash = hashlib.new(name=algorithm, data=json.dumps(configuration, separators=(',', ':'), sort_keys=True).encode())
		return configuration_hash.hexdigest() if hexdigest else configuration_hash.digest()

	@contextlib.contextmanager
	def inference_model(self, release=True):
		# Context manager that temporarily loads the model if it is not currently loaded
		if self.is_model_loaded():
			yield
		else:
			try:
				self.load_model()
				yield
			finally:
				self.unload_model()
				if release:
					utils.release_cuda_memory(device=self.device)

	@contextlib.contextmanager
	def inference_mode(self):
		# Context manager that temporarily sets up PyTorch inference mode and AMP for running model inferences
		with torch.inference_mode():
			if self.amp_context is None or self.amp_context_entered:
				yield
			else:
				self.amp_context_entered = True
				try:
					with self.amp_context:
						yield
				finally:
					self.amp_context_entered = False

	def load_model(self) -> bool:
		# Load the inference model and return whether the model wasn't already loaded (loaded model should be in eval() mode, model should be compiled if self.compile_model)
		raise NotImplementedError

	def unload_model(self) -> bool:
		# Unload the inference model and return whether the model wasn't already unloaded (note that the resources allocated specifically for the model are only guaranteed to be released upon a subsequent utils.release_cuda_memory)
		raise NotImplementedError

	def is_model_loaded(self) -> bool:
		# Return whether the inference model is currently loaded
		raise NotImplementedError

	def tokenize(self, text: Union[str, Sequence[str]], max_tokens: Optional[int] = None, output_dict: bool = False) -> Union[torch.Tensor, dict[str, Any]]:
		# Tokenize a text (or non-empty batch of texts) to a CPU torch tensor of integer token IDs (max_tokens optionally specifies the maximum number of token IDs to output instead of the inference model's default maximum number)
		# The produced token IDs tensor should be padded so that the largest text in the batch exactly fits, i.e. strictly not padded any more than necessary
		# If output_dict is True, then all data required for inference are returned in a dict, as opposed to just the token IDs tensor (must include 'input_ids' (self.token_dtype) and 'attention_mask' (self.token_dtype with values 0 and 1, 0 = padding token NOT end token))
		raise NotImplementedError

	def detokenize(self, token_ids: torch.Tensor) -> Union[str, list[str]]:
		# Detokenize a (batch of) token ID sequences (returns string for shape S tensor, list of strings for shape BxS tensor)
		# This method should be robust to possibly missing start tokens and allow freely interchangeable end/pad tokens, HOWEVER it is the responsibility of the caller to ensure that there is only at most one start token (right at the beginning) and that no content tokens occur after the first end/pad token
		raise NotImplementedError

	def tokenize_target(self, text: Union[str, Sequence[str]], max_tokens: Optional[int] = None) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
		# Tokenize a text just like tokenize() but then apply the target configuration (must exist), returning the tokenized target text as well as the padding mask
		# Tokenizations produced by this method are no longer suitable for text inference, as they are no longer in the tokenized format the text inference model expects

		if not self.target_config:
			raise ValueError("Must provide target configuration before tokenizing a target noun")

		tokens_dict = self.tokenize(text=text, max_tokens=max_tokens, output_dict=True)
		token_ids = tokens_dict['input_ids']
		skip_start = 1 if self.start_token_id is not None and self.target_config.start_token_id is None else 0
		skip_end = token_ids.shape[1] - 1 if self.target_config.end_token_id is None else token_ids.shape[1]
		token_ids = token_ids[:, skip_start:skip_end]
		padding_mask = torch.logical_not(tokens_dict['attention_mask'][:, skip_start:skip_end]) if self.target_config.use_masks else None

		if self.target_config.compact_ids:
			if self.target_config.end_token_id is None and padding_mask is not None:
				padding_mask[torch.eq(token_ids, self.end_token_id)] = True
			token_ids = self.target_config.compact_map[token_ids]  # Note: This maps end token ID to pad token ID if end_token_id is None
			if self.start_token_id is None and self.target_config.start_token_id is not None:
				assert self.target_config.start_token_id == 1
				token_ids = torch.concat(tensors=(token_ids.new_ones(size=(token_ids.shape[0], 1)), token_ids), dim=1)
				if padding_mask is not None:
					padding_mask = torch.concat(tensors=(padding_mask.new_zeros(size=(padding_mask.shape[0], 1)), padding_mask), dim=1)
		elif self.target_config.end_token_id is None:
			end_token_mask = torch.eq(token_ids, self.end_token_id)
			token_ids[end_token_mask] = self.target_config.pad_token_id
			if padding_mask is not None:
				padding_mask[end_token_mask] = True

		if self.target_config.fixed_token_length:
			seq_len = token_ids.shape[1]
			if seq_len > self.target_config.token_length:
				raise ValueError(f"Sequence length {seq_len} is larger than the configured target tokenization fixed length {self.target_config.token_length}")
			elif seq_len < self.target_config.token_length:
				padded_token_ids = token_ids.new_full((token_ids.shape[0], self.target_config.token_length), fill_value=self.target_config.pad_token_id)
				padded_token_ids[:, :seq_len] = token_ids
				token_ids = padded_token_ids
				if padding_mask is not None:
					padded_padding_mask = padding_mask.new_ones((token_ids.shape[0], self.target_config.token_length))
					padded_padding_mask[:, :seq_len] = padding_mask
					padding_mask = padded_padding_mask

		if self.check:
			assert self.target_config.compact_map.shape == (self.vocab_size,) and (self.target_config.compact_unmap.shape == (self.target_config.vocab_size,)) if self.target_config.compact_ids else (self.target_config.vocab_size == self.vocab_size)
			assert token_ids.min() >= 0 and token_ids.max() < self.target_config.vocab_size
			if isinstance(text, str):
				detokenized_text = self.detokenize_target(token_ids.squeeze(dim=0))
				if detokenized_text != text:
					raise ValueError(f"Detokenized text is not equivalent to the original text: '{detokenized_text}' should be '{text}'")
			else:
				detokenized_text = self.detokenize_target(token_ids)
				if len(detokenized_text) != len(text) or any(det_txt != txt for det_txt, txt in zip(detokenized_text, text)):
					raise ValueError("Detokenized text batch is not equivalent to the original text")

		return token_ids, padding_mask

	def detokenize_target(self, token_ids: torch.Tensor) -> Union[str, list[str], list[list[str]]]:
		# Detokenize a (2D/3D batch of) CPU target token ID sequences based on the target configuration (target config must exist, returns string for shape S tensor, list of strings for shape BxS tensor, list of list of strings for shape BxKxS tensor)
		# This method should be robust to missing start tokens and allow freely interchangeable end/pad tokens, HOWEVER it is the responsibility of the caller to ensure that there is only at most one start token (right at the beginning) and that no content tokens occur after the first end/pad token
		# Note: We do not explicitly add any missing start or end tokens to the token IDs tensor, as detokenize() is robust to missing start/end tokens

		if not self.target_config:
			raise ValueError("Must provide target configuration before detokenizing a target noun")

		if self.target_config.compact_ids:
			if self.start_token_id is None and self.target_config.start_token_id is not None:
				token_ids = token_ids[..., 1:]
			token_ids = self.target_config.compact_unmap[token_ids]

		if self.check:
			assert token_ids.min() >= 0 and token_ids.max() < self.vocab_size

		if token_ids.ndim == 3:
			return [self.detokenize(tids) for tids in token_ids]
		else:
			return self.detokenize(token_ids)

	def get_tokenize_details(self, text: Union[str, Sequence[str]], max_tokens: Optional[int] = None, token_id_set: bool = False) -> tuple[int, str, Optional[set[int]]]:
		# Get some details about the tokenization of a text or non-empty sequence of texts (Note: This corresponds to the normal tokenization, NOT the configured target tokenization)
		# Returns the maximum tokenization length (including any start and end tokens), a text that had that tokenization length, and optionally (if token_id_set) a set of used token IDs (it is okay if pad token ID is always in the set even if it is not in this specific tokenization)
		tokens_dict = self.tokenize(text=text, max_tokens=max_tokens, output_dict=True)
		attention_mask = tokens_dict['attention_mask']
		max_tokens = attention_mask.shape[1]
		max_tokens_str = text if isinstance(text, str) else text[attention_mask[:, -1].argmax()]
		token_set = set(tokens_dict['input_ids'].view(-1).tolist()) if token_id_set else None
		return max_tokens, max_tokens_str, token_set

	def inference_tokens(self, tokens_dict: dict[str, Any]) -> torch.Tensor:
		# Inference the text embedding model on tokenized text in dict form (tensors should be on CPU), and return the correponding embedding vector as a device tensor
		# Note: This should always be called inside a self.inference_mode() context manager for performance gains and correct handling of AMP
		raise NotImplementedError

	def inference_text(self, text: Union[str, Sequence[str]], max_tokens: Optional[int] = None) -> torch.Tensor:
		# Return the embedding vector (as a device tensor) corresponding to the provided text (or batch of texts)
		# Note: This should always be called inside a self.inference_mode() context manager for performance gains and correct handling of AMP
		return self.inference_tokens(self.tokenize(text, max_tokens=max_tokens, output_dict=True))

	def get_image_transform(self) -> Callable[[PIL.Image.Image], torch.Tensor]:
		# Retrieve the image transform in the form of a callable that accepts a single PIL image and returns a corresponding preprocessed CPU image tensor
		raise NotImplementedError

	def inference_image(self, images: torch.Tensor) -> torch.Tensor:
		# Inference the image embedding model on a CPU image tensor as returned by the image transform, and return the correponding embedding vector as a device tensor
		# Note: This should always be called inside a self.inference_mode() context manager for performance gains and correct handling of AMP
		raise NotImplementedError

# OpenAI embedder class
class OpenAIEmbedder(Embedder):

	CACHE_HOME = os.path.expanduser(os.getenv('OPENAI_HOME', os.path.join(os.getenv('XDG_CACHE_HOME', '~/.cache'), 'clip')))
	CONTEXT_LENGTH = 77
	EMBED_DIM = {
		'RN50': 1024,
		'RN101': 512,
		'RN50x4': 640,
		'RN50x16': 768,
		'RN50x64': 1024,
		'ViT-B/32': 512,
		'ViT-B/16': 512,
		'ViT-L/14': 768,
		'ViT-L/14@336px': 768,
	}

	# noinspection PyProtectedMember
	def __init__(
		self,
		model_name: str,
		tokenizer_batch_size: int = 1024,
		inference_batch_size: int = 256,
		image_batch_size: int = 128,
		load_model: bool = True,
		compile_model: bool = False,
		device: Union[int, str, torch.device] = 'cuda',
		check: bool = False,
	):

		self.model_name = model_name

		self.tokenizer = clip.clip._tokenizer  # noqa
		log.info(f"Loaded OpenAI tokenizer for '{self.model_name}'")

		self.model = None
		self.preprocess_image: Optional[Callable[[PIL.Image.Image], torch.Tensor]] = None
		self.model_encode_text = None
		self.model_encode_image = None

		super().__init__(
			configuration={'model_name': self.model_name, 'model_checkpoint': clip.clip._MODELS[self.model_name]},  # noqa
			context_length=self.CONTEXT_LENGTH,
			vocab_size=len(self.tokenizer.encoder),
			cased_tokens=False,
			start_token_id=self.tokenizer.encoder["<|startoftext|>"],
			end_token_id=self.tokenizer.encoder["<|endoftext|>"],
			pad_token_id=self.tokenizer.encoder["<|endoftext|>"],  # Note: By default the padding token is 0, but this conflicts with another valid token ID, so we change the padding token to be the same as the end token (verified that this makes no difference)
			token_dtype=torch.int32,
			embed_dtype=torch.float32,
			embed_dim=self.EMBED_DIM[self.model_name],
			amp_mode=False,  # Note: The OpenAI implementation is already in a manual mixed precision configuration so enabling AMP does nothing
			manual_amp_dtype=torch.float16,
			tokenizer_batch_size=tokenizer_batch_size,
			inference_batch_size=inference_batch_size,
			image_batch_size=image_batch_size,
			load_model=load_model,
			compile_model=compile_model,
			device=device,
			check=check,
		)

	def load_model(self) -> bool:
		if self.model is None:
			self.model, self.preprocess_image = clip.load(self.model_name, device=self.device, download_root=self.CACHE_HOME)  # noqa
			self.model.eval()
			self.model_encode_text = torch.compile(self.model.encode_text) if self.compile_model else self.model.encode_text
			self.model_encode_image = torch.compile(self.model.encode_image) if self.compile_model else self.model.encode_image
			log.info(f"Loaded OpenAI embedder '{self.model_name}' onto {self.device.type.upper()} device{' (will compile on demand)' if self.compile_model else ''}")
			return True
		else:
			return False

	def unload_model(self) -> bool:
		if self.model is None:
			return False
		else:
			self.model_encode_image = None
			self.model_encode_text = None
			self.preprocess_image = None
			self.model = None
			log.info("Unloaded OpenAI embedder")
			return True

	def is_model_loaded(self) -> bool:
		return self.model is not None

	def tokenize(self, text: Union[str, Sequence[str]], max_tokens: Optional[int] = None, output_dict: bool = False) -> Union[torch.Tensor, dict[str, Any]]:
		if max_tokens is None:
			max_tokens = self.context_length
		if isinstance(text, str):
			text = (text,)
		token_tensors = []
		token_list = [self.start_token_id]
		for txt in text:
			del token_list[1:]
			token_list.extend(self.tokenizer.encode(txt))
			if len(token_list) >= max_tokens:
				del token_list[max_tokens - 1:]
			token_list.append(self.end_token_id)
			token_tensors.append(torch.tensor(token_list, dtype=self.token_dtype))
		token_ids = torch.nn.utils.rnn.pad_sequence(token_tensors, batch_first=True, padding_value=self.pad_token_id)
		if output_dict:
			attention_mask = torch.empty(token_ids.shape, dtype=self.token_dtype)
			attention_mask[:, 0] = 1
			attention_mask[:, 1:] = torch.ne(token_ids[:, :-1], self.pad_token_id)  # Note: Assumes padding token = end token
			tokens_dict = {'input_ids': token_ids, 'attention_mask': attention_mask}
			if self.check:
				tokens_dict['text'] = text
			return tokens_dict
		else:
			return token_ids

	def detokenize(self, token_ids: torch.Tensor) -> Union[str, list[str]]:
		token_lists = token_ids.tolist()
		if token_ids.ndim <= 1:
			return self.tokenizer.decode(token_id for token_id in token_lists if token_id != self.start_token_id and token_id != self.end_token_id).rstrip()  # Note: Assumes padding token = end token
		else:
			return [self.tokenizer.decode(token_id for token_id in token_list if token_id != self.start_token_id and token_id != self.end_token_id).rstrip() for token_list in token_lists]  # Note: Assumes padding token = end token

	def inference_tokens(self, tokens_dict: dict[str, Any]) -> torch.Tensor:

		token_ids = tokens_dict['input_ids']
		if self.device.type != 'cpu':
			token_ids = token_ids.pin_memory().to(device=self.device, non_blocking=True)

		seq_len = token_ids.shape[1]
		if seq_len > self.context_length:
			raise ValueError(f"Provided token sequences are longer than the context length: {seq_len} > {self.context_length}")
		elif seq_len < self.context_length:
			padded_token_ids = token_ids.new_full((token_ids.shape[0], self.context_length), fill_value=self.pad_token_id)
			padded_token_ids[:, :seq_len] = token_ids
			token_ids = padded_token_ids

		if self.check:
			check_token_ids = clip.tokenize(texts=tokens_dict['text'])  # noqa
			check_token_ids[torch.eq(check_token_ids, self.end_token_id).cummax(dim=1)[0]] = self.pad_token_id  # Note: Robustly change all tokens at and beyond the first end token to the required padding token (assumes padding token = end token)
			if self.device.type != 'cpu':
				check_token_ids = check_token_ids.pin_memory().to(device=self.device, non_blocking=True)
			if check_token_ids.dtype != token_ids.dtype:
				raise ValueError("Token ID consistency check failed due to dtype")
			elif not torch.equal(check_token_ids, token_ids):
				raise ValueError("Token ID consistency check failed due to shape or value")

		assert (self.amp_context is None or self.amp_context_entered) and not self.model.training
		text_features = self.model_encode_text(token_ids)
		return torch.nn.functional.normalize(text_features.to(dtype=torch.float32), dim=-1)

	def get_image_transform(self) -> Callable[[PIL.Image.Image], torch.Tensor]:
		assert self.preprocess_image
		return self.preprocess_image

	def inference_image(self, images: torch.Tensor) -> torch.Tensor:
		if self.device.type != 'cpu':
			images = images.pin_memory().to(device=self.device, non_blocking=True)
		assert (self.amp_context is None or self.amp_context_entered) and not self.model.training
		image_features = self.model_encode_image(images)
		return torch.nn.functional.normalize(image_features.to(dtype=torch.float32), dim=-1)

# OpenCLIP embedder class
class OpenCLIPEmbedder(Embedder):

	def __init__(
		self,
		model_id: str,
		amp: bool = True,
		amp_bf16: bool = False,
		tokenizer_batch_size: int = 1024,
		inference_batch_size: int = 256,
		image_batch_size: int = 128,
		load_model: bool = True,
		compile_model: bool = False,
		device: Union[int, str, torch.device] = 'cuda',
		check: bool = False,
	):

		self.model_id = model_id
		self.model_name = open_clip.factory.HF_HUB_PREFIX + self.model_id

		log.info(f"Loading OpenCLIP configuration for '{self.model_id}'")
		self.config = open_clip.factory._get_hf_config(self.model_id)  # noqa

		try:
			self.tokenizer = self._get_tokenizer(self.model_id)
		except ValueError as e:
			log.warning(f"Getting fast Hugging Face tokenizer failed with {type(e).__qualname__}: {e}")
			self.tokenizer = open_clip.get_tokenizer(self.model_name)
		if isinstance(self.tokenizer, open_clip.tokenizer.SimpleTokenizer):
			raise ValueError("SimpleTokenizer is slow, probably the wrong tokenizer to use anyway, and not easily robustly supported")
		elif not isinstance(self.tokenizer, open_clip.tokenizer.HFTokenizer):
			raise RuntimeError(f"Unsupported OpenCLIP tokenizer type: {type(self.tokenizer).__qualname__}")
		log.info(f"Loaded Hugging Face tokenizer: {type(self.tokenizer.tokenizer).__qualname__}")
		self.strip_sep_token = self.tokenizer.strip_sep_token  # Example: openclip:rwightman/ViT-L-14-CLIPA-datacomp1B
		self.tokenizer_clean = self.tokenizer.clean_fn is not open_clip.tokenizer._clean_whitespace  # noqa / Correct for OpenCLIP 2.23, and assuming vocab/prompts are guaranteed whitespace-perfect
		log.info(f"Loaded OpenCLIP tokenizer for '{self.model_id}': {type(self.tokenizer).__qualname__}{' with cleaning' if self.tokenizer_clean else ''}{' with strip sep token' if self.strip_sep_token else ''}")

		start_token_id = self.tokenizer.tokenizer.bos_token_id if self.tokenizer.tokenizer.bos_token_id is not None else self.tokenizer.tokenizer.cls_token_id
		end_token_id = self.tokenizer.tokenizer.eos_token_id if self.tokenizer.tokenizer.eos_token_id is not None else self.tokenizer.tokenizer.sep_token_id
		end_token = self.tokenizer.tokenizer.eos_token if self.tokenizer.tokenizer.eos_token_id is not None else self.tokenizer.tokenizer.sep_token
		pad_token_id = self.tokenizer.tokenizer.pad_token_id
		pad_token = self.tokenizer.tokenizer.pad_token
		pad_aliases = {token for token, token_id in self.tokenizer.tokenizer.vocab.items() if token_id == pad_token_id}
		pad_aliases.discard(pad_token)
		pad_aliases.discard(end_token)
		if pad_aliases:
			raise ValueError(f"Pad token {pad_token_id} cannot have non-end token aliases: {pad_aliases}")
		if self.strip_sep_token:
			end_token_id = pad_token_id

		self.model = None
		self.preprocess_image: Optional[Callable[[PIL.Image.Image], torch.Tensor]] = None
		self.model_encode_text = None
		self.model_encode_image = None

		super().__init__(
			configuration={'model_id': self.model_id, 'model_config': self.config},
			context_length=self.tokenizer.context_length,
			vocab_size=len(self.tokenizer.tokenizer),
			cased_tokens=(self.tokenizer.tokenizer.encode('CPU') != self.tokenizer.tokenizer.encode('cpu')),
			start_token_id=start_token_id,
			end_token_id=end_token_id,
			pad_token_id=pad_token_id,
			token_dtype=torch.int64,
			embed_dtype=torch.float32,
			embed_dim=self.config['model_cfg']['embed_dim'],
			amp_mode=False if not amp else torch.bfloat16 if amp_bf16 else True,
			manual_amp_dtype=None,
			tokenizer_batch_size=tokenizer_batch_size,
			inference_batch_size=inference_batch_size,
			image_batch_size=image_batch_size,
			load_model=load_model,
			compile_model=compile_model,
			device=device,
			check=check,
		)

	def _get_tokenizer(self, model_id: str):
		# Note: This method is a derivation of open_clip.get_tokenizer() at tag 2.23 (problem was that SimpleTokenizer was being returned for many models that actually have much faster HFTokenizers)
		text_config = self.config['model_cfg'].get('text_cfg', {})
		tokenizer_kwargs = dict(text_config['tokenizer_kwargs']) if 'tokenizer_kwargs' in text_config else {}
		context_length = text_config.get('context_length', open_clip.tokenizer.DEFAULT_CONTEXT_LENGTH)
		return open_clip.tokenizer.HFTokenizer(model_id, context_length=context_length, **tokenizer_kwargs)

	def load_model(self) -> bool:
		if self.model is None:
			self.model, _, self.preprocess_image = open_clip.create_model_and_transforms(model_name=self.model_name, device=self.device)
			self.model.eval()
			self.model_encode_text = torch.compile(self.model.encode_text) if self.compile_model else self.model.encode_text
			self.model_encode_image = torch.compile(self.model.encode_image) if self.compile_model else self.model.encode_image
			log.info(f"Loaded OpenCLIP embedder '{self.model_id}' onto {self.device.type.upper()} device{' (will compile on demand)' if self.compile_model else ''}")
			return True
		else:
			return False

	def unload_model(self) -> bool:
		if self.model is None:
			return False
		else:
			self.model_encode_image = None
			self.model_encode_text = None
			self.preprocess_image = None
			self.model = None
			log.info("Unloaded OpenCLIP embedder")
			return True

	def is_model_loaded(self) -> bool:
		return self.model is not None

	def tokenize(self, text: Union[str, Sequence[str]], max_tokens: Optional[int] = None, output_dict: bool = False) -> Union[torch.Tensor, dict[str, Any]]:
		if max_tokens is None:
			max_tokens = self.context_length
		if self.tokenizer_clean:
			text = self.tokenizer.clean_fn(text) if isinstance(text, str) else tuple(self.tokenizer.clean_fn(txt) for txt in text)
		tokenizer_output = self.tokenizer.tokenizer(text=text, padding=True, truncation=True, max_length=max_tokens, return_tensors='pt')
		if self.strip_sep_token:
			input_ids: torch.Tensor = tokenizer_output.data['input_ids']
			tokenizer_output.data['input_ids'] = torch.where(input_ids == self.tokenizer.tokenizer.sep_token_id, torch.tensor(self.pad_token_id, dtype=input_ids.dtype), input_ids)  # Correct for OpenCLIP 2.23 (although quite a different command)
		if output_dict:
			tokens_dict = dict(tokenizer_output)
		else:
			return tokenizer_output['input_ids']
		if self.check:
			tokens_dict['text'] = text
		return tokens_dict

	def detokenize(self, token_ids: torch.Tensor) -> Union[str, list[str]]:
		if token_ids.ndim <= 1:
			return self.tokenizer.tokenizer.decode(token_ids, skip_special_tokens=True)
		else:
			return self.tokenizer.tokenizer.batch_decode(token_ids, skip_special_tokens=True)

	def inference_tokens(self, tokens_dict: dict[str, Any]) -> torch.Tensor:

		token_ids = tokens_dict['input_ids']
		if self.device.type != 'cpu':
			token_ids = token_ids.pin_memory().to(device=self.device, non_blocking=True)

		seq_len = token_ids.shape[1]
		if seq_len > self.context_length:
			raise ValueError(f"Provided token sequences are longer than the context length: {seq_len} > {self.context_length}")
		elif seq_len < self.context_length:
			padded_token_ids = token_ids.new_full((token_ids.shape[0], self.context_length), fill_value=self.pad_token_id)
			padded_token_ids[:, :seq_len] = token_ids
			token_ids = padded_token_ids

		if self.check:
			check_token_ids = self.tokenizer(texts=tokens_dict['text'])
			if self.device.type != 'cpu':
				check_token_ids = check_token_ids.pin_memory().to(device=self.device, non_blocking=True)
			if check_token_ids.dtype != token_ids.dtype:
				raise ValueError("Token ID consistency check failed due to dtype")
			elif not torch.equal(check_token_ids, token_ids):
				raise ValueError("Token ID consistency check failed due to shape or value")

		assert (self.amp_context is None or self.amp_context_entered) and not self.model.training
		text_features = self.model_encode_text(token_ids, normalize=False)
		return torch.nn.functional.normalize(text_features.to(dtype=torch.float32), dim=-1)

	def get_image_transform(self) -> Callable[[PIL.Image.Image], torch.Tensor]:
		assert self.preprocess_image
		return self.preprocess_image

	def inference_image(self, images: torch.Tensor) -> torch.Tensor:
		if self.device.type != 'cpu':
			images = images.pin_memory().to(device=self.device, non_blocking=True)
		assert (self.amp_context is None or self.amp_context_entered) and not self.model.training
		image_features = self.model_encode_image(images, normalize=False)
		return torch.nn.functional.normalize(image_features.to(dtype=torch.float32), dim=-1)

# Transformers embedder class
class TransformersEmbedder(Embedder):

	def __init__(
		self,
		model_id: str,
		amp: bool = True,
		amp_bf16: bool = False,
		tokenizer_batch_size: int = 1024,
		inference_batch_size: int = 256,
		image_batch_size: int = 128,
		load_model: bool = True,
		compile_model: bool = False,
		use_optimum: bool = False,
		device: Union[int, str, torch.device] = 'cuda',
		check: bool = False,
	):

		self.model_id = model_id
		self.use_optimum = use_optimum

		log.info(f"Loading Transformers configuration for '{self.model_id}'")
		self.config = transformers.AutoConfig.from_pretrained(self.model_id)  # noqa
		self.model_type = self.config.model_type.upper()
		with contextlib.suppress(AttributeError):  # Example: laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K, laion/CLIP-ViT-bigG-14-laion2B-39B-b160k
			if self.config.text_config.projection_dim != self.config.projection_dim:
				log.info(f"Had to correct faulty configured text projection dimension from {self.config.text_config.projection_dim} to {self.config.projection_dim}")
				self.config.text_config.projection_dim = self.config.projection_dim
		with contextlib.suppress(AttributeError):
			if self.config.vision_config.projection_dim != self.config.projection_dim:
				log.info(f"Had to correct faulty configured vision projection dimension from {self.config.vision_config.projection_dim} to {self.config.projection_dim}")
				self.config.vision_config.projection_dim = self.config.projection_dim

		tokenizer_model_id = self.model_id
		if tokenizer_model_id.startswith('facebook/metaclip-'):  # Note: Should check whether this is ever actually fixed (e.g. https://huggingface.co/facebook/metaclip-h14-fullcc2.5b/discussions/1)
			tokenizer_model_id = 'openai/clip-vit-base-patch32'
			log.info(f"Facebook MetaCLIP tokenizer definition is dodgy but should be equivalent to '{tokenizer_model_id}', so hacking together that tokenizer instead!")
		self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_model_id)  # noqa
		log.info(f"Loaded Transformers {self.model_type} tokenizer for '{self.model_id}': {type(self.tokenizer).__qualname__}")

		start_token_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.tokenizer.cls_token_id
		end_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else self.tokenizer.sep_token_id
		end_token = self.tokenizer.eos_token if self.tokenizer.eos_token_id is not None else self.tokenizer.sep_token
		pad_token_id = self.tokenizer.pad_token_id
		pad_token = self.tokenizer.pad_token
		pad_aliases = {token for token, token_id in self.tokenizer.vocab.items() if token_id == pad_token_id}
		pad_aliases.discard(pad_token)
		pad_aliases.discard(end_token)
		if pad_aliases:
			raise ValueError(f"Pad token {pad_token_id} cannot have non-end token aliases: {pad_aliases}")

		self.model = None
		self.image_processor = None  # Note: Is of a class like CLIPImageProcessor, which has a method preprocess(images, return_tensors='pt', **kwargs)
		self.model_get_text_features = None
		self.model_get_image_features = None

		super().__init__(
			configuration={'model_id': self.model_id, 'model_config': format(self.config)},
			context_length=self.tokenizer.model_max_length,
			vocab_size=len(self.tokenizer),
			cased_tokens=(self.tokenizer.encode('CPU') != self.tokenizer.encode('cpu')),
			start_token_id=start_token_id,
			end_token_id=end_token_id,
			pad_token_id=pad_token_id,
			token_dtype=torch.int64,
			embed_dtype=torch.float32,
			embed_dim=self.config.projection_dim,
			amp_mode=False if not amp else torch.bfloat16 if amp_bf16 else True,
			manual_amp_dtype=None,
			tokenizer_batch_size=tokenizer_batch_size,
			inference_batch_size=inference_batch_size,
			image_batch_size=image_batch_size,
			load_model=load_model,
			compile_model=compile_model,
			device=device,
			check=check,
		)

	def load_model(self) -> bool:
		if self.model is None:
			self.model = transformers.AutoModel.from_pretrained(self.model_id, config=self.config, device_map=self.device)  # noqa
			self.image_processor = transformers.AutoImageProcessor.from_pretrained(self.model_id, config=self.config)  # noqa
			if self.use_optimum:
				self.model = optimum.bettertransformer.BetterTransformer.transform(self.model)
			self.model.eval()
			self.model_get_text_features = torch.compile(self.model.get_text_features) if self.compile_model else self.model.get_text_features
			self.model_get_image_features = torch.compile(self.model.get_image_features) if self.compile_model else self.model.get_image_features
			log.info(f"Loaded {'Optimum ' if self.use_optimum else ''}Transformers {self.model_type} embedder '{self.model_id}' onto {self.device.type.upper()} device{' (will compile on demand)' if self.compile_model else ''}")
			return True
		else:
			return False

	def unload_model(self) -> bool:
		if self.model is None:
			return False
		else:
			self.model_get_image_features = None
			self.model_get_text_features = None
			self.image_processor = None
			self.model = None
			log.info(f"Unloaded Transformers {self.model_type} embedder")
			return True

	def is_model_loaded(self) -> bool:
		return self.model is not None

	def tokenize(self, text: Union[str, Sequence[str]], max_tokens: Optional[int] = None, output_dict: bool = False) -> Union[torch.Tensor, dict[str, Any]]:
		# Note: Max tokens of None means to use default model maximum tokens (e.g. 77 for CLIP, 64 for ALIGN)
		tokenizer_output = self.tokenizer(text=text, padding=True, truncation=True, max_length=max_tokens, return_tensors='pt')
		if output_dict:
			return dict(tokenizer_output)
		else:
			return tokenizer_output['input_ids']

	def detokenize(self, token_ids: torch.Tensor) -> Union[str, list[str]]:
		if token_ids.ndim <= 1:
			return self.tokenizer.decode(token_ids, skip_special_tokens=True)
		else:
			return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)

	def inference_tokens(self, tokens_dict: dict[str, Any]) -> torch.Tensor:
		if self.device.type != 'cpu':
			tokens_dict = {key: tensor.pin_memory().to(device=self.device, non_blocking=True) for key, tensor in tokens_dict.items()}
		assert (self.amp_context is None or self.amp_context_entered) and not self.model.training
		text_features = self.model_get_text_features(**tokens_dict)
		return torch.nn.functional.normalize(text_features.to(dtype=torch.float32), dim=-1)

	def get_image_transform(self) -> Callable[[PIL.Image.Image], torch.Tensor]:
		assert self.image_processor
		return self.preprocess_image

	def preprocess_image(self, image: PIL.Image.Image) -> torch.Tensor:
		preprocess_output = self.image_processor(images=image, return_tensors='pt')
		assert len(preprocess_output) == 1  # Note: If there is not just a single value then this also needs to be supplied to self.model_get_image_features() in self.inference_image()
		return preprocess_output['pixel_values'].squeeze(dim=0)

	def inference_image(self, images: torch.Tensor) -> torch.Tensor:
		if self.device.type != 'cpu':
			images = images.pin_memory().to(device=self.device, non_blocking=True)
		assert (self.amp_context is None or self.amp_context_entered) and not self.model.training
		image_features = self.model_get_image_features(pixel_values=images)
		return torch.nn.functional.normalize(image_features.to(dtype=torch.float32), dim=-1)
# EOF
