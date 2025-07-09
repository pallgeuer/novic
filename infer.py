#!/usr/bin/env python3
# Inference a NOVIC model

# Imports
from __future__ import annotations
import gc
import os
import re
import argparse
import itertools
import contextlib
import dataclasses
from enum import Enum
from typing import Iterable, Union, Optional, Any, ContextManager, Callable, Type, Sequence
import PIL.Image
import torch
import torch.utils.data
from logger import log
import utils
import embedders
import embedding_dataset
import embedding_decoder

#
# NOVIC model
#

# Prediction type enumeration
class PredictionType(Enum):  # GenerationTask.COLOR_MAP provides colors for pretty printing results of these types
	Correct = 0              # Correct prediction (only possible if ground truth is known and was provided)
	ValidGuide = 1           # Else if the prediction is a valid guide target
	ValidVocab = 2           # Else if the prediction is a valid vocabulary target
	Other = 3                # Otherwise

# NOVIC output class
@dataclasses.dataclass(frozen=True)
class NOVICOutput:
	embeds: torch.Tensor                           # BxF CPU tensor of image embeddings
	preds: tuple[tuple[str, ...], ...]             # Top-K string label predictions for each sample
	logprobs: tuple[tuple[float, ...], ...]        # Top-K label log-probabilities in descending order for each sample
	probs: tuple[tuple[float, ...], ...]           # Top-K label probabilities in descending order for each sample
	types: tuple[tuple[PredictionType, ...], ...]  # Top-K prediction types for each sample

# NOVIC model class
class NOVICModel:

	def __init__(
		self,
		checkpoint: str,                                        # Decoder checkpoint to load
		*,
		gencfg: str = 'beam_k10_vnone_gp_t1_a0',                # Generation configuration to use
		guide_targets: Union[Iterable[str], str, None] = None,  # If guided decoding is enabled, a manual list (iterable or path to text file with one per line) of target nouns to use for guided decoding (None = Use model vocabulary)
		torch_compile: bool = False,                            # Whether to torch-compile the decoder
		batch_size: int = 128,                                  # Nominal batch size
		device: Union[torch.device, str, int] = 'cuda',         # Torch device to use
		cfg_flat_override: Optional[dict[str, Any]] = None,     # Manually override some loaded flat configuration parameters (nested parameters are indexed by a single compound key joined by '.', see utils.flatten_dict())
		embedder_override: Optional[dict[str, Any]] = None,     # Manually override some keyword arguments to pass to embedders.Embedder.create()
	):

		self.checkpoint = os.path.abspath(checkpoint)
		self.checkpoint_tail = os.path.join(os.path.basename(os.path.dirname(self.checkpoint)), os.path.basename(self.checkpoint))
		log.info(f"Using decoder model checkpoint: {self.checkpoint}")

		lazy_checkpoint = torch.load(self.checkpoint, map_location='cpu', mmap=True)  # Memory mapping of the file ensures that the tensor storages are not copied from disk to CPU memory, making this a much more efficient and memory-saving operation
		cfg_flat = lazy_checkpoint['cfg_flat']
		log.info(f"Loaded {len(cfg_flat)} configuration parameters from the decoder model checkpoint")
		if cfg_flat_override is not None:
			log.info(f"Manually overriding {len(cfg_flat_override)} configuration parameters")
			cfg_flat.update(cfg_flat_override)
		self.cfg = utils.AttrDict.from_dict(utils.unflatten_dict(cfg_flat))
		del lazy_checkpoint, cfg_flat
		gc.collect()

		self.gencfg = GenerationConfig.from_name(name=gencfg)
		log.info(f"Using generation config: {self.gencfg.name}")

		if guide_targets is None:
			self.guide_targets = None
		elif isinstance(guide_targets, str):
			with open(guide_targets, 'r') as file:
				self.guide_targets = tuple(stripped_line for line in file if (stripped_line := line.strip()))
		else:
			self.guide_targets = tuple(guide_targets)

		self.torch_compile = torch_compile
		self.batch_size = batch_size

		self.device, self.device_is_cpu, self.device_is_cuda = load_device(device=device)
		log.info(f"Using torch device: {self.device}")

		embedder_create_kwargs = dict(
			spec=self.cfg.embedder_spec,
			amp=self.cfg.embedder_amp,
			amp_bf16=self.cfg.embedder_amp_bf16,
			tokenizer_batch_size=self.batch_size,
			inference_batch_size=self.batch_size,
			image_batch_size=self.batch_size,
			load_model=False,
			compile_model=self.cfg.embedder_compile,
			use_optimum=self.cfg.embedder_optimum,
			device=self.device,
			check=False,
		)
		if embedder_override is not None:
			embedder_create_kwargs.update(embedder_override)

		log.info(f"Creating embedder of specification {embedder_create_kwargs['spec']}{' with checking' if embedder_create_kwargs['check'] else ''}...")
		self.embedder = embedders.Embedder.create(**embedder_create_kwargs)
		log.info(f"Created embedder of class type {type(self.embedder).__qualname__}")

		self.amp_context, self.amp_dtype = load_decoder_amp(enabled=self.cfg.amp, bf16=self.cfg.amp_bf16, determ=False, device=self.device)  # Assumes determinism is not required
		self.data_config = embedding_dataset.DataConfig.create(data_config_dict=dict(use_weights=False, multi_target=False), use_targets=True)

		self.gentask: Optional[GenerationTask] = None
		self.decoder: Optional[embedding_decoder.EmbeddingDecoder] = None

		self.amp_context_entered = False
		self.__stack = contextlib.ExitStack()

	@contextlib.contextmanager
	def decoder_model(self, release=True):
		# Context manager that temporarily loads the decoder model if it is not currently loaded
		# release = Whether to release CUDA memory after unloading the decoder model
		if self.is_decoder_loaded():
			yield
		else:
			try:
				self.load_decoder()
				yield
			finally:
				self.unload_decoder()
				if release:
					utils.release_cuda_memory(device=self.device)

	def load_decoder(self) -> bool:
		# Returns whether the decoder was loaded (True = Newly loaded, False = Was already loaded)

		if self.decoder is not None:
			return False

		log.info(f"Loading decoder model checkpoint: {self.checkpoint}")
		checkpoint = torch.load(self.checkpoint, map_location='cpu')  # We load to CPU for more control of GPU memory spikes, and because target configuration has tensors that need to stay on CPU
		load_target_config(checkpoint=checkpoint, embedder=self.embedder)

		model_targets = self.embedder.target_vocab
		model_targets_set = set(model_targets)
		vocab_targets_tensor = self.embedder.tokenize_target(model_targets)[0]
		if not self.device_is_cpu:
			vocab_targets_tensor = vocab_targets_tensor.to(device=self.device, non_blocking=True)

		guide_targets_str = self.guide_targets if self.guide_targets is not None else model_targets
		if not ((guide_targets_str_set := set(guide_targets_str)) <= model_targets_set):
			log.warning(f"Some guide target nouns are not in the set of trained model target nouns: {', '.join(sorted(guide_targets_str_set - model_targets_set))}")
		guide_targets_tensor = load_guide_targets(guide_targets=guide_targets_str, embedder=self.embedder, device=self.device, device_is_cpu=self.device_is_cpu)

		with torch.inference_mode():
			self.decoder = load_decoder_model(cfg=self.cfg, embedder=self.embedder, data_config=self.data_config, checkpoint=checkpoint)
			if not self.device_is_cpu:
				log.info(f"Moving decoder model to {self.device.type.upper()}...")
				self.decoder.to(device=self.device)
			if self.torch_compile:
				log.info("Will compile decoder model when it is used")
				self.decoder = torch.compile(self.decoder)
			self.decoder.eval()

		del checkpoint
		gc.collect()

		self.gentask = GenerationTask(
			gencfg=self.gencfg,
			decoder=self.decoder,
			vocab_targets_set=model_targets_set,
			vocab_targets=vocab_targets_tensor,
			guide_targets_set=guide_targets_str_set,
			guide_targets=guide_targets_tensor,
			class_lists=None,
		)

		return True

	def unload_decoder(self) -> bool:
		# Returns whether the decoder was unloaded (True = Newly unloaded, False = Was already unloaded)
		if self.decoder is None:
			return False
		self.gentask = None
		self.decoder = None
		log.info("Unloaded decoder model")
		return True

	def is_decoder_loaded(self) -> bool:
		return self.decoder is not None

	@contextlib.contextmanager
	def inference_mode(self):
		# Context manager that temporarily sets up PyTorch inference mode and AMP for running model inferences
		with torch.inference_mode():
			if self.amp_context_entered:
				yield
			else:
				self.amp_context_entered = True
				try:
					with self.amp_context:
						yield
				finally:
					self.amp_context_entered = False

	@classmethod
	def load_image(cls, image_path: str) -> PIL.Image.Image:
		# image_path = Image path
		# Returns the loaded RGB PIL image
		image = PIL.Image.open(image_path)
		image.load()
		if image.mode != 'RGB':
			image = image.convert('RGB')
		return image

	@classmethod
	def load_images(cls, image_paths: Iterable[str], *, image_dir: Optional[str] = None) -> list[PIL.Image.Image]:
		# image_paths = Image paths (relative paths are resolved with respect to image_dir)
		# image_dir = Directory relative to which to resolve relative image paths (None = Current directory)
		# Returns a list of the loaded RGB PIL images
		if image_dir is None:
			image_dir = ''
		return [cls.load_image(image_path=os.path.join(image_dir, image_path)) for image_path in image_paths]

	def load_image_batches(self, image_paths: Iterable[str], *, image_dir: Optional[str] = None, batch_size: Optional[int] = None) -> list[list[PIL.Image.Image]]:
		# image_paths = Image paths (relative paths are resolved with respect to image_dir)
		# image_dir = Directory relative to which to resolve relative image paths (None = Current directory)
		# batch_size = Custom batch size (None = Use nominal batch size)
		# Returns a list of the loaded RGB PIL image batches (the last batch may be smaller than the batch size)

		if image_dir is None:
			image_dir = ''
		if batch_size is None:
			batch_size = self.batch_size

		image_batches = []
		image_paths_iter = iter(image_paths)
		while image_paths_batch := tuple(itertools.islice(image_paths_iter, batch_size)):
			image_batches.append([self.load_image(image_path=os.path.join(image_dir, image_path)) for image_path in image_paths_batch])

		return image_batches

	def get_image_transform(self) -> Callable[[PIL.Image.Image], torch.Tensor]:  # Returns the required image transform (for preprocessing images to CPU tensor)
		return self.embedder.get_image_transform()

	def transform_images(self, images: Union[Iterable[PIL.Image.Image], PIL.Image.Image]) -> torch.Tensor:
		# images = Image(s) to transform/preprocess to CPU tensor form
		# Returns the preprocessed CPU image(s) tensor (Bx3xHxW)
		if isinstance(images, PIL.Image.Image):
			images = (images,)
		image_transform = self.get_image_transform()
		return torch.utils.data.default_collate(tuple(image_transform(image) for image in images))

	def __enter__(self) -> NOVICModel:
		# Ensure the NOVIC model is loaded and ready for inference
		with self.__stack as stack:
			stack.enter_context(self.embedder.inference_model(release=True))
			stack.enter_context(self.decoder_model(release=False))
			self.__stack = stack.pop_all()
		assert self.__stack is not stack
		return self

	def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
		# Unload the NOVIC model if it is currently loaded
		return self.__stack.__exit__(exc_type, exc_val, exc_tb)

	def embed_images(self, images: Union[torch.Tensor, Iterable[PIL.Image.Image], PIL.Image.Image]) -> torch.Tensor:
		# images = Image(s) to embed (CPU Bx3xHxW if tensor, PIL image(s) are first transformed/preprocessed)
		# Returns the image embeddings (BxF) on the target device
		if not isinstance(images, torch.Tensor):
			images = self.transform_images(images=images)
		with self.embedder.inference_mode():
			return self.embedder.inference_image(images=images)

	def classify_embeds(self, embeds: torch.Tensor) -> NOVICOutput:
		# embeds = Image embeddings (BxF) on the target device
		# Returns the NOVIC classification output details

		with self.inference_mode():
			self.gentask.process(embeds=embeds)

		return NOVICOutput(
			embeds=embeds.cpu(),
			preds=tuple(tuple(' '.join(target.split()) for target in target_list) for target_list in self.gentask.target_str),
			logprobs=tuple(tuple(score for score in score_list) for score_list in self.gentask.target_score),
			probs=tuple(tuple(score for score in score_list) for score_list in self.gentask.target_score),
			types=tuple(tuple(PredictionType(result) for result in result_list) for result_list in self.gentask.result.tolist()),
		)

	def classify_images(self, images: Union[torch.Tensor, Iterable[PIL.Image.Image], PIL.Image.Image]) -> NOVICOutput:
		# images = Image(s) to classify (CPU Bx3xHxW if tensor, PIL image(s) are first transformed/preprocessed)
		# Returns the NOVIC classification output details
		embeds = self.embed_images(images=images)
		return self.classify_embeds(embeds=embeds)

	def __call__(self, images: Union[torch.Tensor, Iterable[PIL.Image.Image], PIL.Image.Image]) -> NOVICOutput:
		# See classify_images()
		return self.classify_images(images=images)

#
# Helper classes
#

# Generation config class
@dataclasses.dataclass(frozen=True)
class GenerationConfig:

	method: str            # Decoding method (Available: greedy, beam, all)
	topk: int              # If not greedy, number of top nouns to generate
	vocab_prior: bool      # If not greedy, whether to score tokens relative to their prior distribution
	vocab_per_token: bool  # If not greedy and if vocab_prior, whether vocab priors are on a per-token or per-target basis
	vocab_scaler: float    # If not greedy and if vocab_prior, dimensionless scaler for the effect of vocab priors
	guided: bool           # Whether decoding should be guided
	guide_renorm: bool     # If guided, whether guiding should affect/renormalise the scores
	temperature: float     # Temperature to divide the logits by prior to softmax during decoding (must be positive non-zero)
	length_alpha: float    # Length normalisation factor alpha (nominally 0-1)
	name: str = dataclasses.field(init=False)  # Compact string name summarising the generation config

	def __post_init__(self):
		object.__setattr__(self, 'name', self.generate_name())

	def generate_name(self) -> str:
		vocab_prior = f"{'tok' if self.vocab_per_token else 'tgt'}{utils.format_semifix(self.vocab_scaler, precision=3)}" if self.vocab_prior else 'none'
		return f"{self.method}_k{self.topk}_v{vocab_prior}_g{'n' if not self.guided else 'r' if self.guide_renorm else 'p'}_t{utils.format_semifix(self.temperature, precision=3)}_a{utils.format_semifix(self.length_alpha, precision=3)}"

	@staticmethod
	def from_name(name: str) -> GenerationConfig:

		parts = name.split('_')
		method = parts[0]

		topk = 0  # Required (0 is an invalid value that will error below)
		vocab_prior = False
		vocab_per_token = False
		vocab_scaler = 0.0
		guided = False
		guide_renorm = False
		tau = 1.0
		alpha = 0.0

		for part in itertools.islice(parts, 1, None):
			if not part:
				raise ValueError(f"Unexpected multiple underscores in generation configuration: {name}")
			prefix = part[:1]
			value = part[1:]
			try:
				if prefix == 'k':
					topk = int(value)
				elif prefix == 'v':
					if value != 'none':
						vocab_prior = True
						match = re.fullmatch(r'(tok|tgt)(.*)', value)
						try:
							vocab_per_token = (match.group(1) == 'tok')
							vocab_scaler = float(match.group(2))
						except (AttributeError, ValueError):
							raise ValueError(f"Invalid vocab prior specification: {value}")
				elif prefix == 'g':
					if value not in ('n', 'p', 'r'):
						raise ValueError(f"Invalid guide specification: {value}")
					guided = (value != 'n')
					guide_renorm = (value == 'r')
				elif prefix == 't':
					tau = float(value)
				elif prefix == 'a':
					alpha = float(value)
				else:
					raise ValueError(f"Invalid prefix: {prefix}")
			except ValueError:
				raise ValueError(f"Failed to parse generation configuration part: {part}")

		gencfg = GenerationConfig(method=method, topk=topk, vocab_prior=vocab_prior, vocab_per_token=vocab_per_token, vocab_scaler=vocab_scaler, guided=guided, guide_renorm=guide_renorm, temperature=tau, length_alpha=alpha)
		if gencfg.method not in ('greedy', 'beam', 'all'):
			raise ValueError(f"Invalid generation configuration method: {gencfg.method}")
		if gencfg.topk < 1:
			raise ValueError(f"Missing or invalid non-positive generation configuration top-k: {gencfg.topk}")
		if gencfg.temperature <= 0:
			raise ValueError(f"Invalid non-positive generation configuration temperature tau: {gencfg.temperature}")
		assert gencfg.name == name

		return gencfg

# Generation task class
@dataclasses.dataclass(eq=False)
class GenerationTask:

	COLOR_MAP = ('\033[92m', '\033[35m', '\033[33m', '\033[91m')  # ANSI colors corresponding to the 'result' field values

	gencfg: GenerationConfig                                      # Generation configuration
	decoder: embedding_decoder.EmbeddingDecoder                   # Embedding decoder model
	vocab_targets_set: set[str]                                   # Set of string vocabulary targets
	vocab_targets: Optional[torch.Tensor]                         # Tensor of tokenized vocabulary targets (on target device, can be None if not required by gencfg)
	guide_targets_set: set[str]                                   # Set of string guide targets
	guide_targets: Optional[torch.Tensor]                         # Tensor of tokenized guide targets (on target device, can be None if not required by gencfg)
	class_lists: Optional[Sequence[Sequence[str]]] = None         # Optional ground truth class lists (defines which string predictions are considered correct for each class index)

	precompute: Optional[Any] = None                              # Any precomputed task data if required (tensors may be on target device)

	target: Optional[torch.Tensor] = None                         # BxKxC predicted token IDs tensor
	target_padding: Optional[torch.Tensor] = None                 # BxKxC predicted token padding tensor
	target_score: Optional[list[list[float]]] = None              # BxK list-of-lists prediction scores

	num_samples: int = 0                                          # Total number of samples seen (sum of all seen B's)
	target_str: Optional[list[list[str]]] = None                  # BxK list-of-lists of predicted top-K noun strings
	invalid: Optional[torch.Tensor] = None                        # BxK boolean tensor whether predicted top-K noun is invalid (not correct, not valid guide, not valid vocab)
	valid_vocab: Optional[torch.Tensor] = None                    # BxK boolean tensor whether predicted top-K noun is valid vocab
	valid_guide: Optional[torch.Tensor] = None                    # BxK boolean tensor whether predicted top-K noun is valid guide
	correct: Optional[torch.Tensor] = None                        # BxK boolean tensor whether predicted top-K noun is correct
	result: Optional[torch.Tensor] = None                         # BxK integer tensor (0 = If correct, 1 = Else if valid guide, 2 = Else if valid vocab, 3 = Otherwise invalid)
	topk_counts: torch.Tensor = dataclasses.field(init=False)     # Kx4 integer tensor of top-k counts per result type (need to be divided by num samples to get actual top-k)
	topk_invalid: Optional[torch.Tensor] = None                   # K tensor of top-k any invalid ratios
	topk_valid: Optional[torch.Tensor] = None                     # K tensor of top-k all valid ratios
	topk_vocab: Optional[torch.Tensor] = None                     # K tensor of top-k any valid vocab ratios
	topk_guide: Optional[torch.Tensor] = None                     # K tensor of top-k any valid guide ratios
	topk: Optional[torch.Tensor] = None                           # K tensor of top-k any correct ratios

	def __post_init__(self):

		self.topk_counts = torch.zeros((self.gencfg.topk, 4), dtype=torch.int64)

		if self.gencfg.vocab_prior and self.vocab_targets is None:
			raise ValueError("Generation config specifies to use vocab priors but no vocab targets were provided")
		if self.gencfg.guided and self.guide_targets is None:
			raise ValueError("Generation config is guided but no guide targets were provided")

		if self.gencfg.method == 'greedy':
			if self.gencfg.topk != 1:
				raise ValueError(f"Top-k must be 1 for greedy generation: {self.gencfg.topk}")
			if self.gencfg.vocab_prior:
				raise ValueError("Greedy generation does not support vocab priors")
		elif self.gencfg.method == 'all':
			if not self.gencfg.guided:
				raise ValueError(f"The '{self.gencfg.method}' generation method must always be guided")

	def clear(self, clear_precompute: bool = False):
		# clear_precompute = Whether to also clear any precomputed data

		if clear_precompute:
			self.precompute = None

		self.target = None
		self.target_padding = None
		self.target_score = None

		self.num_samples = 0
		self.target_str = None
		self.invalid = None
		self.valid_vocab = None
		self.valid_guide = None
		self.correct = None
		self.result = None
		self.topk_counts = torch.zeros((self.gencfg.topk, 4), dtype=torch.int64)
		self.topk_invalid = None
		self.topk_valid = None
		self.topk_vocab = None
		self.topk_guide = None
		self.topk = None

	def process(self, embeds: torch.Tensor, *, class_indices: Optional[Sequence[int]] = None, precompute: bool = True, precompute_cache: Optional[dict[tuple[Any, ...], Any]] = None):
		# embeds = Image embeddings (BxF) on the target device
		# class_indices = Optional sample ground truth classes as a 1D sequence of length B of 0-indexed class indices (indexes class_lists)
		# precompute = Whether to ensure precomputation (if applicable) to store/reuse the results of some calculations
		# precompute_cache = Optionally cache the results of precomputation so that they can be reused across generation tasks
		# Note: It is assumed that this method is called within torch inference mode, and an appropriate entered decoder model AMP context
		target, target_padding, target_score = self.generate(embeds=embeds, precompute=precompute, precompute_cache=precompute_cache)
		self.update(target=target, target_padding=target_padding, target_score=target_score, class_indices=class_indices)

	def ensure_precomputed(self, precompute_cache: Optional[dict[tuple[Any, ...], Any]] = None):
		# precompute_cache = Optionally cache the results of precomputation so that they can be reused across generation tasks
		# Note: It is assumed that this method is called within torch inference mode, and an appropriate entered decoder model AMP context
		if self.precompute is None:
			self.perform_precompute(precompute_cache=precompute_cache)

	def perform_precompute(self, precompute_cache: Optional[dict[tuple[Any, ...], Any]] = None) -> Any:
		# precompute_cache = Optionally cache the results of precomputation so that they can be reused across generation tasks
		# Returns the precomputed data
		# Note: It is assumed that this method is called within torch inference mode, and an appropriate entered decoder model AMP context

		if self.gencfg.method == 'all':
			precompute_method = self.decoder.precompute_generate_all
			precompute_kwargs = dict(
				length_alpha=self.gencfg.length_alpha,
				vocab_targets=self.vocab_targets if self.gencfg.vocab_prior else None,
				vocab_per_token=self.gencfg.vocab_per_token,
				vocab_scaler=self.gencfg.vocab_scaler,
				guide_targets=self.guide_targets,
				guide_renorm=self.gencfg.guide_renorm,
			)
		else:
			self.precompute = None
			return self.precompute

		if precompute_cache is None:
			self.precompute = precompute_method(**precompute_kwargs)
		else:
			precompute_key = (self.gencfg.method, *precompute_kwargs.values())  # Note: Tensors hash by identity
			if (precompute := precompute_cache.get(precompute_key, None)) is None:
				precompute = precompute_method(**precompute_kwargs)
				precompute_cache[precompute_key] = precompute
			self.precompute = precompute

		return self.precompute

	def generate(self, embeds: torch.Tensor, *, precompute: bool = True, precompute_cache: Optional[dict[tuple[Any, ...], Any]] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		# embeds = Image embeddings (BxF) on the target device
		# precompute = Whether to ensure precomputation (if applicable) to store/reuse the results of some calculations
		# precompute_cache = Optionally cache the results of precomputation so that they can be reused across generation tasks
		# Returns BxKxC predicted token IDs tensor, BxKxC predicted token padding tensor, and BxK prediction scores tensor in descending order (all on target device)
		# Note: It is assumed that this method is called within torch inference mode, and an appropriate entered decoder model AMP context

		if precompute:
			self.ensure_precomputed(precompute_cache=precompute_cache)

		if self.gencfg.method == 'greedy':
			target, target_padding, _, _, _, target_score = self.decoder.generate(
				embed=embeds,
				collect_logits=False,
				calc_loss=True,
				temperature=self.gencfg.temperature,
				length_alpha=self.gencfg.length_alpha,
				sample_weight=None,
				guide_targets=self.guide_targets if self.gencfg.guided else None,
				guide_renorm=self.gencfg.guide_renorm,
			)
			target = target.unsqueeze(dim=1)
			target_padding = target_padding.unsqueeze(dim=1)
			target_score = target_score.unsqueeze(dim=1)

		elif self.gencfg.method == 'beam':
			target, target_padding, target_score = self.decoder.generate_beam(
				embed=embeds,
				topk=self.gencfg.topk,
				temperature=self.gencfg.temperature,
				length_alpha=self.gencfg.length_alpha,
				vocab_targets=self.vocab_targets if self.gencfg.vocab_prior else None,
				vocab_per_token=self.gencfg.vocab_per_token,
				vocab_scaler=self.gencfg.vocab_scaler,
				guide_targets=self.guide_targets if self.gencfg.guided else None,
				guide_renorm=self.gencfg.guide_renorm,
			)

		elif self.gencfg.method == 'all':
			target, target_padding, target_score = self.decoder.generate_all(
				embed=embeds,
				topk=self.gencfg.topk,
				temperature=self.gencfg.temperature,
				length_alpha=self.gencfg.length_alpha,
				vocab_targets=self.vocab_targets if self.gencfg.vocab_prior else None,
				vocab_per_token=self.gencfg.vocab_per_token,
				vocab_scaler=self.gencfg.vocab_scaler,
				guide_targets=self.guide_targets,
				guide_renorm=self.gencfg.guide_renorm,
				precompute=self.precompute,
			)

		else:
			raise ValueError(f"Unsupported generation method: {self.gencfg.method}")

		return target, target_padding, target_score

	def update(
		self,
		target: torch.Tensor,                           # BxKxC predicted token IDs tensor (any device)
		target_padding: torch.Tensor,                   # BxKxC predicted token padding tensor (any device)
		target_score: torch.Tensor,                     # BxK prediction scores tensor in descending order (any device)
		*,
		class_indices: Optional[Sequence[int]] = None,  # Optional sample ground truth classes as a 1D sequence of length B of 0-indexed class indices (indexes class_lists)
	):

		self.target = target.cpu()
		self.target_padding = target_padding.cpu()
		self.target_score = target_score.tolist()

		self.num_samples += self.target.shape[0]
		self.target_str = self.decoder.embedder.detokenize_target(self.target)
		self.valid_vocab = torch.tensor(tuple(tuple(pred in self.vocab_targets_set for pred in preds) for preds in self.target_str), dtype=torch.bool)
		self.valid_guide = torch.tensor(tuple(tuple(pred in self.guide_targets_set for pred in preds) for preds in self.target_str), dtype=torch.bool)
		if class_indices is not None and self.class_lists is not None:
			self.correct = torch.tensor(tuple(tuple(pred in self.class_lists[target] for pred in preds) for target, preds in zip(class_indices, self.target_str)), dtype=torch.bool)
		else:
			self.correct = torch.zeros(size=self.target.shape[:-1], dtype=torch.bool)
		self.invalid = torch.logical_or(self.valid_vocab, self.valid_guide).logical_or_(self.correct).logical_not_()  # noqa
		self.result = torch.max((stacked_results := torch.stack(tensors=(self.correct, self.valid_guide, self.valid_vocab, torch.ones_like(self.invalid)), dim=2).cummax(dim=2)[0]), dim=2)[1]
		stacked_results[:, :, -1] = self.invalid
		self.topk_counts.add_(stacked_results.cummax(dim=1)[0].sum(dim=0))
		topk_counts = self.topk_counts.to(dtype=self.decoder.embedder.embed_dtype)
		self.topk_valid = (self.num_samples - topk_counts[:, 3]).div_(self.num_samples)
		topk_ratios = topk_counts.div_(self.num_samples)
		self.topk_invalid = topk_ratios[:, 3]
		self.topk_vocab = topk_ratios[:, 2]
		self.topk_guide = topk_ratios[:, 1]
		self.topk = topk_ratios[:, 0]

#
# Helper functions
#

# Load the torch device
def load_device(device: Union[torch.device, str, int]) -> tuple[torch.device, bool, bool]:
	device = torch.device(device)
	if device.type == 'cuda' and not torch.cuda.is_available():
		log.warning("No CUDA device is available => Running on CPU instead")
		device = torch.device('cpu')
	device = torch.empty(size=(), device=device).device  # This is required to ensure a resolved device index, as device(type='cuda') != device(type='cuda', index=0) even if there is only one CUDA device and tensor.device always has a resolved index
	device_is_cpu = (device.type == 'cpu')
	device_is_cuda = (device.type == 'cuda')
	return device, device_is_cpu, device_is_cuda

# Load AMP for the embedding decoder model
def load_decoder_amp(enabled: bool, bf16: bool, determ: bool, device: torch.device) -> tuple[ContextManager, Optional[torch.dtype]]:

	if enabled and (determ or device.type == 'cpu'):
		log.warning("Decoder AMP was automatically disabled due to determinism or CPU device")
		enabled = False

	if enabled:
		amp_context = torch.autocast(device.type, dtype=torch.bfloat16 if bf16 else None)
		amp_dtype = amp_context.fast_dtype
		log.info(f"Decoder AMP is enabled with dtype {amp_dtype}")
	else:
		amp_context = contextlib.nullcontext()
		amp_dtype = None
		log.info("Decoder AMP is disabled")

	return amp_context, amp_dtype

# Load the target configuration from an already loaded embedding decoder model checkpoint
def load_target_config(checkpoint: dict[str, Any], embedder: embedders.Embedder) -> embedders.TargetConfig:
	log.info(f"Loading {len(checkpoint['target_config'])} target configuration items from checkpoint...")
	target_config = utils.dataclass_from_dict(cls=embedders.TargetConfig, state=checkpoint['target_config'])
	embedder.configure_target(target_config=target_config, target_vocab=checkpoint['target_nouns'][checkpoint['num_invalid_target_nouns']:])
	return target_config

# Load the appropriate guide targets
def load_guide_targets(guide_targets: tuple[str, ...], embedder: embedders.Embedder, device: torch.device, device_is_cpu: bool) -> torch.Tensor:

	log.info(f"Loading and preprocessing {len(guide_targets)} guide target nouns...")

	assert isinstance(guide_targets, tuple) and guide_targets and all(isinstance(tgt, str) for tgt in guide_targets)
	if len(set(guide_targets)) != len(guide_targets):
		raise ValueError("Guide target nouns contain duplicates")

	guide_token_ids = torch.full(size=(len(guide_targets), embedder.target_config.token_length), fill_value=embedder.target_config.pad_token_id, dtype=embedder.target_config.token_dtype, pin_memory=not device_is_cpu)
	for i in range(0, len(guide_targets), embedder.tokenizer_batch_size):
		token_ids = embedder.tokenize_target(guide_targets[i:i + embedder.tokenizer_batch_size])[0]
		if token_ids.shape[1] > guide_token_ids.shape[1]:
			raise ValueError("Some guide target noun(s) have tokenizations that are longer than supported by the model target configuration")
		guide_token_ids[i:i + embedder.tokenizer_batch_size, :token_ids.shape[1]] = token_ids
	guide_valid = torch.all(guide_token_ids >= 0, dim=1)
	guide_targets_invalid = tuple(tgt for tgt, valid in zip(guide_targets, guide_valid.tolist()) if not valid)
	if guide_targets_invalid:
		log.warning(f"Ignoring {len(guide_targets_invalid)} guide target nouns that are not encodable with the model target configuration: {guide_targets_invalid}")
		guide_token_ids = guide_token_ids[guide_valid, :]

	if not device_is_cpu:
		guide_token_ids = guide_token_ids.to(device=device, non_blocking=True)

	return guide_token_ids

# Load the embedding decoder model
def load_decoder_model(cfg: Any, embedder: embedders.Embedder, data_config: embedding_dataset.DataConfig, checkpoint: Optional[dict[str, Any]]) -> embedding_decoder.EmbeddingDecoder:
	# cfg must support attribute-wise access

	model_class: Type[embedding_decoder.EmbeddingDecoder] = getattr(embedding_decoder, cfg.model)
	log.info(f"Creating model of class {model_class.__qualname__}...")

	assert embedder.target_config is not None  # If this triggers then you forgot to call embedder.configure_target() prior to calling the current function

	model_kwargs = dict(
		embedder=embedder,
		data_config=data_config,
		vocab_quant=cfg.vocab_quant,
		num_end_loss=cfg.num_end_loss,
		label_smoothing=cfg.label_smoothing,
		hidden_dim=cfg.hidden_dim,
		feedfwd_scale=cfg.feedfwd_scale,
		mlp_hidden_layer=cfg.mlp_hidden_layer,
		mlp_hidden_bias=cfg.mlp_hidden_bias,
		mlp_hidden_norm=cfg.mlp_hidden_norm,
		mlp_hidden_activation=cfg.mlp_hidden_activation,
		input_dropout=cfg.input_dropout,
		num_layers=cfg.num_layers,
		num_heads=cfg.num_heads,
		layer_dropout=cfg.layer_dropout,
		layer_activation=cfg.layer_activation,
		layer_norm_first=cfg.layer_norm_first,
		layer_bias=cfg.layer_bias,
		logits_bias=cfg.logits_bias,
		init_bias_zero=cfg.init_bias_zero,
		init_mlp_mode=cfg.init_mlp_mode,
		init_mlp_unit_norm=cfg.init_mlp_unit_norm,
		init_tfrm_mode=cfg.init_tfrm_mode,
		init_tfrm_unit_norm=cfg.init_tfrm_unit_norm,
		init_tfrm_unit_postnorm=cfg.init_tfrm_unit_postnorm,
		init_tfrm_proj_layers=cfg.init_tfrm_proj_layers,
		init_zero_norm=cfg.init_zero_norm,
		init_rezero_mode=cfg.init_rezero_mode,
	)

	if model_class is embedding_decoder.PrefixedIterDecoder:
		model_kwargs.update(
			mlp_seq_len=cfg.mlp_seq_len,
			weight_tying=cfg.weight_tying,
			strictly_causal=cfg.strictly_causal,
			enable_nested=cfg.enable_nested,
		)
	elif model_class is embedding_decoder.DudDecoder:
		model_kwargs.update(
			mlp_seq_len=cfg.mlp_seq_len,
		)
	else:
		raise ValueError(f"Unrecognised model class: {model_class.__qualname__}")

	model = model_class(**model_kwargs)
	log.info(f"Created model of class {type(model).__qualname__}")

	total_param_count, param_counts = model.get_num_params()
	log.info("Model parameter counts by part:")
	for model_part, count in itertools.chain(param_counts.items(), (('Total', total_param_count),)):
		log.info(f"  {model_part} = {count.to_str()}")

	if checkpoint is not None:
		log.info(f"Loading {len(checkpoint['model_state_dict'])} items from model state dict...")
		model.load_state_dict(checkpoint['model_state_dict'], strict=True)

	return model

#
# Run
#

# Main function
def main():

	parser = argparse.ArgumentParser(description="Inference a NOVIC model checkpoint on given image(s).")

	parser.add_argument('--checkpoint', type=str, required=True, metavar='CKPT', help="Model checkpoint to load (e.g. outputs/ovod_20240628_142131/ovod_chunk0433_20240630_235415.train)")
	parser.add_argument('--image_dir', type=str, default=None, metavar='DIR', help="Directory relative to which to resolve relative input image paths (default: current directory)")
	parser.add_argument('--images', type=str, nargs='+', required=True, metavar='PATH', help="Input image paths (relative paths are resolved with respect to --image_dir)")

	parser.add_argument('--gencfg', type=str, default='beam_k10_vnone_gp_t1_a0', metavar='GENCFG', help="Generation configuration to use (default: %(default)s)")
	parser.add_argument('--guide_targets', type=str, nargs='+', default=None, metavar='NOUN', help="Manual list of target nouns to use for guided decoding, if guided decoding is enabled in the generation configuration (default: model vocabulary)")
	parser.add_argument('--guide_targets_file', type=str, default=None, metavar='PATH', help="Manual list file of target nouns to use for guided decoding, if guided decoding is enabled in the generation configuration (default: model vocabulary)")
	parser.add_argument('--torch_compile', action='store_true', help="Whether to torch-compile the decoder")
	parser.add_argument('--batch_size', type=int, default=128, metavar='NUM', help="Batch size to use for inference (default: %(default)s, 0 = No batch size limit)")
	parser.add_argument('--device', type=str, default='cuda', metavar='DEV', help="Torch device to use for inference (default: %(default)s)")
	parser.add_argument('--no_tf32', dest='tf32', action='store_false', help="Do not allow TF32")

	args = parser.parse_args()

	utils.allow_tf32(enable=args.tf32)
	utils.set_determinism(deterministic=False, seed=1, cudnn_benchmark_mode=False)  # Determinism not being required is also assumed in the NOVICModel constructor

	if args.guide_targets is not None and args.guide_targets_file is not None:
		parser.error("Cannot specify both --guide_targets and --guide_targets_file")
	elif args.guide_targets_file is not None:
		guide_targets = args.guide_targets_file
	else:
		guide_targets = args.guide_targets

	model = NOVICModel(
		checkpoint=args.checkpoint,
		gencfg=args.gencfg,
		guide_targets=guide_targets,
		torch_compile=args.torch_compile,
		batch_size=args.batch_size,
		device=args.device,
		cfg_flat_override=None,
		embedder_override=None,
	)

	image_batches = model.load_image_batches(image_paths=args.images, image_dir=args.image_dir)
	log.info(f"Loaded {len(args.images)} images as {len(image_batches)} batch(es) of max size {args.batch_size}")

	with model:
		pred_summaries = []
		for image_batch in image_batches:
			output = model.classify_images(images=image_batch)
			# TODO: Show probs instead of logprobs once you have verified they are identical
			pred_summaries.extend(' / '.join(f'{GenerationTask.COLOR_MAP[typ.value]}{pred}\033[0m = {logprob:.3g}' for pred, logprob, typ in itertools.islice(zip(sample_preds, sample_logprobs, sample_types, strict=True), 3)) for sample_preds, sample_logprobs, sample_types in zip(output.preds, output.logprobs, output.types, strict=True))
		for image_path, pred_summary in zip(args.images, pred_summaries, strict=True):
			log.info(f"{image_path} --> {pred_summary}")

	log.info(f"Finished inferencing model checkpoint on {len(args.images)} images in {len(image_batches)} batches: {model.checkpoint_tail}")

# Run main function
if __name__ == '__main__':
	main()
# EOF
