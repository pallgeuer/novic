#!/usr/bin/env python3
# Inference a NOVIC model

# Imports
from __future__ import annotations
import os
import re
import argparse
import itertools
import contextlib
import dataclasses
from typing import Iterable, Union, Optional, Any, ContextManager, Callable
import PIL.Image
import torch
import torch.utils.data
from logger import log
import utils
import embedders
import embedding_dataset

#
# NOVIC model
#

# NOVIC model class
class NOVICModel:

	def __init__(
		self,
		*,
		embedder_spec: str,                                     # Specification of the embedder model to use
		embedder_kwargs: Optional[dict[str, Any]] = None,       # Keyword arguments to pass to embedders.Embedder.create()
		checkpoint: str,                                        # Decoder checkpoint to load
		gencfg: str = 'beam_k10_vnone_gp_t1_a0',                # Generation configuration to use
		guide_targets: Union[Iterable[str], str, None] = None,  # If guided decoding is enabled, a manual list (iterable or path to text file with one per line) of target nouns to use for guided decoding (None = Use model vocabulary)
		amp: bool = False,                                      # Whether to enable AMP for the decoder
		amp_bf16: bool = True,                                  # Whether to use bfloat16 as the decoder AMP data type (if AMP is enabled)
		batch_size: int = 128,                                  # Nominal batch size
		device: Union[torch.device, str, int] = 'cuda',         # Torch device to use
	):

		self.batch_size = batch_size
		self.device, self.device_is_cpu, self.device_is_cuda = load_device(device=device)
		log.info(f"Using torch device: {self.device}")

		embedder_create_kwargs = dict(
			spec=embedder_spec,
			tokenizer_batch_size=self.batch_size,
			inference_batch_size=self.batch_size,
			image_batch_size=self.batch_size,
			load_model=False,
			device=self.device,
		)
		if embedder_kwargs is not None:
			embedder_create_kwargs.update(embedder_kwargs)
		log.info(f"Creating embedder of specification {embedder_create_kwargs['spec']}{' with checking' if embedder_create_kwargs.get('check', False) else ''}...")
		self.embedder = embedders.Embedder.create(**embedder_create_kwargs)
		log.info(f"Created embedder of class type {type(self.embedder).__qualname__}")

		self.checkpoint = os.path.abspath(checkpoint)
		self.checkpoint_tail = os.path.join(os.path.basename(os.path.dirname(self.checkpoint)), os.path.basename(self.checkpoint))
		log.info(f"Using decoder checkpoint: {self.checkpoint}")

		self.gencfg = GenerationConfig.from_name(name=gencfg)
		log.info(f"Using generation config: {self.gencfg.name}")

		if guide_targets is None:
			self.guide_targets = None
		elif isinstance(guide_targets, str):
			with open(guide_targets, 'r') as file:
				self.guide_targets = tuple(stripped_line for line in file if (stripped_line := line.strip()))
		else:
			self.guide_targets = tuple(guide_targets)

		self.amp_context, self.amp_dtype = load_decoder_amp(enabled=amp, bf16=amp_bf16, determ=False, device=self.device)  # Note: Assumes determinism is not required
		self.data_config = embedding_dataset.DataConfig.create(data_config_dict=dict(use_weights=False, multi_target=False), use_targets=True)

		self.__stack = contextlib.ExitStack()

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

	def get_image_transform(self) -> Callable[[PIL.Image.Image], torch.Tensor]:  # Returns the required image transform (for preprocessing images to tensor)
		return self.embedder.get_image_transform()

	def transform_images(self, images: Union[Iterable[PIL.Image.Image], PIL.Image.Image]) -> torch.Tensor:
		# images = Image(s) to transform/preprocess to tensor form
		# Returns the preprocessed image(s) tensor (Bx3xHxW)
		if isinstance(images, PIL.Image.Image):
			images = (images,)
		image_transform = self.get_image_transform()
		return torch.utils.data.default_collate(tuple(image_transform(image) for image in images))

	def __enter__(self) -> NOVICModel:
		# Ensure the NOVIC model is loaded and ready for inference
		with self.__stack as stack:
			stack.enter_context(self.embedder.inference_model())
			self.__stack = stack.pop_all()
		assert self.__stack is not stack
		return self

	def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
		# Unload the NOVIC model if it is currently loaded
		return self.__stack.__exit__(exc_type, exc_val, exc_tb)

	def embed_images(self, images: Union[torch.Tensor, Iterable[PIL.Image.Image], PIL.Image.Image]) -> torch.Tensor:
		# images = Image(s) to embed (PIL image(s) are first transformed/preprocessed, tensors are Bx3xHxW)
		# Returns the image embeddings (BxF)
		if not isinstance(images, torch.Tensor):
			images = self.transform_images(images=images)
		with self.embedder.inference_mode():
			return self.embedder.inference_image(images=images)

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

#
# Helper functions
#

# Load the torch device
def load_device(device: Union[torch.device, str, int]) -> tuple[torch.device, bool, bool]:
	device = torch.device(device)
	if device.type == 'cuda' and not torch.cuda.is_available():
		log.warning("No CUDA device is available => Running on CPU instead")
		device = torch.device('cpu')
	device = torch.empty(size=(), device=device).device  # Note: This is required to ensure a resolved device index, as device(type='cuda') != device(type='cuda', index=0) even if there is only one CUDA device and tensor.device always has a resolved index
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

#
# Run
#

# Main function
def main():

	parser = argparse.ArgumentParser(description="Inference a NOVIC model checkpoint on given image(s).")

	parser.add_argument('--embedder_spec', type=str, required=True, metavar='SPEC', help="Specification of the embedder model to use (e.g. openclip:apple/DFN5B-CLIP-ViT-H-14-378)")
	parser.add_argument('--checkpoint', type=str, required=True, metavar='CKPT', help="Model checkpoint to load (e.g. outputs/ovod_20240628_142131/ovod_chunk0433_20240630_235415.train)")
	parser.add_argument('--images', type=str, nargs='+', required=True, metavar='PATH', help="Input image paths (relative paths are resolved with respect to --image_dir)")
	parser.add_argument('--image_dir', type=str, default=None, metavar='DIR', help="Directory relative to which to resolve relative input image paths (default: current directory)")

	parser.add_argument('--no_tf32', dest='tf32', action='store_false', help="Do not allow TF32")
	parser.add_argument('--gencfg', type=str, default='beam_k10_vnone_gp_t1_a0', metavar='GENCFG', help="Generation configuration to use (default: %(default)s)")
	parser.add_argument('--guide_targets', type=str, nargs='+', default=None, metavar='NOUN', help="Manual list of target nouns to use for guided decoding, if guided decoding is enabled in the generation configuration (default: model vocabulary)")
	parser.add_argument('--guide_targets_file', type=str, default=None, metavar='PATH', help="Manual list file of target nouns to use for guided decoding, if guided decoding is enabled in the generation configuration (default: model vocabulary)")
	parser.add_argument('--amp', action='store_true', help="Whether to enable AMP for the decoder")
	parser.add_argument('--no_amp_bf16', dest='amp_bf16', action='store_false', help="Do not use bfloat16 as the decoder AMP data type (if AMP is enabled)")
	parser.add_argument('--batch_size', type=int, default=128, metavar='NUM', help="Batch size to use for inference (default: %(default)s, 0 = No batch size limit)")
	parser.add_argument('--device', type=str, default='cuda', metavar='DEV', help="Torch device to use for inference (default: %(default)s)")

	args = parser.parse_args()

	utils.allow_tf32(enable=args.tf32)
	utils.set_determinism(deterministic=False, seed=1, cudnn_benchmark_mode=False)  # Note: Determinism not being required is also assumed in the NOVICModel constructor

	if args.guide_targets is not None and args.guide_targets_file is not None:
		parser.error("Cannot specify both --guide_targets and --guide_targets_file")
	elif args.guide_targets_file is not None:
		guide_targets = args.guide_targets_file
	else:
		guide_targets = args.guide_targets

	model = NOVICModel(
		embedder_spec=args.embedder_spec,
		embedder_kwargs=None,
		checkpoint=args.checkpoint,
		gencfg=args.gencfg,
		guide_targets=guide_targets,
		amp=args.amp,
		amp_bf16=args.amp_bf16,
		batch_size=args.batch_size,
		device=args.device,
	)

	image_batches = model.load_image_batches(image_paths=args.images, image_dir=args.image_dir)
	log.info(f"Loaded {len(args.images)} images as {len(image_batches)} batch(es) of max size {args.batch_size}")

# Run main function
if __name__ == '__main__':
	main()
# EOF
