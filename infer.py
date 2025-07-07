#!/usr/bin/env python3
# Inference a NOVIC model

# Imports
from __future__ import annotations
import os
import re
import argparse
import itertools
import dataclasses
from typing import Iterable, Union, Optional, Any
import PIL.Image
import torch
from logger import log
import utils

#
# NOVIC model
#

# NOVIC model class
class NOVICModel:

	@classmethod
	def load_images(
		cls,
		images: Iterable[Union[PIL.Image.Image, str]],  # Image paths (relative paths are resolved with respect to image_dir)
		*,
		image_dir: Optional[str] = None,                # Directory relative to which to resolve relative image paths (None = Current directory)
		batch_size: int = 64,                           # Batch size (0 = No batch size limit)
	) -> list[list[PIL.Image.Image]]:                   # Returns a list of batches of PIL images (the last batch may be smaller than the nominal batch size)
		raise NotImplementedError

	def __init__(
		self,
		*,
		embedder_spec: str,                                # Specification of the embedder model to use
		embedder_kwargs: Optional[dict[str, Any]] = None,  # Keyword arguments to pass to embedders.Embedder.create()
		checkpoint: str,                                   # Decoder checkpoint to load
		gencfg: str,                                       # Generation configuration to use
		device: Union[torch.device, str, int] = 'cuda',    # Torch device to use
	):

		self.checkpoint = os.path.abspath(checkpoint)
		self.checkpoint_tail = os.path.join(os.path.basename(os.path.dirname(self.checkpoint)), os.path.basename(self.checkpoint))
		log.info(f"Using decoder checkpoint: {self.checkpoint}")

		self.gencfg = GenerationConfig.from_name(name=gencfg)
		log.info(f"Using generation config: {self.gencfg.name}")

		self.device, self.device_is_cpu, self.device_is_cuda = load_device(device=device)
		log.info(f"Using torch device: {self.device}")

	def __call__(self):
		raise NotImplementedError

	def classify(self):
		raise NotImplementedError

	def classify_batches(self):
		raise NotImplementedError

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

#
# Run
#

# Main function
def main():

	parser = argparse.ArgumentParser(description="Inference a NOVIC model checkpoint on given image(s).")
	parser.add_argument('--image_dir', type=str, default=None, metavar='DIR', help="Directory relative to which to resolve relative input image paths (default: current directory)")
	parser.add_argument('--images', type=str, nargs='+', required=True, metavar='PATH', help="Input image paths (relative paths are resolved with respect to --image_dir)")
	parser.add_argument('--load_model', type=str, required=True, metavar='CKPT', help="Model checkpoint to load (e.g. outputs/ovod_20240628_142131/ovod_chunk0433_20240630_235415.train)")
	parser.add_argument('--embedder_spec', type=str, required=True, metavar='SPEC', help="Specification of the embedder model to use (e.g. openclip:apple/DFN5B-CLIP-ViT-H-14-378)")
	parser.add_argument('--gencfg', type=str, default='beam_k10_vnone_gp_t1_a0', metavar='GENCFG', help="Generation configuration to use (default: %(default)s)")
	parser.add_argument('--batch_size', type=int, default=64, metavar='NUM', help="Batch size to use for inference (default: %(default)s, 0 = No batch size limit)")
	parser.add_argument('--device', type=str, default='cuda', metavar='DEV', help="Torch device to use for inference (default: %(default)s)")
	parser.add_argument('--tf32', dest='tf32', action='store_true', help="Allow TF32 (default)")
	parser.add_argument('--no_tf32', dest='tf32', action='store_false', help="Do not allow TF32")
	args = parser.parse_args()

	utils.allow_tf32(enable=args.tf32)
	utils.set_determinism(deterministic=False, seed=1, cudnn_benchmark_mode=False)

	image_batches = NOVICModel.load_images(images=args.images, image_dir=args.image_dir, batch_size=args.batch_size)
	log.info(f"Loaded {len(args.images)} images as {len(image_batches)} batch(es) of size up to {args.batch_size}")

	model = NOVICModel(embedder_spec=args.embedder_spec, checkpoint=args.load_model, gencfg=args.gencfg, device=args.device)

# Run main function
if __name__ == '__main__':
	main()
# EOF
