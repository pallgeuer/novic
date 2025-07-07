#!/usr/bin/env python3
# Train/evaluate/inference an embedding decoder based on the idea of embedder model inversion

# Imports
from __future__ import annotations
import os
import gc
import io
import re
import sys
import copy
import time
import math
import json
import random
import shutil
import logging
import fnmatch
import datetime
import operator
import itertools
import contextlib
import collections
import dataclasses
import urllib.parse
from typing import Optional, Iterable, Iterator, Sequence, ContextManager, Type, Union, Any, Counter
import requests
import tabulate
import tqdm
import wandb
import hydra
import hydra.core.hydra_config
import omegaconf
import PIL.Image
import torch
import torch.utils.data
import timm.optim
import logger
from logger import log
import utils
import utils_config
import classification_dataset
import embedders
import embedding_dataset
import noun_dataset
import embedding_cache
import embedding_cache_writers
import embedding_decoder
import embedding_noise
import infer

# Environment variables
os.environ['TOKENIZERS_PARALLELISM'] = 'true'  # We do not use tokenizers inside the forked dataloader, so it is safe to enable tokenizer parallelism for positive speed benefit (Note: The open_clip library sets this to false during the imports above)
os.environ['NUMEXPR_MAX_THREADS'] = '8'        # Set NumExpr threads to the value that is selected by default if no value is set

# Configure tabulate
tabulate.PRESERVE_WHITESPACE = True

# Constants
MODEL_CFGS = {'vocab_quant', 'hidden_dim', 'feedfwd_scale', 'mlp_hidden_layer', 'mlp_hidden_bias', 'mlp_hidden_norm', 'mlp_hidden_activation', 'num_layers', 'num_heads', 'layer_activation', 'layer_norm_first', 'layer_bias', 'logits_bias', 'mlp_seq_len', 'weight_tying', 'strictly_causal', 'enable_nested', 'cross_encoder', 'num_encoder_layers'}  # All of these cfg.* configs MUST satisfy that they are ONLY ever referenced with the load_decoder_model() method, and can under no reasonable circumstances have a different value from training to evaluation
IGNORE_CFG_DIFFS = {*MODEL_CFGS, 'action', 'wandb', 'load_model', 'load_train_state', 'load_lr_state', 'load_models', 'load_models_dirnum', 'eval_train_top1', 'eval_cls_datasets', 'eval_train', 'infer_texts', 'infer_images', 'infer_image_dir'}
IMAGE_DIR_TAG = '$IMAGEDIR'
SOURCE_TAG = '$SOURCE'
WIKIMEDIA_USER_AGENT = {'User-Agent': 'OvodBot/1.0 (https://github.com/ky-ah/ovod)'}

#
# Main
#

# Main function
@hydra.main(config_path="config", config_name="train", version_base=None)
def main(cfg: omegaconf.DictConfig):

	hydra_config = hydra.core.hydra_config.HydraConfig.get()
	hydra_dir = hydra_config.runtime.output_dir
	file_handler = logging.FileHandler(filename=f'{hydra_dir}/{hydra_config.job.name}.log')  # Note: The default job name is the name of the launched Python file (without file extension)
	file_handler.setFormatter(logger.formatter)
	log.addHandler(file_handler)

	log.info(f"Hydra output dir: {hydra_dir}")
	log.info(f"Run configuration:\n{omegaconf.OmegaConf.to_yaml(cfg).strip()}")

	utils.allow_tf32(enable=cfg.allow_tf32)
	utils.set_determinism(deterministic=cfg.determ, seed=cfg.determ_seed, cudnn_benchmark_mode=cfg.cudnn_bench)

	wandb_log_dir = os.environ.get('WANDB_DIR') or os.path.join(os.path.dirname(os.path.realpath(__file__)), 'log')
	with contextlib.suppress(OSError):
		os.mkdir(wandb_log_dir)

	wandb_tags = cfg.wandb_tags
	if wandb_tags:
		wandb_tags = sorted({tag.strip() for tag in wandb_tags.split(',') if tag.strip()})
	if not wandb_tags:
		wandb_tags = None

	use_wandb = cfg.wandb and cfg.action in ('train', 'eval', 'eval_cls', 'eval_cls_decoding', 'infer')  # Note: This is the master select of which actions use wandb
	with wandb.init(
		mode='online' if use_wandb else 'disabled',
		project=cfg.wandb_project,
		entity=cfg.wandb_entity,
		group=cfg.wandb_group,
		job_type=cfg.wandb_job_type or cfg.action,
		name=cfg.wandb_name,
		tags=wandb_tags,
		dir=wandb_log_dir,
		config=utils_config.wandb_from_omegaconf(cfg, hydra_dir=hydra_dir, hydra_name=os.path.basename(hydra_dir)),
	):

		if use_wandb:
			log.info(f"Wandb run: {wandb.run.name} ({wandb.run.url})")
			log.info(f"Wandb run path: {wandb.run._settings.sync_dir}")  # noqa
			utils_config.print_wandb_config(newline=False)

		if cfg.action == 'test_data_loader':
			action_test_data_loader(cfg=cfg)
		elif cfg.action == 'test_embed_cache':
			action_test_embed_cache(cfg=cfg)
		elif cfg.action == 'embedder_zero_shot':
			action_embedder_zero_shot(cfg=cfg, hydra_dir=hydra_dir)
		elif cfg.action == 'cache_noun_dataset':
			action_cache_noun_dataset(cfg=cfg)
		elif cfg.action == 'convert_noun_dataset':
			action_convert_noun_dataset(cfg=cfg)
		elif cfg.action == 'cache_noun_multiset':
			action_cache_noun_multiset(cfg=cfg)
		elif cfg.action == 'cache_captions':
			action_cache_captions(cfg=cfg)
		elif cfg.action == 'cache_cls':
			action_cache_cls(cfg=cfg)
		elif cfg.action == 'cache_images':
			action_cache_images(cfg=cfg)
		elif cfg.action == 'merge_caches':
			action_merge_caches(cfg=cfg)
		elif cfg.action == 'train':
			action_train(cfg=cfg, hydra_dir=hydra_dir, use_wandb=use_wandb)
		elif cfg.action == 'fix_checkpoints':
			action_fix_checkpoints(cfg=cfg, hydra_dir=hydra_dir)
		elif cfg.action == 'eval':
			action_eval(cfg=cfg, hydra_dir=hydra_dir, use_wandb=use_wandb)
		elif cfg.action == 'eval_cls':
			action_eval_cls(cfg=cfg, hydra_dir=hydra_dir, use_wandb=use_wandb)
		elif cfg.action == 'eval_cls_decoding':
			action_eval_cls_decoding(cfg=cfg, hydra_dir=hydra_dir, use_wandb=use_wandb)
		elif cfg.action == 'infer':
			action_infer(cfg=cfg, hydra_dir=hydra_dir, use_wandb=use_wandb)
		elif cfg.action == 'format_preds':
			action_format_preds(cfg=cfg)
		elif cfg.action == 'format_wandb':
			action_format_wandb(cfg=cfg)
		elif cfg.action == 'collect_wiki_images':
			action_collect_wiki_images(cfg=cfg, hydra_dir=hydra_dir)
		elif cfg.action == 'sample_images':
			action_sample_images(cfg=cfg, hydra_dir=hydra_dir)
		else:
			raise ValueError(f"Unsupported action: {cfg.action}")

#
# Common action classes
#

# Model information class
@dataclasses.dataclass(frozen=True)
class ModelInfo:
	model_path: str = dataclasses.field(compare=False)  # Note: The same model on different installations/PCs can have different absolute paths
	model_spec: str
	model_dir: str
	model_name: str

# Generation task class
@dataclasses.dataclass(eq=False)
class GenerationTask:

	COLOR_MAP = ('\033[92m', '\033[35m', '\033[33m', '\033[91m')           # ANSI colors corresponding to the 'result' field values

	gencfg: infer.GenerationConfig                                         # Generation configuration
	precompute: Optional[Any] = None                                       # Any precomputed task data if required
	target: Optional[torch.Tensor] = None                                  # BxKxC predicted token IDs tensor
	target_padding: Optional[torch.Tensor] = None                          # BxKxC predicted token padding tensor
	target_score: Optional[Union[torch.Tensor, list[list[float]]]] = None  # BxK prediction scores tensor/list-of-lists
	target_str: Optional[Sequence[Sequence[str]]] = None                   # BxK seq-of-seqs of predicted top-K noun strings
	invalid: Optional[torch.Tensor] = None                                 # BxK boolean tensor whether predicted top-K noun is invalid (not correct, not valid guide, not valid vocab)
	valid_vocab: Optional[torch.Tensor] = None                             # BxK boolean tensor whether predicted top-K noun is valid vocab
	valid_guide: Optional[torch.Tensor] = None                             # BxK boolean tensor whether predicted top-K noun is valid guide
	correct: Optional[torch.Tensor] = None                                 # BxK boolean tensor whether predicted top-K noun is correct
	result: Optional[torch.Tensor] = None                                  # BxK integer tensor (0 = If correct, 1 = Else if valid guide, 2 = Else if valid vocab, 3 = Otherwise invalid)
	topk_counts: torch.Tensor = dataclasses.field(init=False)              # Kx4 integer tensor of top-k counts per result type (need to be divided by num samples to get actual top-k)
	topk_invalid: Optional[torch.Tensor] = None                            # K tensor of top-k any invalid ratios
	topk_valid: Optional[torch.Tensor] = None                              # K tensor of top-k all valid ratios
	topk_vocab: Optional[torch.Tensor] = None                              # K tensor of top-k any valid vocab ratios
	topk_guide: Optional[torch.Tensor] = None                              # K tensor of top-k any valid guide ratios
	topk: Optional[torch.Tensor] = None                                    # K tensor of top-k any correct ratios

	def __post_init__(self):
		self.topk_counts = torch.zeros((self.gencfg.topk, 4), dtype=torch.int64)

# Generation task list class
class GenerationTaskList:

	def __init__(self, gencfgs: Sequence[infer.GenerationConfig], model: embedding_decoder.EmbeddingDecoder, vocab_targets_set: set[str], vocab_targets: Optional[torch.Tensor], guide_targets_set: set[str], guide_targets: Optional[torch.Tensor], class_lists: Optional[Sequence[Sequence[str]]] = None):
		self.gencfgs = gencfgs
		self.model = model
		self.vocab_targets_set = vocab_targets_set
		self.vocab_targets = vocab_targets
		self.guide_targets_set = guide_targets_set
		self.guide_targets = guide_targets
		self.class_lists = class_lists
		self.tasks = tuple(GenerationTask(gencfg=gencfg) for gencfg in self.gencfgs)
		self.precompute_cache = {}
		self.num_samples = 0

	def __len__(self) -> int:
		return len(self.tasks)

	def __getitem__(self, index: int) -> GenerationTask:
		return self.tasks[index]

	def __iter__(self) -> Iterator[GenerationTask]:
		return iter(self.tasks)

	def iter_generate(self, embeds: torch.Tensor, targets: Union[torch.Tensor, Sequence[int], None] = None) -> Iterator[tuple[int, GenerationTask]]:
		# Note: The 'targets' input should correpond to a 1D container of 0-indexed class indices
		# Careful: The yielding is intended for tqdm, and the tasks are NOT completed yet when they are yielded!

		self.num_samples += embeds.shape[0]
		if isinstance(targets, torch.Tensor):
			targets = targets.tolist()

		prev_task = None
		for i, task in enumerate(self.tasks, 1):
			yield i, task
			if task.gencfg.method == 'all' and task.precompute is None:
				task.precompute = model_precompute(model=self.model, gencfg=task.gencfg, vocab_targets=self.vocab_targets, guide_targets=self.guide_targets, precompute_cache=self.precompute_cache)
			task.target, task.target_padding, task.target_score = model_generate(model=self.model, embeds=embeds, gencfg=task.gencfg, vocab_targets=self.vocab_targets, guide_targets=self.guide_targets, precompute=task.precompute)
			if prev_task is not None:
				self._collect_task_results(task=prev_task, targets=targets)
			prev_task = task

		if prev_task is not None:
			self._collect_task_results(task=prev_task, targets=targets)

	def generate(self, embeds: torch.Tensor, targets: Union[torch.Tensor, Sequence[int], None] = None):
		for _ in self.iter_generate(embeds=embeds, targets=targets):
			pass

	def _collect_task_results(self, task: GenerationTask, targets: Optional[Sequence[int]]):
		task.target = task.target.cpu()
		task.target_padding = task.target_padding.cpu()
		task.target_score = task.target_score.tolist()
		task.target_str = self.model.embedder.detokenize_target(task.target)
		task.valid_vocab = torch.tensor(tuple(tuple(pred in self.vocab_targets_set for pred in preds) for preds in task.target_str), dtype=torch.bool)
		task.valid_guide = torch.tensor(tuple(tuple(pred in self.guide_targets_set for pred in preds) for preds in task.target_str), dtype=torch.bool)
		if targets is not None and self.class_lists is not None:
			task.correct = torch.tensor(tuple(tuple(pred in self.class_lists[target] for pred in preds) for target, preds in zip(targets, task.target_str)), dtype=torch.bool)
		else:
			task.correct = torch.zeros(size=task.target.shape[:-1], dtype=torch.bool)
		task.invalid = torch.logical_or(task.valid_vocab, task.valid_guide).logical_or_(task.correct).logical_not_()  # noqa
		task.result = torch.max((stacked_results := torch.stack(tensors=(task.correct, task.valid_guide, task.valid_vocab, torch.ones_like(task.invalid)), dim=2).cummax(dim=2)[0]), dim=2)[1]
		stacked_results[:, :, -1] = task.invalid
		task.topk_counts.add_(stacked_results.cummax(dim=1)[0].sum(dim=0))
		topk_counts = task.topk_counts.to(dtype=self.model.embedder.embed_dtype)
		task.topk_valid = (self.num_samples - topk_counts[:, 3]).div_(self.num_samples)
		topk_ratios = topk_counts.div_(self.num_samples)
		task.topk_invalid = topk_ratios[:, 3]
		task.topk_vocab = topk_ratios[:, 2]
		task.topk_guide = topk_ratios[:, 1]
		task.topk = topk_ratios[:, 0]

# Prediction scorer class
class PredictionScorer:

	CATEGORY_SCORES = {'correct_primary': 1.0, 'correct_secondary': 0.8, 'close_primary': 0.5, 'close_secondary': 0.4, 'incorrect': 0.0}

	missing_classes: dict[str, set[str]]
	missing_samples: set[str]
	multiple_categories: set[tuple[str, str, tuple[str, ...]]]

	def __init__(self, class_annotations: dict[str, dict[str, set[str]]], categories: dict[str, None]):
		# Note: All categories included in class_annotations must be in 'categories'
		self.class_annotations = class_annotations
		self.categories = tuple(sorted(categories.keys(), key=lambda category: (-self.CATEGORY_SCORES.get(category, 0.0), category)))
		if unscored_categories := set(self.categories).difference(self.CATEGORY_SCORES):
			log.warning(f"Applying zero score contribution for unrecognised categories: {sorted(unscored_categories)}")
		self.category_scores = {None: 0.0, **{category: self.CATEGORY_SCORES.get(category, 0.0) for category in self.categories}}
		self.reset()

	def reset(self):
		self.missing_classes = {sample: set() for sample in self.class_annotations.keys()}
		self.missing_samples = set()
		self.multiple_categories = set()

	def score(self, counter: Counter[Optional[str]]) -> float:
		return sum(count * self.category_scores[category] for category, count in counter.items())

	def categorise(self, predictions: dict[str, str]) -> tuple[float, Counter[Optional[str]]]:
		# Note: The returned counter has a total of exactly len(predictions)
		counter = collections.Counter()
		for sample, pred in predictions.items():
			if (annotation := self.class_annotations.get(sample, None)) is None:
				counter[None] += 1
				self.missing_samples.add(sample)
			else:
				if matched_category := [category for category, classes in annotation.items() if pred in classes]:
					if len(matched_category) > 1:
						matched_category.sort()
						self.multiple_categories.add((sample, pred, tuple(matched_category)))
					counter[matched_category[0]] += 1
				else:
					self.missing_classes[sample].add(pred)
					counter[None] += 1
		assert counter.total() == len(predictions) and counter.keys() <= self.category_scores.keys()
		return self.score(counter), counter

	def categorise_topk(self, predictions: dict[str, Sequence[str]], topk: int) -> tuple[tuple[float, ...], tuple[Counter[Optional[str]], ...]]:
		# Note: The topk argument must have a value such that every sequence of predictions has at least that many items
		# Note: All returned counters have a total of exactly len(predictions)
		counters = tuple(collections.Counter() for _ in range(topk))
		for sample, preds in predictions.items():
			if (annotation := self.class_annotations.get(sample, None)) is None:
				for counter in counters:
					counter[None] += 1
				self.missing_samples.add(sample)
			else:
				top_category, top_category_score = None, -math.inf
				for counter, pred in zip(counters, itertools.islice(preds, topk), strict=True):
					if matched_category := [category for category, classes in annotation.items() if pred in classes]:
						if len(matched_category) > 1:
							matched_category.sort()
							self.multiple_categories.add((sample, pred, tuple(matched_category)))
						pred_category = matched_category[0]
					else:
						self.missing_classes[sample].add(pred)
						pred_category = None
					pred_category_score = self.category_scores[pred_category]
					if pred_category_score < top_category_score:
						pred_category = top_category
					elif pred_category_score > top_category_score:
						top_category, top_category_score = pred_category, pred_category_score
					counter[pred_category] += 1
		assert all(counter.total() == len(predictions) and counter.keys() <= self.category_scores.keys() for counter in counters)
		return tuple(self.score(counter) for counter in counters), counters

	def finalise(self):
		for sample, missing in self.missing_classes.items():
			if missing:
				log.warning(f"Sample {sample} needs: {json.dumps(sorted(missing))}")
		if any(missing for missing in self.missing_classes.values()):
			log.warning(f"{sum(bool(missing) for missing in self.missing_classes.values())} samples need a total of {sum(len(missing) for missing in self.missing_classes.values())} extra annotations")
		if self.missing_samples:
			log.error(f"Class annotations is missing {len(self.missing_samples)} samples: {json.dumps(sorted(self.missing_samples))}")
		if self.multiple_categories:
			for sample, pred, matched_category in sorted(self.multiple_categories):
				log.error(f"Sample {sample} prediction '{pred}' matches multiple categories: {json.dumps(matched_category)}")

	@classmethod
	def format_counter(cls, counter: Counter[Optional[str]], width: Optional[int] = None) -> str:
		total = counter.total()
		if width is None:
			width = len(format(total, 'd'))
		correct_prim, correct_sec, close_prim, close_sec, incorrect = counter['correct_primary'], counter['correct_secondary'], counter['close_primary'], counter['close_secondary'], counter['incorrect']
		other = total - correct_prim - correct_sec - close_prim - close_sec - incorrect
		return f"{correct_prim:{width}d}/{correct_sec:{width}d}\u2713 {close_prim:{width}d}/{close_sec:{width}d}~ {incorrect:{width}d}/{other:{width}d}\u2717 = {total:{width}d}"

	@classmethod
	def format_score(cls, score: float, total: int, width: Optional[int] = None) -> str:
		if width is None:
			width = len(format(total, 'd'))
		return f"{score:{width + 2}.1f}"

	@classmethod
	def format_score_pct(cls, score: float, total: int) -> str:
		return format_ratio_str(score / total) if total != 0 else format_percent_str(math.nan)

#
# Action: Test data loader
#

# Action: Test data loader
def action_test_data_loader(cfg: omegaconf.DictConfig):

	device, device_is_cpu, device_is_cuda = load_device(cfg=cfg)
	embedder = load_embedder(cfg=cfg, device=device, check_consistent=cfg.test_consistent)
	dataset = load_embedding_dataset(cfg=cfg, embedder=embedder, check_consistent=cfg.test_consistent, check_print=cfg.test_print, training=cfg.test_training, strict_embedder=cfg.strict_embedder)
	gen_target_config(cfg=cfg, embedder=embedder, targets=dataset.targets, num_invalid_targets=dataset.num_invalid_targets)
	gen_data_config(cfg=cfg, dataset=dataset)
	loader, loader_info = load_embedding_dataset_loader(cfg=cfg, dataset=dataset, training=cfg.test_training, device=device, patch=cfg.test_patch and cfg.test_device)  # Note: For an uncached noun dataset, the strings will not be embedded/tokenized if cfg.test_device is False
	grad_accum = embedding_dataset.GradAccum(loader=loader, loader_info=loader_info, accum_size=cfg.accum_factor, drop_last=cfg.test_training) if cfg.test_training else None

	cpu_device = torch.device('cpu')
	embed_eps = torch.finfo(embedder.embed_dtype).eps
	expected_device = device if loader_info.on_device else cpu_device
	expected_batches = loader_info.epoch_batches if grad_accum is None else grad_accum.loader_batches
	expected_samples = loader_info.epoch_samples if grad_accum is None else grad_accum.loader_samples

	assert (loader_info.on_device or device != cpu_device)
	assert not (cfg.test_training and loader_info.incomplete_batch)
	assert loader_info.available_samples == dataset.num_embeds == (dataset.num_items if getattr(dataset, 'batch_size', None) is None else dataset.num_items * dataset.batch_size)
	assert (loader_info.on_device or not device_is_cpu) and loader_info.pin_memory == (not loader_info.on_device)
	assert 0 <= loader_info.batch_size_last < loader_info.batch_size and loader_info.incomplete_batch == (loader_info.batch_size_last != 0)
	assert loader_info.epoch_batches == loader_info.complete_batches + loader_info.incomplete_batch == len(loader)
	assert loader_info.epoch_samples == loader_info.complete_batches * loader_info.batch_size + loader_info.incomplete_batch * loader_info.batch_size_last
	assert (loader_info.available_samples == loader_info.epoch_samples) if loader_info.incomplete_batch else (loader_info.available_samples < loader_info.epoch_samples + loader_info.batch_size)

	if grad_accum is not None:
		assert not (grad_accum.drop_last and grad_accum.incomplete_step)
		assert grad_accum.batch_size == loader_info.batch_size and grad_accum.accum_batch_size == loader_info.batch_size * grad_accum.accum_size
		assert grad_accum.loader_steps == grad_accum.complete_steps + grad_accum.incomplete_step and grad_accum.loader_batches == grad_accum.complete_batches + grad_accum.incomplete_batches and grad_accum.loader_samples == grad_accum.complete_samples + grad_accum.incomplete_samples

	progress_bar = None
	with dataset.loaded(), contextlib.ExitStack() as stack:

		seen = set()
		total_batches = 0
		total_samples = 0
		manual_to_device = cfg.test_device and not loader_info.on_device  # Note: This will not embed/tokenize an unpatched uncached CPU dataset
		uncached_noun_dataset = isinstance(dataset, noun_dataset.NounDataset) and not dataset.use_cache
		if manual_to_device:
			log.warning(f"Manually managing {'embedding/tokenization and ' if uncached_noun_dataset else ''}moving tensors to device")

		accum_batches = 0
		accum_samples = 0
		accum_loss = torch.zeros(size=(), dtype=embedder.embed_dtype)
		unit_tensor = torch.ones(size=(), dtype=embedder.embed_dtype)
		complete_steps = complete_batches = complete_samples = 0
		loader_steps = loader_batches = loader_samples = 0

		start_time = time.perf_counter()
		for batch_id, data in enumerate(loader if grad_accum is None else grad_accum.loader(), 1):

			embed, target, mask, weight = data

			if isinstance(embed, torch.Tensor):
				assert embed.device == expected_device and embed.dtype == embedder.embed_dtype and embed.is_pinned() == loader_info.pin_memory
			else:
				assert loader_info.on_device == device_is_cpu and loader_info.pin_memory != device_is_cpu
			if isinstance(target, torch.Tensor):
				assert target.device == expected_device and target.dtype == embedder.target_config.token_dtype and target.is_pinned() == loader_info.pin_memory
			if isinstance(mask, torch.Tensor):
				assert mask.device == expected_device and mask.dtype == embedder.target_config.mask_dtype and mask.is_pinned() == loader_info.pin_memory
			if isinstance(weight, torch.Tensor):
				assert weight.device == expected_device and weight.dtype == embedder.embed_dtype and weight.is_pinned() == loader_info.pin_memory

			if manual_to_device:
				if uncached_noun_dataset:
					with embedder.inference_mode():
						embed = embedder.inference_text(text=embed)
						assert embed.device == device and not embed.is_pinned()
					if target is not None:
						target, mask = embedder.tokenize_target(text=target)
					if not device_is_cpu:
						if target is not None:
							assert target.device == cpu_device and not target.is_pinned()
							target = target.pin_memory().to(device=device, non_blocking=True)
						if mask is not None:
							assert mask.device == cpu_device and not mask.is_pinned()
							mask = mask.pin_memory().to(device=device, non_blocking=True)
						if weight is not None:
							weight = weight.to(device=device, non_blocking=True)
				elif not device_is_cpu:
					embed = embed.to(device=device, non_blocking=True)
					if target is not None:
						target = target.to(device=device, non_blocking=True)
					if mask is not None:
						mask = mask.to(device=device, non_blocking=True)
					if weight is not None:
						weight = weight.to(device=device, non_blocking=True)
				data = (embed, target, mask, weight)

			summary = ', '.join("{device}[{shape}]".format(device=elem.device.type.upper(), shape='\xD7'.join(str(s) for s in elem.shape)) if isinstance(elem, torch.Tensor) else 'None' if elem is None else type(elem).__qualname__ for elem in data)
			if summary not in seen:
				with tqdm.tqdm.external_write_mode():
					print(f"Batch {batch_id} = ({summary})")
				seen.add(summary)

			total_batches += 1
			batch_samples = len(embed)  # Note: Works for both BxF tensors and tuple[str, ...] to give B
			total_samples += batch_samples
			assert batch_samples == (loader_info.batch_size_last if loader_info.incomplete_batch and batch_id == loader_info.epoch_batches else loader_info.batch_size)

			if grad_accum is not None:
				mean_accum_batch_loss, optimizer_step = grad_accum.accum_loss(mean_batch_loss=unit_tensor, num_in_batch=batch_samples)
				accum_batches += 1
				accum_samples += batch_samples
				accum_loss += mean_accum_batch_loss
				if optimizer_step:
					assert (accum_loss - 1).abs() < 2 * embed_eps
					if grad_accum.incomplete_step and batch_id == loader_info.epoch_batches:
						assert 1 <= accum_batches <= grad_accum.accum_size and 1 <= accum_samples < grad_accum.accum_batch_size
						assert accum_batches == grad_accum.incomplete_batches and accum_samples == grad_accum.incomplete_samples
					else:
						assert accum_batches == grad_accum.accum_size and accum_samples == grad_accum.accum_batch_size
						complete_steps += 1
						complete_batches += accum_batches
						complete_samples += accum_samples
					loader_steps += 1
					loader_batches += accum_batches
					loader_samples += accum_samples
					accum_batches = 0
					accum_samples = 0
					accum_loss = torch.zeros(size=(), dtype=embedder.embed_dtype)

			if device_is_cuda:
				torch.cuda.synchronize(device)  # Note: In most real loops the results of some calculations will be required on CPU before iterating to the next batch, so we simulate at least that the GPU computations need to have finished

			if batch_id == 2:
				progress_bar = stack.enter_context(tqdm.tqdm(desc='Testing data loader', total=expected_samples, unit='embed', unit_scale=False, dynamic_ncols=True, smoothing=0.08, initial=total_samples))
			elif batch_id > 2:
				progress_bar.update(n=batch_samples)

		elapsed_time = time.perf_counter() - start_time

		assert total_batches == expected_batches == batch_id
		assert total_samples == expected_samples and (progress_bar is None or progress_bar.n == expected_samples)
		if grad_accum is not None:
			assert optimizer_step and accum_batches == 0 and accum_samples == 0 and accum_loss == 0
			assert complete_steps == grad_accum.complete_steps and complete_batches == grad_accum.complete_batches and complete_samples == grad_accum.complete_samples
			assert loader_steps == grad_accum.loader_steps and loader_batches == grad_accum.loader_batches and loader_samples == grad_accum.loader_samples

	log.info(f"Finished testing data loader{' with consistency check' if cfg.test_consistent else ''} in {elapsed_time:.1f}s")

	if device_is_cpu:
		assert not torch.cuda.is_initialized()  # Note: This catches situations for example where CPU tensors are unnecessarily pinned even though the target device is CPU

#
# Action: Test embedding cache
#

# Action: Test embedding cache
def action_test_embed_cache(cfg: omegaconf.DictConfig):

	torch.set_printoptions(precision=8, threshold=1000, linewidth=180, sci_mode=False)

	log.info("Testing writing and loading some embedding caches...")
	cache_path = '/tmp/ovod_test_embed_cache.bin'

	device, device_is_cpu, device_is_cuda = load_device(cfg=cfg)
	embedder = load_embedder(cfg=cfg, device=device)

	test_random_cache(cfg=cfg, embedder=embedder, cache_path=cache_path, num_embeds=1 << 16, device=device)

	target_nouns = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
	test_photo_cache(cfg=cfg, embedder=embedder, cache_path=cache_path, target_nouns=target_nouns)

	test_index_cache(cfg=cfg, embedder=embedder, cache_path=cache_path, target_nouns=tuple('abcdefghijklmnopqrstuvwxyz'))

	test_multi_cache(cfg=cfg, embedder=embedder, cache_path=cache_path, target_nouns=target_nouns, num_embeds=16, num_embed_targets=6, device=device)

	with contextlib.suppress(FileNotFoundError):
		os.remove(cache_path)
		log.info(f"Removed temporary cache file: {cache_path}")

# Test a random cache
def test_random_cache(cfg: omegaconf.DictConfig, embedder: embedders.Embedder, cache_path: str, num_embeds: int, device: torch.device):

	log.info("Testing random cache...")

	embedding_cache_writers.RandomCacheWriter(cache_path=cache_path, embedder=embedder, num_embed=num_embeds).generate()

	dataset = load_embedding_dataset(cfg=cfg, embedder=embedder, embed_dataset=cache_path, training=True)
	gen_data_config(cfg=cfg, dataset=dataset)
	loader, loader_info = load_embedding_dataset_loader(cfg=cfg, dataset=dataset, training=True, device=device)

	with dataset.loaded(), tqdm.tqdm(desc='Iterating random cache', total=loader_info.epoch_samples, unit='embed', unit_scale=False, dynamic_ncols=True, smoothing=0.08) as progress_bar:

		embed_eps = torch.finfo(embedder.embed_dtype).eps
		for embed, target, mask, weight in loader:

			assert embed.shape == (loader_info.batch_size, embedder.embed_dim) and embed.dtype == embedder.embed_dtype
			assert target is None and mask is None and weight is None
			assert torch.all((torch.linalg.vector_norm(embed, dim=1) - 1).abs() <= 4 * embed_eps)

			if progress_bar.n == 0:
				with tqdm.tqdm.external_write_mode():
					print(embed[:, :8])
			progress_bar.update(n=embed.shape[0])

		assert progress_bar.n == loader_info.epoch_samples == num_embeds - num_embeds % loader_info.batch_size

# Test a photo cache
def test_photo_cache(cfg: omegaconf.DictConfig, embedder: embedders.Embedder, cache_path: str, target_nouns: Sequence[str]):

	log.info("Testing photo prompt cache...")

	writer_target_config = gen_target_config(cfg=cfg, embedder=embedder, targets=target_nouns, num_invalid_targets=0)
	written_embeds, written_targets, written_masks = embedding_cache_writers.PhotoCacheWriter(cache_path=cache_path, embedder=embedder, target_nouns=target_nouns, debug=True).generate()
	print('Target nouns =', target_nouns)

	dataset = load_embedding_dataset(cfg=cfg, embedder=embedder, embed_dataset=cache_path, use_targets=True, training=False)
	assert dataset.targets[1:] == tuple(target_nouns)
	target_config = gen_target_config(cfg=cfg, embedder=embedder, targets=dataset.targets, num_invalid_targets=dataset.num_invalid_targets)
	assert target_config == writer_target_config
	data_config = gen_data_config(cfg=cfg, dataset=dataset, multi_target=False)
	loader, loader_info = load_embedding_dataset_loader(cfg=cfg, dataset=dataset, training=False, device=torch.device('cpu'))

	with dataset.loaded():

		read_embeds = torch.full(size=(loader_info.epoch_samples, embedder.embed_dim), fill_value=torch.nan, dtype=embedder.embed_dtype)
		read_targets = torch.full(size=(loader_info.epoch_samples, target_config.token_length), fill_value=target_config.pad_token_id, dtype=target_config.token_dtype)
		read_masks = torch.ones(size=(loader_info.epoch_samples, target_config.token_length), dtype=target_config.mask_dtype) if target_config.use_masks else None
		read_weights = torch.full(size=(loader_info.epoch_samples,), fill_value=torch.nan, dtype=embedder.embed_dtype) if data_config.use_weights else None

		count = 0
		for embed, target, mask, weight in loader:
			print(f"{'-' * 120}\n{embed[:, :8]}\n{target}\n{mask}\n{weight}")
			batch_size = embed.shape[0]
			new_count = count + batch_size
			read_embeds[count:new_count, :] = embed
			read_targets[count:new_count, :target.shape[1]] = target
			if mask is not None:
				read_masks[count:new_count, :mask.shape[1]] = mask
			if weight is not None:
				read_weights[count:new_count] = weight
			count = new_count

		assert read_weights is None or torch.all(read_weights == 1)

		read_reorder = read_embeds[:, 0].argsort()
		read_embeds = read_embeds[read_reorder, :]
		read_targets = read_targets[read_reorder, :]
		if read_masks is not None:
			read_masks = read_masks[read_reorder, :]

		written_reorder = written_embeds[:, 0].argsort()
		written_embeds = written_embeds[written_reorder, :]
		written_targets = written_targets[written_reorder, :]
		if written_masks is not None:
			written_masks = written_masks[written_reorder, :]

		# Note: These checks could erroneously fail if two embeddings have EXACTLY the same first element and happen to be in a different order initially
		assert read_embeds.dtype == written_embeds.dtype and torch.equal(read_embeds, written_embeds) and read_targets.dtype == written_targets.dtype and torch.equal(read_targets, written_targets)
		assert read_masks is written_masks or (read_masks is not None and written_masks is not None and read_masks.dtype == written_masks.dtype and torch.equal(read_masks, written_masks))

# Test cache indexing
def test_index_cache(cfg: omegaconf.DictConfig, embedder: embedders.Embedder, cache_path: str, target_nouns: Sequence[str]):

	log.info("Testing cache indexing...")

	writer_target_config = gen_target_config(cfg=cfg, embedder=embedder, targets=target_nouns, num_invalid_targets=0)
	embedding_cache_writers.IndexCacheWriter(cache_path=cache_path, embedder=embedder, target_nouns=target_nouns).generate()
	print('Target nouns =', target_nouns)
	target_nouns_set = set(target_nouns)

	def load_cache(training: bool):
		dataset_ = load_embedding_dataset(cfg=cfg, embedder=embedder, embed_dataset=cache_path, use_targets=True, training=training)
		assert dataset_.targets[1:] == tuple(target_nouns)
		target_config = gen_target_config(cfg=cfg, embedder=embedder, targets=dataset_.targets, num_invalid_targets=dataset_.num_invalid_targets)
		assert target_config == writer_target_config
		gen_data_config(cfg=cfg, dataset=dataset_, multi_target=False)
		loader_, loader_info_ = load_embedding_dataset_loader(cfg=cfg, dataset=dataset_, training=training, device=torch.device('cpu'))
		return dataset_, loader_, loader_info_

	dataset, loader, loader_info = load_cache(training=False)
	with dataset.loaded():
		print("EVAL:")
		for _ in range(3):
			eval_data = tuple(tuple(embedder.detokenize_target(token_ids=target)) for embed, target, mask, weight in loader)
			eval_data_flat = tuple(noun for nouns in eval_data for noun in nouns)
			print(eval_data)
			assert all(len(nouns) == loader_info.batch_size for nouns in eval_data[:-1])
			assert eval_data_flat == target_nouns

	dataset, loader, loader_info = load_cache(training=True)
	with dataset.loaded():
		print("TRAIN:")
		counter = collections.Counter()
		for _ in range(len(target_nouns)):
			train_data = tuple(tuple(embedder.detokenize_target(token_ids=target)) for embed, target, mask, weight in loader)
			train_data_flat = tuple(noun for nouns in train_data for noun in nouns)
			train_data_set = set(train_data_flat)
			counter.update(train_data_flat)
			print(train_data)
			assert all(len(nouns) == loader_info.batch_size for nouns in train_data)
			assert len(train_data_flat) == len(train_data_set) == loader_info.epoch_samples
			assert train_data_set.issubset(target_nouns_set)
		print(dict(sorted(counter.items())))

# Test a multi-target cache
def test_multi_cache(cfg: omegaconf.DictConfig, embedder: embedders.Embedder, cache_path: str, target_nouns: Sequence[str], num_embeds: int, num_embed_targets: int, device: torch.device):

	log.info("Testing multi-target cache...")

	writer_target_config = gen_target_config(cfg=cfg, embedder=embedder, targets=target_nouns, num_invalid_targets=0)
	embedding_cache_writers.TestMultiCacheWriter(cache_path=cache_path, embedder=embedder, num_embed=num_embeds, target_nouns=target_nouns, num_embed_targets=num_embed_targets, debug=True).generate()

	dataset = load_embedding_dataset(cfg=cfg, embedder=embedder, embed_dataset=cache_path, use_targets=True, training=True)
	target_config = gen_target_config(cfg=cfg, embedder=embedder, targets=dataset.targets, num_invalid_targets=dataset.num_invalid_targets)
	assert target_config == writer_target_config
	gen_data_config(cfg=cfg, dataset=dataset)
	loader, loader_info = load_embedding_dataset_loader(cfg=cfg, dataset=dataset, training=True, device=device)

	with dataset.loaded():
		for embed, target, mask, weight in loader:
			print('-' * 120)
			print("EMBED {shape}:".format(shape='\xD7'.join(str(dim) for dim in embed.shape)))
			print(embed[:, :8])
			print("TARGET {shape}:".format(shape='\xD7'.join(str(dim) for dim in target.shape)))
			print(target)
			if mask is not None:
				print("MASK {shape}:".format(shape='\xD7'.join(str(dim) for dim in mask.shape)))
				print(mask)
			if weight is not None:
				print("WEIGHT {shape}:".format(shape='\xD7'.join(str(dim) for dim in weight.shape)))
				print(weight)

#
# Action: Embedder zero-shot
#

# Action: Embedder zero-shot
def action_embedder_zero_shot(cfg: omegaconf.DictConfig, hydra_dir: str):

	device, device_is_cpu, device_is_cuda = load_device(cfg=cfg)
	embedder = load_embedder(cfg=cfg, device=device)

	with torch.inference_mode(), embedder.inference_model():

		dataset, loader, cls_variant, cls_clean = load_cls_dataset(cfg=cfg, embedder=embedder, device_is_cpu=device_is_cpu, variant='clip', clean=cfg.clip_clean)
		if cfg.clip_prompts:
			prompts = classification_dataset.load_image_dataset_prompts(name=dataset.cls_name, variant=cls_variant)
		else:
			prompts = tuple((prompt.template, prompt.need_article) for prompt in load_embedding_dataset(cfg=cfg, embedder=embedder, embed_dataset='NounDataset', use_targets=False, use_cache=False).singular_prompts)
			log.info(f"Using {len(prompts)} {dataset.cls_name} prompts from noun dataset singular prompts")

		log.info(f"Checking embedder zero-shot performance on the {dataset.cls_name} dataset {dataset.cls_split} split with {len(prompts)} {'CLIP' if cfg.clip_prompts else 'custom'} prompts")
		text_embeds, text_embeds_T = compute_text_embeddings(embedder=embedder, nouns=dataset.cls_classes, prompts=prompts)
		log.info(f"Preparing and running zero-shot image classification on the {dataset.cls_name} {dataset.cls_split} split...")

		num_samples = len(loader.dataset)  # noqa
		if (last_batch_size := num_samples % loader.batch_size) > 0:  # Model warmup
			with embedder.inference_mode():
				tuple(embedder.inference_image(torch.utils.data.default_collate(tuple(dataset[i][0] for i in range(last_batch_size)))).cpu() for _ in range(2))

		with contextlib.ExitStack() as stack:

			num_correct = 0
			num_total = 0
			num_total_logits = 0

			if measure_gap := cfg.measure_gap:
				histc_bins, histc_min, histc_max = 360, 0, 180
				all_angle_mean = torch.zeros(size=(), dtype=torch.float64, device=device)
				all_angle_m2 = torch.zeros(size=(), dtype=torch.float64, device=device)
				all_angle_histc = torch.empty(size=(0,), dtype=embedder.embed_dtype, device=device).histc(bins=histc_bins, min=histc_min, max=histc_max).to(dtype=torch.int64)
				angle_mean = torch.zeros(size=(), dtype=torch.float64, device=device)
				angle_m2 = torch.zeros(size=(), dtype=torch.float64, device=device)
				angle_histc = torch.empty(size=(0,), dtype=embedder.embed_dtype, device=device).histc(bins=histc_bins, min=histc_min, max=histc_max).to(dtype=torch.int64)
				mean_shift = torch.zeros(size=(1, embedder.embed_dim), dtype=torch.float64, device=device)

			for i, (images, target) in enumerate(loader):

				target = target.to(embedder.device, non_blocking=True)
				with embedder.inference_mode():
					image_embeds = embedder.inference_image(images)
				logits = image_embeds @ text_embeds_T
				num_total_logits += logits.numel()

				correct = logits.argmax(axis=1).eq(target)
				correct_sum = correct.sum()
				num_total += correct.shape[0]

				if measure_gap:
					all_angles = torch.rad2deg(logits.clamp(min=-1, max=1).acos())
					all_angle_delta = all_angles - all_angle_mean
					all_angle_mean.add_(all_angle_delta.sum(), alpha=1 / num_total_logits)
					all_angle_m2.add_(torch.sum(all_angle_delta * (all_angles - all_angle_mean)))
					all_angle_histc += all_angles.histc(bins=histc_bins, min=histc_min, max=histc_max).to(dtype=torch.int64)
					angles = torch.rad2deg(torch.gather(input=logits, dim=1, index=target.unsqueeze(dim=1)).squeeze(dim=1).clamp_(min=-1, max=1).acos())
					angle_delta = angles - angle_mean
					angle_mean.add_(angle_delta.sum(), alpha=1 / num_total)
					angle_m2.add_(torch.sum(angle_delta * (angles - angle_mean)))
					angle_histc += angles.histc(bins=histc_bins, min=histc_min, max=histc_max).to(dtype=torch.int64)
					mean_shift.add_(torch.sum((image_embeds - text_embeds[target, :]) - mean_shift, dim=0, keepdim=True), alpha=1 / num_total)

				num_correct += correct_sum.item()

				if i == 1:
					progress_bar = stack.enter_context(tqdm.tqdm(desc=f'Zero-shot on {dataset.cls_split} split', total=num_samples, unit='img', unit_scale=False, dynamic_ncols=True, smoothing=0.08, initial=num_total))
				elif i > 1:
					postfix = dict(top1=format(num_correct / num_total, '.2%'))
					if measure_gap:
						postfix.update(angle_mean=f"{angle_mean.item():.3f}/{all_angle_mean.item():.3f}", angle_std=f"{math.sqrt(angle_m2.item() / num_total):.3f}/{math.sqrt(all_angle_m2.item() / num_total_logits):.3f}")
					progress_bar.set_postfix(postfix, refresh=False)
					progress_bar.update(images.shape[0])

			if measure_gap:
				all_angle_mean = all_angle_mean.item()
				all_angle_m2 = all_angle_m2.item()
				all_angle_histc = all_angle_histc.tolist()
				angle_mean = angle_mean.item()
				angle_m2 = angle_m2.item()
				angle_histc = angle_histc.tolist()
				mean_shift = mean_shift.squeeze(dim=0).tolist()

		log.info(f"Embedder zero-shot performance on {dataset.cls_name} {dataset.cls_split} split over {num_total} samples is {num_correct / num_total:.2%}")
		assert num_total == num_samples and num_total_logits == num_samples * text_embeds.shape[0]

		if measure_gap:
			assert sum(angle_histc) == num_total and sum(all_angle_histc) == num_total_logits
			with open(os.path.join(hydra_dir, f"modality_gap_{safe_embedder_spec(cfg.embedder_spec)}.json"), 'w') as file:
				utils.json_dump(dict(
					cfg_embedder={name: getattr(cfg, name) for name in ('device', 'allow_tf32', 'embedder_spec', 'embedder_amp', 'embedder_amp_bf16', 'embedder_compile', 'embedder_optimum')},
					cfg_cls={name: getattr(cfg, name) for name in ('cls_dataset', 'cls_split', 'clip_prompts')},
					all_angle_mean=all_angle_mean,
					all_angle_std=math.sqrt(all_angle_m2 / num_total_logits),
					all_angle_histc=dict(min=histc_min, max=histc_max, bins=histc_bins, counts=all_angle_histc),
					angle_mean=angle_mean,
					angle_std=math.sqrt(angle_m2 / num_total),
					angle_histc=dict(min=histc_min, max=histc_max, bins=histc_bins, counts=angle_histc),
					mean_shift=mean_shift,
				), file, indent=2)

#
# Action: Cache noun dataset
#

# Action: Cache noun dataset
def action_cache_noun_dataset(cfg: omegaconf.DictConfig):

	device, device_is_cpu, device_is_cuda = load_device(cfg=cfg)
	embedder = load_embedder(cfg=cfg, device=device)
	dataset = load_embedding_dataset(cfg=cfg, embedder=embedder, embed_dataset='NounDataset', use_targets=True, use_cache=True)
	gen_target_config(cfg=cfg, embedder=embedder, targets=dataset.targets, num_invalid_targets=dataset.num_invalid_targets)
	gen_data_config(cfg=cfg, dataset=dataset)

	with dataset.loaded():
		pass

	log.info("Finished ensuring noun dataset is cached")

#
# Action: Convert noun dataset
#

# Action: Convert noun dataset
def action_convert_noun_dataset(cfg: omegaconf.DictConfig):

	cache_path = get_cache_save_path(cfg=cfg)

	device, device_is_cpu, device_is_cuda = load_device(cfg=cfg)
	embedder = load_embedder(cfg=cfg, device=device)
	dataset = load_embedding_dataset(cfg=cfg, embedder=embedder, embed_dataset='NounDataset', use_targets=True, use_cache=False)
	gen_target_config(cfg=cfg, embedder=embedder, targets=dataset.targets, num_invalid_targets=dataset.num_invalid_targets)

	embedding_cache_writers.NounDatasetCacheWriter(cache_path=cache_path, dataset=dataset).generate()

#
# Action: Cache noun multiset
#

# Action: Cache noun multiset
def action_cache_noun_multiset(cfg: omegaconf.DictConfig):

	cache_path = get_cache_save_path(cfg=cfg)

	device, device_is_cpu, device_is_cuda = load_device(cfg=cfg)
	embedder = load_embedder(cfg=cfg, device=device)
	dataset = load_embedding_dataset(cfg=cfg, embedder=embedder, embed_dataset='NounDataset', use_targets=True, use_cache=False)
	gen_target_config(cfg=cfg, embedder=embedder, targets=dataset.targets, num_invalid_targets=dataset.num_invalid_targets)

	embedding_cache_writers.NounMultisetCacheWriter(cache_path=cache_path, dataset=dataset, multi_target_freq=cfg.multi_target_freq).generate()

#
# Action: Cache captions
#

# Action: Cache captions
def action_cache_captions(cfg: omegaconf.DictConfig):

	cache_path = get_cache_save_path(cfg=cfg)
	captions_path = os.path.abspath(resolve_source_path(cfg.captions_path))

	device, device_is_cpu, device_is_cuda = load_device(cfg=cfg)
	embedder = load_embedder(cfg=cfg, device=device)
	dataset = load_embedding_dataset(cfg=cfg, embedder=embedder, embed_dataset='NounDataset', use_targets=True, use_cache=False)
	gen_target_config(cfg=cfg, embedder=embedder, targets=dataset.targets, num_invalid_targets=dataset.num_invalid_targets)

	embedding_cache_writers.CaptionsCacheWriter(
		cache_path=cache_path,
		captions_path=captions_path,
		dataset=dataset,
		template_multiplier=cfg.template_multiplier,
		sample_multiplier=cfg.sample_multiplier,
		print_approx=cfg.captions_print,
	).generate()

#
# Action: Cache classification dataset
#

# Action: Cache classification dataset
def action_cache_cls(cfg: omegaconf.DictConfig):

	cache_path = get_cache_save_path(cfg=cfg)

	device, device_is_cpu, device_is_cuda = load_device(cfg=cfg)
	embedder = load_embedder(cfg=cfg, device=device)

	with embedder.inference_model():
		dataset, loader, cls_variant, cls_clean = load_cls_dataset(cfg=cfg, embedder=embedder, device_is_cpu=device_is_cpu)
		cls_class_lists, targets = align_cls_classes(cfg=cfg, embedder=embedder, dataset=dataset)
		gen_target_config(cfg=cfg, embedder=embedder, targets=targets, num_invalid_targets=0)
		embedding_cache_writers.ClassificationCacheWriter(cache_path=cache_path, embedder=embedder, loader=loader, targets=targets, class_targets=cls_class_lists).generate()

#
# Action: Cache images
#

# Action: Cache images
def action_cache_images(cfg: omegaconf.DictConfig):

	cache_path = get_cache_save_path(cfg=cfg)

	device, device_is_cpu, device_is_cuda = load_device(cfg=cfg)
	embedder = load_embedder(cfg=cfg, device=device)

	embedding_cache_writers.ImageCacheWriter(cache_path=cache_path, embedder=embedder, images=cfg.images, num_workers=cfg.dataset_workers).generate()

#
# Action: Merge caches
#

# Action: Merge caches
def action_merge_caches(cfg: omegaconf.DictConfig):

	cache_path = get_cache_save_path(cfg=cfg)

	dataset_freq = collections.Counter(cfg.embedding_datasets)
	if dataset_freq.total() < 1:
		raise ValueError("Specify multiple input embedding caches to merge using embedding_datasets=\"LIST\"")

	device, device_is_cpu, device_is_cuda = load_device(cfg=cfg)
	embedder = load_embedder(cfg=cfg, device=device)

	if cfg.save_targets not in (None, False, True):
		raise ValueError("Config save_targets must be None, False or True")
	datasets = tuple(load_embedding_dataset(cfg=cfg, embedder=embedder, embed_dataset=embed_dataset, use_targets=cfg.save_targets, training=False, strict_embedder=cfg.strict_embedder) for embed_dataset in dataset_freq.keys())
	if not all(isinstance(dataset, embedding_cache.EmbeddingCache.Dataset) for dataset in datasets):
		raise ValueError("All datasets to merge must be embedding caches")
	embed_caches: tuple[embedding_cache.EmbeddingCache, ...] = tuple(dataset.embed_cache for dataset in datasets)  # noqa

	use_targets_seq = tuple(embed_cache.use_targets for embed_cache in embed_caches)
	if any(use_targets_seq) and not all(use_targets_seq):
		raise ValueError("Cannot merge embedding caches using targets with embedding caches not using targets")
	if use_targets := use_targets_seq[0]:
		gen_target_config(cfg=cfg, embedder=embedder, targets=datasets[0].targets, num_invalid_targets=datasets[0].num_invalid_targets)

	assert len(dataset_freq) == len(embed_caches)
	embedding_cache_writers.MergeCachesWriter(
		cache_path=cache_path,
		embedder=embedder,
		caches=embed_caches,
		freqs=tuple(dataset_freq.values()),
		use_targets=use_targets,
		multi_mode=cfg.multi_mode,
		batch_size=cfg.batch_size,
	).generate()

#
# Action: Train
#

# Training loop configuration class (frozen JSON-ifiable precomputed values passed to training loop)
@dataclasses.dataclass(frozen=True)
class TrainLoopConfig:
	run_dir: str
	wandb: bool
	save_every_min: int
	save_every_max: int
	save_top1_min: float
	save_top1_delta: float
	gradient_clip: float
	last_dropout_chunks: int
	last_dropout_factor: float
	device_is_cpu: bool
	epoch_batches: int
	chunk_batches: int
	chunk_samples: int
	max_chunks: int
	ewa_factor: float
	ewa_factor_inv: float

# Training loop state class (torch-saveable running variables in the training loop)
@dataclasses.dataclass(eq=False)
class TrainLoopState:
	epoch_id: int = 1
	chunk_id: int = 1
	batch_id: int = 1
	sample_id: int = 1
	epoch_started: bool = False
	chunk_started: bool = False
	epoch_batches_left: int = -1
	start_time: Optional[float] = None
	epoch_start_time: Optional[float] = None
	chunk_start_time: Optional[float] = None
	num_grad_norms: int = 0
	grad_norms: Optional[torch.Tensor] = None
	ewa_train_loss_sum: float = 0.0
	ewa_train_loss_basis: float = 0.0
	ewa_train_loss: Optional[float] = None
	ewa_train_correct: float = 0.0
	ewa_train_tokens: float = 0.0
	ewa_train_top1: float = 0.0
	ewa_train_top1_max: float = 0.0
	ewa_train_top1_last: float = 0.0
	allow_save_delta: bool = False
	saved_num: int = 0
	saved_chunk_id: int = 0
	saved_ewa_train_loss: float = math.inf
	saved_ewa_train_top1: float = 0.0
	saved_ewa_train_top1_max: float = 0.0

# Action: Train
def action_train(cfg: omegaconf.DictConfig, hydra_dir: str, use_wandb: bool):

	device, device_is_cpu, device_is_cuda = load_device(cfg=cfg)
	embedder = load_embedder(cfg=cfg, device=device)
	dataset = load_embedding_dataset(cfg=cfg, embedder=embedder, use_targets=True, training=True, strict_embedder=cfg.strict_embedder)
	target_config = gen_target_config(cfg=cfg, embedder=embedder, targets=dataset.targets, num_invalid_targets=dataset.num_invalid_targets)
	data_config = gen_data_config(cfg=cfg, dataset=dataset)
	loader, loader_info = load_embedding_dataset_loader(cfg=cfg, dataset=dataset, training=True, device=device)
	grad_accum = embedding_dataset.GradAccum(loader=loader, loader_info=loader_info, accum_size=cfg.accum_factor, drop_last=True)

	batch_size = grad_accum.batch_size
	epoch_batches = grad_accum.loader_batches
	epoch_samples = grad_accum.loader_samples
	chunk_batches = max(math.ceil(dataset.num_valid_targets * cfg.chunk_scale / batch_size), grad_accum.accum_size, 1)
	chunk_samples = chunk_batches * batch_size

	max_chunks = sys.maxsize - 1
	max_chunks_reason = 'no maximums specified'
	if max_chunks > cfg.max_chunks >= 1:
		max_chunks = cfg.max_chunks
		max_chunks_reason = f'max chunks {cfg.max_chunks} specified'
	if cfg.max_epochs >= 1 and max_chunks > (max_epoch_chunks := (cfg.max_epochs * epoch_batches) // chunk_batches):
		max_chunks = max_epoch_chunks
		max_chunks_reason = f'max epochs {cfg.max_epochs} specified'

	log.info(f"Have {loader_info.available_samples} training samples available in the dataset")
	log.info(f"Training {batch_size} samples per batch")
	log.info(f"Training {epoch_batches} batches = {epoch_samples} samples per epoch{f' (gradient accumulation factor {grad_accum.accum_size} => {grad_accum.loader_steps} optimizer updates)' if grad_accum.accum_size > 1 else ''}")
	log.info(f"Training {chunk_batches} batches = {chunk_samples} samples per chunk")
	log.info(f"Training nominally for {max_chunks if max_chunks < sys.maxsize - 1 else 'unlimited'} chunks ({max_chunks_reason})")

	if cfg.mean_shift:
		mean_shift_path = resolve_source_path(cfg.mean_shift_path.replace('$EMBEDDER', safe_embedder_spec(cfg.embedder_spec)))
		log.info(f"Applying mean shift to embedding vectors based on: {mean_shift_path}")
		with open(mean_shift_path, 'r') as file:
			mean_shift_json = json.load(file)
		for key, value in mean_shift_json['cfg_embedder'].items():
			if (cfg_value := getattr(cfg, key)) != value:
				msg = f"Mean shift was calculated with {key}={value} but current config has {key}={cfg_value}"
				if key == 'embedder_spec':
					raise ValueError(msg)
				else:
					log.warning(msg)
		mean_shift = torch.tensor(mean_shift_json['mean_shift'], dtype=embedder.embed_dtype, device=device)
		if mean_shift.shape != (embedder.embed_dim,):
			raise ValueError(f"Mean shift has wrong shape: {mean_shift.shape} vs {(embedder.embed_dim,)}")
		mean_shift = mean_shift.unsqueeze(dim=0)
		log.info(f"Mean shift vector has norm {torch.linalg.norm(mean_shift, dim=1).item():.3f} and element range {mean_shift.min():.3f} to {mean_shift.max():.3f}")
	else:
		mean_shift = None

	embed_noise = embedding_noise.EmbeddingNoise.create(scheme=cfg.noise_scheme, embed_dim=embedder.embed_dim, vec_norm=cfg.noise_vec_norm, angle_min=cfg.noise_angle_min, angle_max=cfg.noise_angle_max, angle_std=cfg.noise_angle_std, mix_ratio=cfg.noise_mix_ratio)

	if cfg.last_dropout_chunks >= 1:
		if cfg.last_dropout_factor == 0:
			log.info(f"Will use zero dropout probability in last {cfg.last_dropout_chunks} chunks")
		else:
			log.info(f"Will rescale dropout probabilities by \u00D7{cfg.last_dropout_factor:g} for last {cfg.last_dropout_chunks} chunks")

	train_loop_config = TrainLoopConfig(
		run_dir=hydra_dir,
		wandb=use_wandb,
		save_every_min=cfg.save_every_min,
		save_every_max=cfg.save_every_max,
		save_top1_min=cfg.save_top1_min / 100,
		save_top1_delta=cfg.save_top1_delta / 100,
		gradient_clip=cfg.gradient_clip,
		last_dropout_chunks=cfg.last_dropout_chunks,
		last_dropout_factor=cfg.last_dropout_factor,
		device_is_cpu=device_is_cpu,
		epoch_batches=epoch_batches,
		chunk_batches=chunk_batches,
		chunk_samples=chunk_samples,
		max_chunks=max_chunks,
		ewa_factor=(ewa_factor := 0.5 ** (1 / (cfg.loss_ewa_halflife * chunk_batches))),
		ewa_factor_inv=1 - ewa_factor,
	)

	with dataset.loaded():

		checkpoint, checkpoint_path = load_decoder_checkpoint(cfg=cfg, hydra_dir=hydra_dir, target_config=target_config, data_config=data_config)
		if checkpoint is None:
			log.info("Training model from scratch")
		elif cfg.load_train_state:
			log.info(f"Resuming training from checkpoint: {checkpoint_path}")
			if checkpoint_path.endswith('.model'):
				log.warning("Attempting to resume training from a checkpoint that is likely a model-only checkpoint")
			check_loaded_config(name='train loop config', using=dataclasses.asdict(train_loop_config), loaded=checkpoint['train_loop_config'], ignore={'run_dir'})
		else:
			log.info(f"Starting training from pretrained weights: {checkpoint_path}")

		model = load_decoder_model(cfg=cfg, embedder=embedder, data_config=data_config, checkpoint=checkpoint)

		if checkpoint is not None and not cfg.load_train_state:
			checkpoint = None
			gc.collect()

		if checkpoint is None:
			train_loop_state = TrainLoopState()
		else:
			log.info(f"Loading {len(checkpoint['train_loop_state'])} training loop state items from checkpoint...")
			train_loop_state = utils.dataclass_from_dict(cls=TrainLoopState, state=checkpoint['train_loop_state'])

		amp_context, amp_dtype = load_decoder_amp(cfg=cfg, device=device)
		amp_scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16))
		if amp_scaler_enabled := amp_scaler.is_enabled():
			log.info(f"Created AMP gradient scaler for dtype: {amp_dtype}")
		if checkpoint is not None:
			if amp_scaler_enabled == checkpoint['amp_scaler_enabled']:
				if amp_scaler_enabled:
					log.info(f"Loading {len(checkpoint['amp_scaler_state_dict'])} items from AMP gradient scaler state dict...")
				amp_scaler.load_state_dict(checkpoint['amp_scaler_state_dict'])
			else:
				log.warning(f"Loaded AMP gradient scaler was {'enabled' if checkpoint['amp_scaler_enabled'] else 'disabled'} but current one is {'enabled' if amp_scaler_enabled else 'disabled'} => Not loading AMP gradient scaler state dict")

		if not device_is_cpu:
			log.info(f"Moving model to {device.type.upper()}...")
			model.to(device=device)
			utils.dataclass_to(train_loop_state, device=device)

		if checkpoint is not None and cfg.load_lr_state:
			checkpoint_cfg_flat = checkpoint['cfg_flat']
			init_lr, final_lr, lr_scheduler, lr_warmup = checkpoint_cfg_flat['init_lr'], checkpoint_cfg_flat['final_lr'], checkpoint_cfg_flat['lr_scheduler'], checkpoint_cfg_flat['lr_warmup']
		else:
			init_lr, final_lr, lr_scheduler, lr_warmup = cfg.init_lr, cfg.final_lr, cfg.lr_scheduler, cfg.lr_warmup

		model_params = tuple(param for param in model.parameters() if param.requires_grad)
		if cfg.weight_decay_1d:
			model_param_groups = ({'params': model_params, 'weight_decay': cfg.weight_decay},)
			log.info(f"Applying weight decay to all {len(model_params)} trainable param tensors = {sum(param.numel() for param in model_params)} trainable parameters")
		else:
			model_params_1d = tuple(param for param in model_params if param.dim() < 2)
			model_params_nd = tuple(param for param in model_params if param.dim() >= 2)
			model_param_groups = []
			if model_params_1d:
				model_param_groups.append({'params': model_params_1d, 'weight_decay': 0.0})
			if model_params_nd:
				model_param_groups.append({'params': model_params_nd, 'weight_decay': cfg.weight_decay})
			log.info(f"Applying weight decay to {len(model_params_nd)}/{len(model_params)} trainable param tensors = {sum(param.numel() for param in model_params_nd)}/{sum(param.numel() for param in model_params)} trainable parameters")

		optimizer_lower = cfg.optimizer.lower()
		if optimizer_lower == 'adamw':
			optimizer = torch.optim.AdamW(model_param_groups, lr=init_lr, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay, fused=device_is_cuda)
		elif optimizer_lower == 'adamp':
			optimizer = timm.optim.AdamP(model_param_groups, lr=init_lr, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay, nesterov=cfg.nesterov)
		else:
			raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")
		optimizer_str = repr(optimizer).replace('\n\n', ' | ').replace('\n    ', ', ').replace('\n', '')
		log.info(f"Created optimizer: {optimizer_str}")

		if checkpoint is not None:
			optimizer_type = utils.get_class_str(type(optimizer))
			if optimizer_type == checkpoint['optimizer_type']:
				log.info(f"Loading {len(checkpoint['optimizer_state_dict']['state'])} items from optimizer state dict...")
				optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # Note: It is okay to load a CPU checkpoint to a GPU optimizer because appropriate device casting is automatically performed on loading the state dict
				if not cfg.load_lr_state:
					for param_group in optimizer.param_groups:
						param_group['lr'] = param_group['initial_lr'] = init_lr
			else:
				log.warning(f"Loaded optimizer type ({checkpoint['optimizer_type']}) is different to current optimizer type ({optimizer_type}) => Not loading optimizer state dict")

		if lr_warmup >= 1:
			scheduler_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1 / (lr_warmup + 1), end_factor=1, total_iters=lr_warmup)
			log.info(f"Using warmup scheduler: {utils.get_type_str(type(scheduler_warmup))}(chunks={scheduler_warmup.total_iters})")
			if checkpoint is not None:
				if cfg.load_lr_state:
					log.info(f"Loading {len(checkpoint['scheduler_warmup_state_dict'])} items from warmup scheduler state dict...")
					scheduler_warmup.load_state_dict(checkpoint['scheduler_warmup_state_dict'])
				for param_group, param_lr in zip(optimizer.param_groups, scheduler_warmup.get_last_lr()):
					param_group['lr'] = param_lr
		else:
			scheduler_warmup = None

		lr_scheduler_lower = lr_scheduler.lower()
		if lr_scheduler_lower == 'const':
			scheduler = None
		elif lr_scheduler_lower == 'cosine':
			T_max = max((max_chunks if final_lr > 0 else max_chunks + 1) - train_loop_state.chunk_id, 1)
			scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=final_lr)
			log.info(f"Using scheduler: {utils.get_type_str(type(scheduler))}(chunks={scheduler.T_max}, baselr={scheduler.base_lrs[0]:.2e}, finallr={scheduler.eta_min:.2e})")
		else:
			raise ValueError(f"Unsupported learning rate scheduler: {lr_scheduler}")

		if scheduler is not None and checkpoint is not None:
			if cfg.load_lr_state:
				log.info(f"Loading {len(checkpoint['scheduler_state_dict'])} items from scheduler state dict...")
				scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
			for param_group, param_lr in zip(optimizer.param_groups, scheduler.get_last_lr()):
				param_group['lr'] = param_lr

		del checkpoint  # Note: Free up memory
		utils.release_cuda_memory(device=device)

		model = finalise_decoder_model(cfg=cfg, model=model)

		training_loop(
			cfg_flat=utils_config.flatten_config(cfg),
			C=train_loop_config,
			S=train_loop_state,
			model=model,
			model_params=model_params,
			target_nouns=dataset.targets,
			num_invalid_target_nouns=dataset.num_invalid_targets,
			mean_shift=mean_shift,
			embed_noise=embed_noise,
			grad_accum=grad_accum,
			optimizer=optimizer,
			scheduler_warmup=scheduler_warmup,
			scheduler=scheduler,
			preinit_lr=optimizer.param_groups[0]['lr'] if scheduler_warmup is None or (train_loop_state.chunk_id > 1 and cfg.load_lr_state) else 0,
			amp_context=amp_context,
			amp_scaler=amp_scaler,
			device=device,
		)

# Training loop
def training_loop(
	cfg_flat: dict[str, Any],  # Note: This should only be used for the purpose of saving checkpoints
	C: TrainLoopConfig,
	S: TrainLoopState,
	model: embedding_decoder.EmbeddingDecoder,
	model_params: Iterable[torch.nn.Parameter],
	target_nouns: tuple[str, ...],
	num_invalid_target_nouns: int,
	mean_shift: Optional[torch.Tensor],
	embed_noise: Optional[embedding_noise.EmbeddingNoise],
	grad_accum: embedding_dataset.GradAccum,
	optimizer: torch.optim.Optimizer,
	scheduler_warmup: Optional[torch.optim.lr_scheduler.LRScheduler],
	scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
	preinit_lr: float,
	amp_context: ContextManager,
	amp_scaler: torch.cuda.amp.GradScaler,
	device: torch.device,
):

	S.start_time = time.perf_counter()

	stop_training = (S.chunk_id >= C.max_chunks + 1)
	if C.last_dropout_chunks >= 1 and S.chunk_id > C.max_chunks - C.last_dropout_chunks:
		utils.rescale_dropout(model=model, factor=C.last_dropout_factor)

	assert C.epoch_batches >= 1
	if S.epoch_batches_left < 0:  # Fresh training start
		S.epoch_batches_left = C.epoch_batches
		if C.gradient_clip > 0:
			S.grad_norms = torch.empty(size=(math.ceil(C.chunk_batches / grad_accum.accum_size),), device=device)
	elif S.epoch_batches_left == 0:  # Resuming from a point where an epoch was imminently finishing (if resuming from any other situation then no action is required)
		S.epoch_batches_left = C.epoch_batches
		S.epoch_id += 1
		S.epoch_started = False

	if C.wandb:
		wandb.log(dict(
			epoch=S.epoch_id - 1,
			chunk=S.chunk_id - 1,
			batch=S.batch_id - 1,
			sample=S.sample_id - 1,
			lr=preinit_lr,
			loss=S.ewa_train_loss,
			top1=S.ewa_train_top1 * 100,
			top1_max=S.ewa_train_top1_max * 100,
		))

	with utils.ProgressBar(total=C.chunk_samples, leave=False, unit='noun', unit_scale=False, dynamic_ncols=True, smoothing=0.08, delay=1) as progress_bar:
		while not stop_training:

			with progress_bar.pause_display():
				log.info('-' * 80)
				log.info(f"Epoch {S.epoch_id} = Batch {S.batch_id} = Sample {S.sample_id}")
			S.epoch_start_time = time.perf_counter()  # Note: This may lead to a shorter than usual epoch time if resuming mid-epoch

			model.train()
			optimizer.zero_grad(set_to_none=True)

			for embed, target, mask, weight in grad_accum.loader():

				S.epoch_started = True
				S.chunk_started = True

				chunk_batch_id = (S.batch_id - 1) % C.chunk_batches
				if chunk_batch_id == 0:
					log.info(f"Chunk {S.chunk_id} = Batch {S.batch_id} = Sample {S.sample_id}")
					progress_bar.start(desc=f'[Epoch {S.epoch_id}] Chunk {S.chunk_id}')
					S.chunk_start_time = time.perf_counter()

				if mean_shift is not None:
					embed.add_(mean_shift)
					torch.nn.functional.normalize(embed, dim=-1, out=embed)
				if embed_noise:
					embed = embed_noise(embed)

				batch_samples = embed.shape[0]
				with amp_context:
					_, batch_padding, batch_loss_sum, batch_loss_basis, batch_correct = model(embed=embed, target=target, target_padding=mask, target_weight=weight, calc_loss=True, calc_correct=True, only_pred=False, guide_targets=None)
					mean_accum_batch_loss, optimizer_step = grad_accum.accum_loss(mean_batch_loss=batch_loss_sum / batch_loss_basis, num_in_batch=batch_samples)  # Note: Gradient accumulation accumulates on the basis of samples and not the returned loss basis (because loss bases are not known in advance), but for large enough batch sizes this should make minimal difference to overall training
				amp_scaler.scale(mean_accum_batch_loss).backward()

				if batch_padding is not None:
					batch_padding_sum = batch_padding.sum()
				batch_correct_sum = batch_correct.sum()

				if optimizer_step:  # Note: Saving and resuming potentially does not optimize two partially accumulated batches (the accumulation batch in progress during the save and the accumulation batch when S.epoch_batches_left reaches 0 long after resuming), but this is acceptable and has extremely minimal impact on the training
					if C.gradient_clip > 0:
						amp_scaler.unscale_(optimizer)
						S.grad_norms[S.num_grad_norms] = torch.nn.utils.clip_grad_norm_(model_params, max_norm=C.gradient_clip, error_if_nonfinite=True)  # Note: Output gradient norm is already detached, i.e. not on the computational graph
						S.num_grad_norms += 1
					amp_scaler.step(optimizer)  # Note: This implicitly calls amp_scaler.unscale_(optimizer) if it hasn't already manually been called
					amp_scaler.update()
					optimizer.zero_grad(set_to_none=True)

				S.ewa_train_loss_sum *= C.ewa_factor
				S.ewa_train_loss_basis *= C.ewa_factor
				S.ewa_train_loss_sum += batch_loss_sum.item()  # GPU-CPU synchronization point
				S.ewa_train_loss_basis += batch_loss_basis.item()  # GPU-CPU synchronization point
				S.ewa_train_loss = S.ewa_train_loss_sum / S.ewa_train_loss_basis

				batch_train_tokens_int = target.numel()
				if batch_padding is not None:
					batch_train_tokens_int -= batch_padding_sum.item()  # GPU-CPU synchronization point
				batch_train_correct_int = batch_correct_sum.item()  # GPU-CPU synchronization point
				batch_top1 = batch_train_correct_int / batch_train_tokens_int  # Note: For the multi-target case this is more of a proxy measure than an exact measure because it is impossible to get 100% accuracy (the first sequence location where each pair of multi-targets for a sample diverge can only have one of the two target tokens being correct), but it is still true that 'better' models will have strictly better top-1

				S.ewa_train_correct *= C.ewa_factor
				S.ewa_train_tokens *= C.ewa_factor
				S.ewa_train_correct += batch_train_correct_int
				S.ewa_train_tokens += batch_train_tokens_int
				S.ewa_train_top1 = S.ewa_train_correct / S.ewa_train_tokens  # Note: Refer to the note for batch_top1 above explaining why this is a proxy measure in the multi-target case (but still inducing the same total order on the set of all possible models as an ideal top-1 calculation would)
				S.ewa_train_top1_max = max(S.ewa_train_top1_max, S.ewa_train_top1)

				current_lr = optimizer.param_groups[0]['lr']
				progress_bar_postfix = f"lr={current_lr:.2e}, loss={S.ewa_train_loss:.2e}, top1={batch_top1:.2%}/{S.ewa_train_top1:.3%}"
				progress_bar.update(n=batch_samples, postfix=progress_bar_postfix)

				S.sample_id += batch_samples
				S.batch_id += 1
				S.epoch_batches_left -= 1

				if chunk_batch_id == C.chunk_batches - 1:

					chunk_elapsed_time = time.perf_counter() - S.chunk_start_time
					progress_bar.stop()

					if C.gradient_clip > 0:
						if S.num_grad_norms < 1:
							raise ValueError("Chunk size must be such that at least a single gradient accumulation optimizer step happens per chunk")
						elif S.num_grad_norms == 1:
							grad_norm_min = grad_norm_mean = grad_norm_max = S.grad_norms[0].item()  # GPU-CPU synchronization point
							grad_norm_std = 0
						else:
							grad_norms = S.grad_norms[:S.num_grad_norms]
							grad_norm_min = grad_norms.min().item()  # GPU-CPU synchronization point
							grad_norm_mean = grad_norms.mean().item()
							grad_norm_std = grad_norms.std().item()
							grad_norm_max = grad_norms.max().item()
						log.info(f"Total gradient norm stats for {S.num_grad_norms} steps: {grad_norm_min:.4g} <= {grad_norm_mean:.4g} + {grad_norm_std:.4g}z <= {grad_norm_max:.4g}{f' (clipped to {C.gradient_clip:.4g})' if grad_norm_max > C.gradient_clip else ''}")
						S.num_grad_norms = 0
					else:
						grad_norm_min = grad_norm_mean = grad_norm_std = grad_norm_max = None

					log.info(f"Trained chunk {S.chunk_id} in {chunk_elapsed_time:.1f}s at {C.chunk_samples / chunk_elapsed_time:.0f}noun/s: {progress_bar_postfix}")

					if scheduler_warmup:
						scheduler_warmup.step()
					if scheduler:
						scheduler.step()

					S.chunk_id += 1
					S.chunk_started = False
					if S.chunk_id >= C.max_chunks + 1:
						stop_training = True

					save_chunk_id = S.chunk_id - 1
					chunks_since_save = save_chunk_id - S.saved_chunk_id
					if S.ewa_train_top1 >= C.save_top1_min and S.ewa_train_top1 - S.ewa_train_top1_last <= C.save_top1_delta:
						S.allow_save_delta = True
					S.ewa_train_top1_last = S.ewa_train_top1
					if stop_training or chunks_since_save >= C.save_every_max or (chunks_since_save >= C.save_every_min and S.ewa_train_top1 >= C.save_top1_min and S.allow_save_delta and S.ewa_train_top1 >= S.saved_ewa_train_top1_max):
						S.saved_num += 1
						S.saved_chunk_id = save_chunk_id
						S.saved_ewa_train_loss = S.ewa_train_loss
						S.saved_ewa_train_top1 = S.ewa_train_top1
						S.saved_ewa_train_top1_max = max(S.saved_ewa_train_top1_max, S.ewa_train_top1)
						checkpoint_path = save_train_checkpoint(cfg_flat=cfg_flat, model=model, C=C, S=S, target_nouns=target_nouns, num_invalid_target_nouns=num_invalid_target_nouns, optimizer=optimizer, scheduler_warmup=scheduler_warmup, scheduler=scheduler, amp_scaler=amp_scaler, model_only=False)
						log.info(f"Saved checkpoint: {checkpoint_path}")

					if C.last_dropout_chunks >= 1 and S.chunk_id == C.max_chunks - C.last_dropout_chunks + 1:
						utils.rescale_dropout(model=model, factor=C.last_dropout_factor)

					if C.wandb:
						wandb.log(dict(
							chunk=S.chunk_id - 1,
							batch=S.batch_id - 1,
							sample=S.sample_id - 1,
							lr=current_lr,
							loss=S.ewa_train_loss,
							top1=S.ewa_train_top1 * 100,
							top1_max=S.ewa_train_top1_max * 100,
							batch_top1=batch_top1 * 100,
							chunk_time=chunk_elapsed_time,
							grad_norm_min=grad_norm_min,
							grad_norm_mean=grad_norm_mean,
							grad_norm_std=grad_norm_std,
							grad_norm_max=grad_norm_max,
							saved_num=S.saved_num,
							saved_chunk=S.saved_chunk_id,
							saved_loss=S.saved_ewa_train_loss,
							saved_top1=S.saved_ewa_train_top1 * 100,
							saved_top1_max=S.saved_ewa_train_top1_max * 100,
						))

					if stop_training:
						break

				if S.epoch_batches_left == 0:  # Note: Other than in the first epoch after a resume, this should never actually shorten an epoch even though the break executes
					break

			if S.epoch_batches_left == 0:

				epoch_elapsed_time = time.perf_counter() - S.epoch_start_time
				with progress_bar.pause_display():
					log.info(f"Epoch {S.epoch_id} finished in {epoch_elapsed_time:.1f}s")

				S.epoch_batches_left = C.epoch_batches
				S.epoch_id += 1
				S.epoch_started = False

				if C.wandb:
					wandb.log(dict(
						epoch=S.epoch_id - 1,
						epoch_time=epoch_elapsed_time,
					))

	if not S.epoch_started:
		S.epoch_id -= 1
	if not S.chunk_started:
		S.chunk_id -= 1
	S.batch_id -= 1
	S.sample_id -= 1

	log.info('-' * 80)
	elapsed_time = time.perf_counter() - S.start_time
	log.info(f"Trained for {S.chunk_id} chunks (up to {S.epoch_id} epochs) in {elapsed_time:.1f}s")
	log.info(f"Trained {S.batch_id} batches = {S.sample_id} samples")

	if C.wandb:
		wandb.log(dict(
			epoch=S.epoch_id,
			chunk=S.chunk_id,
			batch=S.batch_id,
			sample=S.sample_id,
			total_time=elapsed_time,
		))

# Save a training checkpoint (Note: GradAccum state is better not saved as then the gradient accumulation never triggers an optimizer step with less than the required accumulation size, and instead safely discards up to two partially accumulated gradients instead)
TORCH_SAVE_CLASSES = utils.TorchSaveClasses()
def save_train_checkpoint(
	cfg_flat: dict[str, Any],
	model: embedding_decoder.EmbeddingDecoder,
	C: Optional[TrainLoopConfig],
	S: Optional[TrainLoopState],
	target_nouns: tuple[str, ...],
	num_invalid_target_nouns: int,
	optimizer: Optional[torch.optim.Optimizer],
	scheduler_warmup: Optional[torch.optim.lr_scheduler.LRScheduler],
	scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
	amp_scaler: Optional[torch.cuda.amp.GradScaler],
	*,
	model_only: bool = False,
	run_dir: Optional[str] = None,
	chunk_id: Optional[int] = None,
) -> str:

	checkpoint = dict(
		cfg_flat=cfg_flat,
		target_config=dataclasses.asdict(model.target_config),
		data_config=dataclasses.asdict(model.data_config),
		model_state_dict=getattr(model, '_orig_mod', model).state_dict(),  # Note: Compiled models store the original model as model._orig_mod (child modules, parameters, buffers are shared between the two)
		target_nouns=target_nouns,
		num_invalid_target_nouns=num_invalid_target_nouns,
	)

	checkpoint_path = os.path.join((C.run_dir if run_dir is None else run_dir), f"ovod_chunk{S.saved_chunk_id if chunk_id is None else chunk_id:04d}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
	if model_only:
		checkpoint_path += '.model'
	else:
		checkpoint_path += '.train'
		checkpoint.update(
			train_loop_config=dataclasses.asdict(C),
			train_loop_state=dataclasses.asdict(S),
			optimizer_type=utils.get_class_str(type(optimizer)),
			optimizer_state_dict=optimizer.state_dict(),
			scheduler_warmup_state_dict=scheduler_warmup and scheduler_warmup.state_dict(),
			scheduler_state_dict=scheduler and scheduler.state_dict(),
			amp_scaler_enabled=amp_scaler.is_enabled(),
			amp_scaler_state_dict=amp_scaler.state_dict(),
		)

	if reqd_classes := TORCH_SAVE_CLASSES.get_classes(checkpoint, exclude_builtins=True, exclude_torch=True):  # Note: This has a very short runtime (<1ms on i5 was seen), so the extra pickling effort should not be an efficiency issue
		log.warning(f"Checkpoint required non-standard classes: {{{', '.join(utils.get_class_str(cls) for cls in sorted(reqd_classes, key=lambda x: x.__qualname__))}}}")
	torch.save(checkpoint, checkpoint_path)

	return checkpoint_path

#
# Action: Fix checkpoints
#

# Action: Fix checkpoints
# noinspection PyBroadException
def action_fix_checkpoints(cfg: omegaconf.DictConfig, hydra_dir: str):

	reqd_configs = ('action',)
	device_configs = ('device',)
	embedder_configs = ('embedder_spec', 'embedder_amp', 'embedder_amp_bf16', 'batch_size_token', 'batch_size_embed', 'batch_size_image', 'embedder_compile', 'embedder_optimum')
	dataset_configs = ('embedding_dataset', 'vocab_thres', 'prompt_collection', 'hypernym_collection', 'embedding_cache_dir', 'vocab_path', 'prompt_path', 'noun_cache', 'noun_cache_dir', 'batch_size', 'strict_embedder')
	all_configs = tuple(itertools.chain(reqd_configs, device_configs, embedder_configs, dataset_configs))
	checkpoint_keys = ('cfg_flat', 'target_config', 'data_config', 'model_state_dict', 'train_loop_config', 'train_loop_state', 'optimizer_type', 'optimizer_state_dict', 'scheduler_warmup_state_dict', 'scheduler_state_dict', 'amp_scaler_enabled', 'amp_scaler_state_dict')

	fix_queue = []
	hydra_parent_dir = os.path.dirname(hydra_dir)
	log.info(f"Scanning all trained model directories in: {hydra_parent_dir}")
	model_dirs = sorted(os.listdir(hydra_parent_dir))
	with tqdm.tqdm(desc=f'Scanning model dirs', total=len(model_dirs), unit='dir', unit_scale=False, dynamic_ncols=True, smoothing=0.08) as progress_bar:
		for entry in model_dirs:
			entry_path = os.path.join(hydra_parent_dir, entry)
			if os.path.isdir(entry_path) and re.fullmatch(r'ovod_[0-9]{8}_[0-9]{6}', entry):
				for subentry in sorted(os.listdir(entry_path)):
					model_path = os.path.join(entry_path, subentry)
					if os.path.isfile(model_path) and re.fullmatch(r'ovod_chunk[0-9]{4,}_[0-9]{8}_[0-9]{6}\.train', subentry):
						checkpoint = torch.load(model_path, map_location='cpu')
						if all(key in checkpoint for key in checkpoint_keys):
							cfg_flat = checkpoint['cfg_flat']
							if all(name in cfg_flat for name in all_configs):
								if cfg_flat['action'] == 'train' and ('target_nouns' not in checkpoint or 'num_invalid_target_nouns' not in checkpoint):
									fix_queue.append((
										tuple((name, cfg_flat[name]) for name in device_configs),
										tuple((name, cfg_flat[name]) for name in embedder_configs),
										tuple((name, cfg_flat[name]) for name in dataset_configs),
										model_path,
									))
							else:
								with tqdm.tqdm.external_write_mode():
									log.warning(f"Skipping {os.path.join(entry, subentry)} because it does not have the expected config parameters")
						else:
							with tqdm.tqdm.external_write_mode():
								log.warning(f"Skipping {os.path.join(entry, subentry)} because it does not have the expected checkpoint keys")
						del checkpoint
			progress_bar.set_postfix_str(f"fix={len(fix_queue)}", refresh=False)
			progress_bar.update(n=1)
		assert progress_bar.n == progress_bar.total

	if not fix_queue:
		log.info("No models to fix!")
		return

	if cfg.fix_force_vtx:

		model_paths = set(load_checkpoint_paths(cfg=cfg, hydra_dir=hydra_dir, allow_empty=True))

		device, device_is_cpu, device_is_cuda = load_device(cfg=cfg)
		embedder = load_embedder(cfg=cfg, device=device)
		dataset = load_embedding_dataset(cfg=cfg, embedder=embedder, embed_dataset='NounDataset', use_targets=False, use_cache=False)
		target_nouns = dataset.targets
		num_invalid_target_nouns = dataset.num_invalid_targets

		assert num_invalid_target_nouns == 0
		target_nouns_bin = ('',) + target_nouns
		num_invalid_target_nouns_bin = num_invalid_target_nouns + 1

		log.info('-' * 80)

		for _, _, _, model_path in fix_queue:

			if model_paths and model_path not in model_paths:
				continue

			checkpoint = torch.load(model_path, map_location='cpu')
			cfg_flat = checkpoint['cfg_flat']
			emb_dataset = cfg_flat['embedding_dataset']
			vocab_thres = cfg_flat['vocab_thres']
			log.info(f"Loaded checkpoint with: embedding_dataset={emb_dataset}, vocab_thres={vocab_thres}")
			if emb_dataset == 'NounDataset':
				if vocab_thres != cfg.vocab_thres:
					log.warning(f"Skipping as vocab threshold mismatch: {model_path}")
					continue
				checkpoint.update(target_nouns=target_nouns, num_invalid_target_nouns=num_invalid_target_nouns)
			else:
				if not emb_dataset.endswith(f'_vt{cfg.vocab_thres}.bin'):
					log.warning(f"Skipping as vocab threshold mismatch: {model_path}")
					continue
				checkpoint.update(target_nouns=target_nouns_bin, num_invalid_target_nouns=num_invalid_target_nouns_bin)

			if not cfg.dry_run:
				torch.save(checkpoint, model_path)
			log.info(f"{'DRYRUN: ' if cfg.dry_run else ''}Fixed model with {len(checkpoint['target_nouns'])} - {checkpoint['num_invalid_target_nouns']} = {len(checkpoint['target_nouns']) - checkpoint['num_invalid_target_nouns']} target nouns: {model_path}")
			del checkpoint

	else:

		fix_queue.sort()
		localcfg = cfg.copy()
		localcfg['action'] = 'train'
		last_fix_device = last_fix_embedder = last_fix_dataset = None
		device = embedder = target_nouns = num_invalid_target_nouns = None

		for fix_device, fix_embedder, fix_dataset, model_path in fix_queue:

			changed_device = changed_embedder = changed_dataset = False
			if fix_device != last_fix_device:
				changed_device = changed_embedder = changed_dataset = True
			elif fix_embedder != last_fix_embedder:
				changed_embedder = changed_dataset = True
			elif fix_dataset != last_fix_dataset:
				changed_dataset = True

			if changed_dataset:
				log.info('-' * 80)

			if changed_device:
				last_fix_device = fix_device
				fix_device_map = dict(fix_device)
				log.info(f"NEW DEVICE: {fix_device_map}")
				localcfg.update(fix_device_map)
			if changed_embedder:
				last_fix_embedder = fix_embedder
				fix_embedder_map = dict(fix_embedder)
				log.info(f"NEW EMBEDDER: {fix_embedder_map}")
				localcfg.update(fix_embedder_map)
			if changed_dataset:
				last_fix_dataset = fix_dataset
				fix_dataset_map = dict(fix_dataset)
				log.info(f"NEW DATASET: {fix_dataset_map}")
				localcfg.update(fix_dataset_map)

			if changed_device:
				device, device_is_cpu, device_is_cuda = load_device(cfg=localcfg)
			if changed_embedder:
				try:
					embedder = load_embedder(cfg=localcfg, device=device)
				except Exception as e:
					log.error(f"Failed to load embedder: {e}")
					log.warning(f"Skipping: {model_path}")
					embedder = last_fix_embedder = None
					continue
			if changed_dataset:
				try:
					dataset = load_embedding_dataset(cfg=localcfg, embedder=embedder, use_targets=True, training=True, strict_embedder=localcfg.strict_embedder)
				except Exception as e:
					log.error(f"Failed to load dataset: {e}")
					log.warning(f"Skipping: {model_path}")
					target_nouns = num_invalid_target_nouns = last_fix_dataset = None
					continue
				target_nouns = dataset.targets
				num_invalid_target_nouns = dataset.num_invalid_targets

			checkpoint = torch.load(model_path, map_location='cpu')
			assert {key: value for key, value in utils_config.flatten_config(localcfg).items() if key in all_configs} == {key: value for key, value in checkpoint['cfg_flat'].items() if key in all_configs}
			assert target_nouns is not None and num_invalid_target_nouns is not None
			checkpoint.update(target_nouns=target_nouns, num_invalid_target_nouns=num_invalid_target_nouns)

			if not cfg.dry_run:
				torch.save(checkpoint, model_path)
			log.info(f"{'DRYRUN: ' if cfg.dry_run else ''}Fixed model with {len(target_nouns)} - {num_invalid_target_nouns} = {len(target_nouns) - num_invalid_target_nouns} target nouns: {model_path}")
			del checkpoint

	log.info('-' * 80)

#
# Action: Evaluate
#

# Action: Evaluate
def action_eval(cfg: omegaconf.DictConfig, hydra_dir: str, use_wandb: bool):

	model_paths = load_checkpoint_paths(cfg=cfg, hydra_dir=hydra_dir, allow_empty=True)
	if not model_paths:
		model_paths.append('')

	device, device_is_cpu, device_is_cuda = load_device(cfg=cfg)
	dataset_embedder = load_embedder(cfg=cfg, device=device)
	model_embedder = load_embedder(cfg=cfg, device=device)
	amp_context, amp_dtype = load_decoder_amp(cfg=cfg, device=device)

	for embed_dataset in (cfg.embedding_datasets or (cfg.embedding_dataset,)):
		eval_top1_metric(
			cfg=cfg,
			use_wandb=use_wandb,
			model_paths=model_paths,
			embed_dataset=embed_dataset,
			device=device,
			device_is_cpu=device_is_cpu,
			amp_context=amp_context,
			dataset_embedder=dataset_embedder,
			model_embedder=model_embedder,
		)

# Evaluate the full-epoch top-1 performance of a list of models on a given embedding dataset
def eval_top1_metric(
	cfg: omegaconf.DictConfig,
	use_wandb: bool,
	model_paths: list[str],
	embed_dataset: str,
	device: torch.device,
	device_is_cpu: bool,
	amp_context: ContextManager,
	dataset_embedder: embedders.Embedder,
	model_embedder: embedders.Embedder,
):

	dataset = load_embedding_dataset(cfg=cfg, embedder=dataset_embedder, embed_dataset=embed_dataset, use_targets=True, training=False, strict_embedder=cfg.strict_embedder)
	gen_target_config(cfg=cfg, embedder=dataset_embedder, targets=dataset.targets, num_invalid_targets=dataset.num_invalid_targets)
	gen_data_config(cfg=cfg, dataset=dataset)

	log.info('-' * 80)
	for model_path in model_paths:

		eval_loss, eval_top1_noun, eval_top1, eval_top1_seq, eval_tokens_total, num_valid_targets, num_samples, num_batches, elapsed_time = eval_top1_single(
			cfg=cfg,
			dataset=dataset,
			model_path=model_path,
			device=device,
			device_is_cpu=device_is_cpu,
			amp_context=amp_context,
			dataset_embedder=dataset_embedder,
			model_embedder=model_embedder,
		)

		if use_wandb:
			wandb.log({'Eval/' + key: value for key, value in dict(
				dataset=embed_dataset,
				model_path=model_path,
				model_dir=os.path.basename(os.path.dirname(model_path)),
				model_name=os.path.basename(model_path),
				loss=eval_loss,
				noun_top1=eval_top1_noun * 100,
				top1=eval_top1 * 100,
				**{f'top1_{i}': top1 * 100 for i, top1 in enumerate(eval_top1_seq, 1)},
				tokens=eval_tokens_total,
				num_valid_targets=num_valid_targets,
				num_samples=num_samples,
				num_batches=num_batches,
				elapsed_time=elapsed_time,
			).items()})

		utils.release_cuda_memory(device=device)
		log.info('-' * 80)

# Evaluate the full-epoch top-1 performance of a single model on a given embedding dataset
def eval_top1_single(
	cfg: omegaconf.DictConfig,
	dataset: noun_dataset.NounDataset,
	model_path: str,
	device: torch.device,
	device_is_cpu: bool,
	amp_context: ContextManager,
	dataset_embedder: embedders.Embedder,
	model_embedder: embedders.Embedder,
) -> tuple[float, float, float, list[float], int, int, int, int, float]:

	if model_path:
		log.info(f"Loading model: {model_path}")
		checkpoint, checkpoint_path = load_decoder_checkpoint(cfg=cfg, checkpoint_path=model_path)
		load_target_config(checkpoint=checkpoint, embedder=model_embedder)
	else:
		log.warning(f"Evaluating randomly initialised model")
		checkpoint = None
		model_embedder.configure_target(target_config=dataset_embedder.target_config, target_vocab=dataset_embedder.target_vocab)

	log.info("Possibly translating loaded dataset target configuration (Dataset) based on the model target configuration (Translation)")
	dataset.set_translation(target_config=model_embedder.target_config)
	guide_token_ids = load_guide_targets(guide_targets=dataset_embedder.target_vocab, embedder=model_embedder, device=device, device_is_cpu=device_is_cpu) if cfg.eval_guided else None

	loader, loader_info = load_embedding_dataset_loader(cfg=cfg, dataset=dataset, training=False, device=device)
	log.info(f"Using {loader_info.epoch_samples}/{loader_info.available_samples} samples available in the dataset")
	log.info(f"Evaluating {loader_info.batch_size} samples per batch => {loader_info.epoch_batches} batches")

	with dataset.loaded(), torch.inference_mode(), contextlib.ExitStack() as stack:  # Note: Dataset must be loaded() here each time because data_config can change from model to model even though the underlying dataset is not changing

		model = prepare_decoder_model_eval(cfg=cfg, embedder=model_embedder, data_config=dataset.data_config, checkpoint=checkpoint, device=device, device_is_cpu=device_is_cpu, eval_train=cfg.eval_train)
		del checkpoint
		gc.collect()

		log.info("Evaluating top-1 performance of loaded model on embedding dataset...")

		num_batches = 0
		num_samples = 0
		num_valid_targets = 0
		eval_samples_correct = 0
		eval_correct_seq = torch.zeros(dataset.translation.token_length, dtype=torch.int64)
		eval_tokens_seq = torch.zeros(dataset.translation.token_length, dtype=torch.int64)
		eval_loss_sum_sum = 0.0
		eval_loss_basis_sum = 0.0

		debug = cfg.eval_debug
		start_time = time.perf_counter()
		for embed, target, mask, weight in loader:

			with amp_context:
				_, batch_padding, batch_loss_sum, batch_loss_basis, batch_correct = model(embed=embed, target=target, target_padding=mask, target_weight=weight, calc_loss=True, calc_correct=True, only_pred=False, guide_targets=guide_token_ids)

			multi_dim = None if not dataset.data_config.multi_target else 0 if dataset.data_config.multi_first else 1
			multi_dims = target.shape[:-1]  # Torch size equal to (B,) or (B, M) or (M, B)
			if batch_padding is not None:
				batch_valid_targets = batch_padding.all(dim=-1).logical_not_()
				num_batch_valid_targets = batch_valid_targets.sum()

			batch_correct_seq = batch_correct.sum(dim=tuple(range(batch_correct.ndim - 1)))
			if batch_padding is not None:
				batch_padding_seq = batch_padding.sum(dim=tuple(range(batch_padding.ndim - 1)))
				batch_correct.logical_or_(batch_padding)  # Note: It's okay to taint batch_correct as we don't need it anymore after this code chunk (we just need batch_correct_seq, which remains unaffected)
			batch_sample_correct = batch_correct.all(dim=-1)
			if batch_padding is not None:
				batch_sample_correct.logical_and_(batch_valid_targets)
			if multi_dim is not None:
				batch_sample_correct = batch_sample_correct.any(dim=multi_dim)
			num_samples_correct = batch_sample_correct.sum()

			if debug:
				first_target = target if not dataset.data_config.multi_target else target[0, :, :] if dataset.data_config.multi_first else target[:, 0, :]
				with amp_context:
					if cfg.eval_guided:
						print('\n'.join('{color}{tgt} --> {guided} ({unguided})\033[0m'.format(color='\033[92m' if correct else '\033[91m', tgt=tgt, unguided=unguided, guided=guided) for correct, tgt, guided, unguided in zip(
							batch_sample_correct.tolist(),
							model_embedder.detokenize_target(first_target.cpu()),
							model_embedder.detokenize_target(model.generate(embed=embed, collect_logits=False, calc_loss=False, temperature=1, length_alpha=0, sample_weight=None, guide_targets=guide_token_ids, guide_renorm=False)[0].cpu()),
							model_embedder.detokenize_target(model.generate(embed=embed, collect_logits=False, calc_loss=False, temperature=1, length_alpha=0, sample_weight=None, guide_targets=None, guide_renorm=False)[0].cpu()),
						)))  # GPU-CPU synchronization point
					else:
						print('\n'.join('{color}{tgt} --> {predict}\033[0m'.format(color='\033[92m' if correct else '\033[91m', tgt=tgt, predict=predict) for correct, tgt, predict in zip(
							batch_sample_correct.tolist(),
							model_embedder.detokenize_target(first_target.cpu()),
							model_embedder.detokenize_target(model.generate(embed=embed, collect_logits=False, calc_loss=False, temperature=1, length_alpha=0, sample_weight=None, guide_targets=None, guide_renorm=False)[0].cpu()),
						)))  # GPU-CPU synchronization point
				debug = False

			batch_samples = embed.shape[0]
			num_batches += 1
			num_samples += batch_samples

			eval_loss_sum_sum += batch_loss_sum.item()  # GPU-CPU synchronization point
			eval_loss_basis_sum += batch_loss_basis.item()  # GPU-CPU synchronization point
			eval_loss = eval_loss_sum_sum / eval_loss_basis_sum

			num_batch_targets = math.prod(multi_dims)  # Equal to B (single target) or BM/MB (multi-target)
			if batch_padding is None:
				num_valid_targets += num_batch_targets
			else:
				num_valid_targets += num_batch_valid_targets.item()  # GPU-CPU synchronization point

			seq_tokens = target.shape[-1]
			batch_correct_seq_cpu = batch_correct_seq.cpu()  # GPU-CPU synchronization point
			eval_correct_seq[:seq_tokens] += batch_correct_seq_cpu
			batch_correct_total = batch_correct_seq_cpu.sum().item()

			if batch_padding is None:  # Note: This occurs if both mask and weight are None, i.e. every single sequence location of every single target is predicted
				eval_tokens_seq[:seq_tokens] += num_batch_targets
				batch_tokens_total = num_batch_targets * seq_tokens
			else:
				batch_padding_seq_cpu = batch_padding_seq.cpu()  # GPU-CPU synchronization point
				batch_tokens_seq = num_batch_targets - batch_padding_seq_cpu
				eval_tokens_seq[:seq_tokens] += batch_tokens_seq
				batch_tokens_total = batch_tokens_seq.sum().item()

			batch_top1 = batch_correct_total / batch_tokens_total
			eval_tokens_total = eval_tokens_seq.sum().item()
			eval_correct_total = eval_correct_seq.sum().item()
			eval_top1 = eval_correct_total / eval_tokens_total

			eval_samples_correct += num_samples_correct.item()  # GPU-CPU synchronization point
			eval_top1_noun = eval_samples_correct / num_samples

			if num_batches == 2:
				progress_bar = stack.enter_context(tqdm.tqdm(desc='Eval top-1', total=loader_info.epoch_samples, unit='noun', unit_scale=False, dynamic_ncols=True, smoothing=0.08, initial=num_samples))
			elif num_batches > 2:
				progress_bar.set_postfix_str(f"loss={eval_loss:.2e}, top1={batch_top1:.2%}/{eval_top1:.3%}, noun={eval_top1_noun:.3%}", refresh=False)
				progress_bar.update(n=batch_samples)

		elapsed_time = time.perf_counter() - start_time

		eval_top1_seq = eval_correct_seq / eval_tokens_seq
		eval_top1_seq = eval_top1_seq.tolist()

		del embed, target, mask, weight  # Note: Don't want these to be around after dataset is unloaded as they may refer to memory-mapped areas, even possibly causing a SIGSEGV by the debugger

	assert num_batches == loader_info.epoch_batches and num_samples == loader_info.epoch_samples and (progress_bar is None or progress_bar.n == loader_info.epoch_samples)
	log.info(f"Evaluated {num_batches} batches = {num_samples} samples = {num_valid_targets} valid targets")
	log.info(f"Evaluated 1 epoch in {elapsed_time:.1f}s to get loss {eval_loss:.2e}, noun {eval_top1_noun:.3%}, top-1 {eval_top1:.3%} ({eval_tokens_total} tokens predicted)")
	log.info(f"Token top-1 by sequence location: ({', '.join(format(top1, '.3%') for top1 in eval_top1_seq)})")
	log.info(f"Tokens predicted by sequence location: ({', '.join(format(num) for num in eval_tokens_seq.tolist())})")

	return eval_loss, eval_top1_noun, eval_top1, eval_top1_seq, eval_tokens_total, num_valid_targets, num_samples, num_batches, elapsed_time

#
# Action: Evaluate classification dataset
#

# Action: Evaluate classification dataset
def action_eval_cls(cfg: omegaconf.DictConfig, hydra_dir: str, use_wandb: bool):

	hydra_name = os.path.basename(hydra_dir)

	model_paths = load_checkpoint_paths(cfg=cfg, hydra_dir=hydra_dir, allow_empty=True)
	if not model_paths:
		model_paths.append('')

	device, device_is_cpu, device_is_cuda = load_device(cfg=cfg)
	dataset_embedder = load_embedder(cfg=cfg, device=device)
	model_embedder = load_embedder(cfg=cfg, device=device)
	amp_context, amp_dtype = load_decoder_amp(cfg=cfg, device=device)

	with dataset_embedder.inference_model():

		dataset_noun = load_embedding_dataset(cfg=cfg, embedder=dataset_embedder, embed_dataset='NounDataset', use_targets=False, use_cache=False)
		dataset, loader, cls_variant, cls_clean = load_cls_dataset(cfg=cfg, embedder=dataset_embedder, device_is_cpu=device_is_cpu, paths=(use_paths := cfg.eval_images), path_optional=False)
		cls_class_lists, targets = align_cls_classes(cfg=cfg, embedder=dataset_embedder, dataset=dataset, dataset_noun=dataset_noun)
		gen_target_config(cfg=cfg, embedder=dataset_embedder, targets=targets, num_invalid_targets=0)
		data_config = gen_data_config(cfg=cfg, dataset=dataset_noun)
		del dataset_noun
		log.info("Unloaded noun dataset")

		num_images = len(dataset)
		log.info(f"Evaluating {loader.batch_size} samples per batch => {len(loader)} batches")
		log.info(f"Embedding the {num_images} images in the classification dataset...")
		if num_images > (1 << 21):
			log.warning(f"Current implementation loads entire dataset into memory and so may cause memory issues for giant datasets: {num_images} samples")

		paths_list = []
		embeds_list = []
		targets_list = []
		with tqdm.tqdm(desc='Embedding cls images', total=num_images, unit='img', unit_scale=False, dynamic_ncols=True, smoothing=0.08) as progress_bar:
			for data in loader:
				with dataset_embedder.inference_mode():
					embeds = dataset_embedder.inference_image(images=data[0])
				batch_embeds = embeds.shape[0]
				paths_list.append(data[2] if use_paths else [None] * batch_embeds)
				targets_list.append(tuple(cls_class_lists[index] for index in data[1].tolist()))
				embeds_list.append(embeds.cpu())
				progress_bar.update(n=batch_embeds)
			assert progress_bar.n == progress_bar.total
			assert len(paths_list) == len(embeds_list) == len(targets_list)

	log.info('-' * 80)
	for model_path in model_paths:

		eval_loss, eval_top1_noun, eval_direct, eval_guided, eval_validity, eval_result_ratios, num_samples, num_batches, elapsed_time = eval_cls_top1(
			cfg=cfg,
			hydra_name=hydra_name,
			targets=targets,
			num_images=num_images,
			paths_list=paths_list,
			embeds_list=embeds_list,
			targets_list=targets_list,
			model_path=model_path,
			data_config=data_config,
			device=device,
			device_is_cpu=device_is_cpu,
			amp_context=amp_context,
			dataset_embedder=dataset_embedder,
			model_embedder=model_embedder,
		)

		if use_wandb:
			eval_direct_correct, eval_direct_valid, eval_direct_invalid = eval_direct
			eval_guided_correct, eval_guided_incorrect = eval_guided
			eval_valid, eval_invalid = eval_validity
			wandb.log({'Eval/' + key: value for key, value in dict(
				dataset=dataset.cls_name,
				dataset_split=dataset.cls_split,
				model_path=model_path,
				model_dir=os.path.basename(os.path.dirname(model_path)),
				model_name=os.path.basename(model_path),
				loss=eval_loss,
				noun_top1=eval_top1_noun * 100,
				direct_correct=eval_direct_correct * 100,
				direct_valid=eval_direct_valid * 100,
				direct_invalid=eval_direct_invalid * 100,
				guided_correct=eval_guided_correct * 100,
				guided_incorrect=eval_guided_incorrect * 100,
				valid=eval_valid * 100,
				invalid=eval_invalid * 100,
				**{result: ratio * 100 for result, ratio in eval_result_ratios.items()},
				num_samples=num_samples,
				num_batches=num_batches,
				elapsed_time=elapsed_time,
			).items()})

		utils.release_cuda_memory(device=device)
		log.info('-' * 80)

# Evaluate the top-1 performance of a single model on a given classification dataset
def eval_cls_top1(
	cfg: omegaconf.DictConfig,
	hydra_name: str,
	targets: tuple[str, ...],
	num_images: int,
	paths_list: list[list[Optional[str]]],
	embeds_list: list[torch.Tensor],
	targets_list: list[tuple[list[str], ...]],
	model_path: str,
	data_config: embedding_dataset.DataConfig,
	device: torch.device,
	device_is_cpu: bool,
	amp_context: ContextManager,
	dataset_embedder: embedders.Embedder,
	model_embedder: embedders.Embedder,
) -> tuple[float, float, tuple[float, float, float], tuple[float, float], tuple[float, float], dict[str, float], int, int, float]:

	if model_path:
		log.info(f"Loading model: {model_path}")
		checkpoint, checkpoint_path = load_decoder_checkpoint(cfg=cfg, checkpoint_path=model_path)
		load_target_config(checkpoint=checkpoint, embedder=model_embedder)
		model_targets_set = set(model_embedder.target_vocab)
	else:
		log.warning(f"Evaluating randomly initialised model")
		checkpoint = None
		model_embedder.configure_target(target_config=dataset_embedder.target_config, target_vocab=dataset_embedder.target_vocab)
		model_targets_set = {'unknown'}  # Note: This doesn't quite match up with model_embedder, but is harmless and purely so that DudDecoder returning 'unknown' is considered a 'valid' target prediction

	eval_result = dict(DirectCorrectGuidedCorrect=[], DirectCorrectGuidedIncorrect=[], DirectValidGuidedCorrect=[], DirectValidGuidedIncorrect=[], DirectInvalidGuidedCorrect=[], DirectInvalidGuidedIncorrect=[])
	result_images_dir = None
	if cfg.eval_images:
		result_images = tuple(result for result in eval_result if cfg.eval_images in result)
		if result_images:
			log.info(f"Resolved specification '{cfg.eval_images}' of which evaluation images to copy to a directory: {', '.join(result_images)}")
			eval_images_dir = os.path.abspath(resolve_source_path(cfg.eval_images_dir))
			with contextlib.suppress(OSError):
				os.mkdir(eval_images_dir)
			image_dir = os.path.join(eval_images_dir, f"{hydra_name}-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
			os.mkdir(image_dir)
			log.info(f"Output image directory: {image_dir}")
			result_images_dir = {result: os.path.join(image_dir, result) for result in result_images}
			for result_path in result_images_dir.values():
				os.mkdir(result_path)
		else:
			log.warning(f"Failed to resolve specification '{cfg.eval_images}' of which evaluation images to copy")

	guide_token_ids = load_guide_targets(guide_targets=targets, embedder=model_embedder, device=device, device_is_cpu=device_is_cpu) if cfg.eval_guided else None

	with torch.inference_mode():

		model = prepare_decoder_model_eval(cfg=cfg, embedder=model_embedder, data_config=data_config, checkpoint=checkpoint, device=device, device_is_cpu=device_is_cpu, eval_train=cfg.eval_train)
		del checkpoint
		gc.collect()

		log.info("Evaluating top-1 performance of loaded model on classification dataset...")

		num_batches = 0
		num_samples = 0
		eval_samples_correct = 0
		eval_loss_sum_sum = 0.0
		eval_loss_basis_sum = 0.0

		with tqdm.tqdm(desc='Evaluating cls dataset', total=num_images, unit='img', unit_scale=False, dynamic_ncols=True, smoothing=0.08) as progress_bar:

			debug = cfg.eval_debug
			start_time = time.perf_counter()
			for paths, embeds, targets in zip(paths_list, embeds_list, targets_list):

				embeds = embeds.pin_memory().to(device=device, non_blocking=True)
				with amp_context:
					pred, _, _, batch_loss_sum, batch_loss_basis, _ = model.generate(embed=embeds, collect_logits=False, calc_loss=True, temperature=1, length_alpha=0, sample_weight=None, guide_targets=guide_token_ids, guide_renorm=False)
					pred_unguided = model.generate(embed=embeds, collect_logits=False, calc_loss=False, temperature=1, length_alpha=0, sample_weight=None, guide_targets=None, guide_renorm=False)[0] if guide_token_ids is not None else None

				batch_samples = embeds.shape[0]
				num_batches += 1
				num_samples += batch_samples

				pred_nouns = model_embedder.detokenize_target(pred.cpu())  # GPU-CPU synchronization point
				pred_correct = tuple(noun in target_list for noun, target_list in zip(pred_nouns, targets))
				eval_samples_correct += sum(pred_correct)
				eval_top1_noun = eval_samples_correct / num_samples

				eval_loss_sum_sum += batch_loss_sum.item()  # GPU-CPU synchronization point
				eval_loss_basis_sum += batch_loss_basis.item()  # GPU-CPU synchronization point
				eval_loss = eval_loss_sum_sum / eval_loss_basis_sum

				pred_valid = tuple(p in model_targets_set for p in pred_nouns)
				if guide_token_ids is None:
					for path, tgt_list, predict, predict_correct, predict_valid in zip(paths, targets, pred_nouns, pred_correct, pred_valid):
						if predict_correct:
							result = 'DirectCorrectGuidedCorrect'
						elif predict_valid:
							result = 'DirectValidGuidedIncorrect'
						else:
							result = 'DirectInvalidGuidedIncorrect'
						eval_result[result].append((path, tgt_list[0], predict))
						if result_images_dir and (result_dir := result_images_dir.get(result, None)):
							name, ext = os.path.splitext(os.path.basename(path))
							shutil.copy2(src=path, dst=os.path.join(result_dir, f'{name}---{tgt_list[0]}---{predict}{ext}'))
				else:
					pred_unguided_nouns = model_embedder.detokenize_target(pred_unguided.cpu())  # GPU-CPU synchronization point
					pred_unguided_correct = tuple(noun in target_list for noun, target_list in zip(pred_unguided_nouns, targets))
					pred_unguided_valid = tuple(p in model_targets_set for p in pred_unguided_nouns)
					for path, tgt_list, predict, predict_correct, predict_valid, predict_unguided, predict_unguided_correct, predict_unguided_valid in zip(paths, targets, pred_nouns, pred_correct, pred_valid, pred_unguided_nouns, pred_unguided_correct, pred_unguided_valid):
						if predict_unguided_correct:
							result = 'DirectCorrectGuidedCorrect' if predict_correct else 'DirectCorrectGuidedIncorrect'
						elif predict_unguided_valid:
							result = 'DirectValidGuidedCorrect' if predict_correct else 'DirectValidGuidedIncorrect'
						else:
							result = 'DirectInvalidGuidedCorrect' if predict_correct else 'DirectInvalidGuidedIncorrect'
						eval_result[result].append((path, tgt_list[0], predict, predict_unguided))
						if result_images_dir and (result_dir := result_images_dir.get(result, None)):
							name, ext = os.path.splitext(os.path.basename(path))
							shutil.copy2(src=path, dst=os.path.join(result_dir, f'{name}---{tgt_list[0]}---{predict_unguided}---{predict}{ext}'))

				if debug:
					if guide_token_ids is None:
						print('\n'.join('{color}{tgt} --> {predict}\033[0m'.format(color='\033[92m' if correct else '\033[91m', tgt=tgt_list[0], predict=predict) for correct, tgt_list, predict in zip(pred_correct, targets, pred_nouns)))
					else:
						print('\n'.join('{color}{tgt} --> {guided} ({unguided})\033[0m'.format(color='\033[92m' if correct else '\033[91m', tgt=tgt_list[0], unguided=unguided, guided=guided) for correct, tgt_list, guided, unguided in zip(pred_correct, targets, pred_nouns, pred_unguided_nouns)))
					debug = False

				progress_bar.set_postfix_str(f"loss={eval_loss:.2e}, noun={eval_top1_noun:.3%}", refresh=False)
				progress_bar.update(n=batch_samples)

			elapsed_time = time.perf_counter() - start_time
			assert progress_bar.n == progress_bar.total

	assert num_batches == len(paths_list) and num_samples == num_images
	log.info(f"Evaluated {num_batches} batches = {num_samples} samples")
	log.info(f"Evaluated 1 epoch in {elapsed_time:.1f}s to get loss {eval_loss:.2e}, noun top-1 {eval_top1_noun:.3%}")

	if eval_result['DirectCorrectGuidedIncorrect']:
		log.error(f"Unexpectedly found {len(eval_result['DirectCorrectGuidedIncorrect'])} samples where the direct output is correct but the guided output is incorrect")
	assert sum(len(items) for items in eval_result.values()) == num_samples and sum(len(items) for result, items in eval_result.items() if result.endswith('GuidedCorrect')) == eval_samples_correct
	eval_valid = sum(len(items) for result, items in eval_result.items() if not result.startswith('DirectInvalid')) / num_samples
	eval_invalid = sum(len(items) for result, items in eval_result.items() if result.startswith('DirectInvalid')) / num_samples
	eval_validity = (eval_valid, eval_invalid)
	log.info(f"Direct predictions are valid {eval_valid:.3%} of the time, and invalid non-nouns {eval_invalid:.3%} of the time")
	eval_direct_correct = (len(eval_result['DirectCorrectGuidedCorrect']) + len(eval_result['DirectCorrectGuidedIncorrect'])) / num_samples
	eval_direct_valid = (len(eval_result['DirectValidGuidedCorrect']) + len(eval_result['DirectValidGuidedIncorrect'])) / num_samples
	eval_direct_invalid = (len(eval_result['DirectInvalidGuidedCorrect']) + len(eval_result['DirectInvalidGuidedIncorrect'])) / num_samples
	eval_direct = (eval_direct_correct, eval_direct_valid, eval_direct_invalid)
	log.info(f"Direct predictions were {eval_direct_correct:.3%} correct, {eval_direct_valid:.3%} incorrect but valid nouns, {eval_direct_invalid:.3%} invalid nouns")
	eval_guided_correct = (len(eval_result['DirectCorrectGuidedCorrect']) + len(eval_result['DirectValidGuidedCorrect']) + len(eval_result['DirectInvalidGuidedCorrect'])) / num_samples
	eval_guided_incorrect = (len(eval_result['DirectCorrectGuidedIncorrect']) + len(eval_result['DirectValidGuidedIncorrect']) + len(eval_result['DirectInvalidGuidedIncorrect'])) / num_samples
	eval_guided = (eval_guided_correct, eval_guided_incorrect)
	log.info(f"{'Final' if guide_token_ids is None else 'Guided'} predictions were {eval_guided_correct:.3%} correct, {eval_guided_incorrect:.3%} incorrect")
	eval_result_ratios = {result: len(items) / num_samples for result, items in eval_result.items()}
	for result, ratio in eval_result_ratios.items():
		log.info(f"  {result} = {ratio:.3%}")

	return eval_loss, eval_top1_noun, eval_direct, eval_guided, eval_validity, eval_result_ratios, num_samples, num_batches, elapsed_time

#
# Action: Evaluate classification dataset decoding
#

# Action: Evaluate classification dataset decoding
def action_eval_cls_decoding(cfg: omegaconf.DictConfig, hydra_dir: str, use_wandb: bool):

	model_paths = load_checkpoint_paths(cfg=cfg, hydra_dir=hydra_dir, allow_empty=False)
	model_path_tails = tuple(os.path.join(os.path.basename(os.path.dirname(model_path)), os.path.basename(model_path)) for model_path in model_paths)
	gencfgs = load_generation_configs(cfg=cfg, guided=cfg.eval_guided)

	device, device_is_cpu, device_is_cuda = load_device(cfg=cfg)
	embedder = load_embedder(cfg=cfg, device=device)
	amp_context, amp_dtype = load_decoder_amp(cfg=cfg, device=device)
	data_config = embedding_dataset.DataConfig.create(data_config_dict=dict(use_weights=False, multi_target=False), use_targets=True)

	vocab_id_map = load_vocab_id_map(cfg=cfg, embedder=embedder)

	results_map = {}
	log.info('=' * 80)
	for cls_dataset in (cfg.cls_datasets or (cfg.cls_dataset,)):

		results = eval_cls_decoding(
			cfg=cfg,
			use_wandb=use_wandb,
			embedder=embedder,
			vocab_id_map=vocab_id_map,
			cls_dataset=cls_dataset,
			model_paths=model_paths,
			model_path_tails=model_path_tails,
			data_config=data_config,
			gencfgs=gencfgs,
			device=device,
			device_is_cpu=device_is_cpu,
			amp_context=amp_context,
		)

		if results is not None:

			if cls_dataset in results_map:
				log.warning(f"Ignoring duplicate data for {cls_dataset} dataset")
			else:
				results_map[cls_dataset] = results

		log.info('=' * 80)

	topk_tensor = torch.stack(tuple(results[2][:, :, :3] for results in results_map.values()), dim=0).permute(1, 2, 3, 0)  # GxMxKxD
	mean_topk_tensor = topk_tensor.mean(dim=1)  # GxKxD
	K = topk_tensor.shape[2]  # May be <3

	if use_wandb:
		table_rows = [[
			gencfg.name,
			*(correct * 100 if math.isfinite(correct) else None for corrects in mean_topk for correct in corrects),
		] for gencfg, mean_topk in zip(gencfgs, mean_topk_tensor)]
		if table_rows:
			table_rows = [row for order, row in sorted(zip((-mean_topk[0].mean() for mean_topk in mean_topk_tensor), table_rows))]
			wandb_table = wandb.Table(data=table_rows, columns=['Gen Cfg', *(f"Top-{k} {results[0].replace(' ', '/')} Mean" for k in range(1, K + 1) for results in results_map.values())])
			best_row = table_rows[0]
			best_data = tuple(tuple(best_row[1 + k * len(results_map) + i] for i in range(len(results_map))) for k in range(K))
			wandb.log(dict(
				table_overall=wandb_table,
				best_gen_cfg=best_row[0],
				**{f"best_top{k}_mean": None if any(value is None for value in topk) else sum(topk) / len(topk) for k, topk in enumerate(best_data, 1)},
				**{f"best_top{k}_{results[0].replace(' ', '_')}_mean": value for k, topk in enumerate(best_data, 1) for value, results in zip(topk, results_map.values())},
			))

	log.info(f"Top-1/2 {' + '.join(results_map.keys())} correct scores of all models:")
	table_rows = tuple((
		f"\033[36m{gencfg.name}\033[0m",
		*(' + '.join("{best_color}{correct}{best_uncolor}".format(
			best_color='\033[92m' if best_correct else '',
			correct=format_ratio_str(correct),
			best_uncolor='\033[0m' if best_correct else '',
		) for correct, best_correct in zip(mean_topk[k, :], best_mean_topk[k, :])) if k < gencfg.topk else None for k in range(2)),
		*(' < '.join(
			' + '.join("{best_color}{correct}{best_uncolor}".format(
				best_color='\033[92m' if best_correct else '',
				correct=format_ratio_str(correct),
				best_uncolor='\033[0m' if best_correct else '',
			) for correct, best_correct in zip(corrects, best_corrects)) for corrects, best_corrects, _ in zip(correct_topk, best_correct_topk, range(gencfg.topk))
		) for correct_topk, best_correct_topk in zip(topk[:, :2, :], best_topk[:, :2, :])),
	) for gencfg, topk, best_topk, mean_topk, best_mean_topk in zip(gencfgs, topk_tensor, topk_tensor.eq(topk_tensor.amax(dim=0)), mean_topk_tensor, mean_topk_tensor.eq(mean_topk_tensor.amax(dim=0))))
	table_rows = tuple(row for order, row in sorted(zip((-mean_topk[0].mean() for mean_topk in mean_topk_tensor), table_rows)))
	print(tabulate.tabulate(table_rows, headers=('\nGen Cfg', *(f'\nMean Top-{k}' for k in range(1, 3)), *("\033[35m{0}\033[0m".format(model_path_tail.replace('/', '/\033[0m\n\033[35m')) for model_path_tail in model_path_tails)), tablefmt='pretty', numalign='left', stralign='left'))

	log.info("Finished evaluating all decoding methods and models on all datasets")

# Evaluate decoding strategies on a single classification dataset with multiple models
def eval_cls_decoding(
	cfg: omegaconf.DictConfig,
	use_wandb: bool,
	embedder: embedders.Embedder,
	vocab_id_map: dict[str, set[int]],
	cls_dataset: str,
	model_paths: Sequence[str],
	model_path_tails: Sequence[str],
	data_config: embedding_dataset.DataConfig,
	gencfgs: Sequence[infer.GenerationConfig],
	device: torch.device,
	device_is_cpu: bool,
	amp_context: ContextManager,
) -> Optional[tuple[str, tuple[infer.GenerationConfig, ...], torch.Tensor, torch.Tensor]]:

	with embedder.inference_model():

		dataset, loader, cls_variant, cls_clean = load_cls_dataset(cfg=cfg, embedder=embedder, device_is_cpu=device_is_cpu, cls_dataset=cls_dataset, paths=True, path_optional=True)
		dataset_str = f"{dataset.cls_name} {dataset.cls_split} {cls_variant}"

		num_images = len(dataset)
		num_batches = len(loader)
		if num_images > cfg.eval_samples_max > 0:
			num_batches = min(max(cfg.eval_samples_max // loader.batch_size, 1), num_batches)
			num_images = num_batches * loader.batch_size
			log.warning(f"Limiting evaluation to at most {cfg.eval_samples_max} samples => Using {num_batches} batches = {num_images} samples only")

		log.info(f"Evaluating {loader.batch_size} samples per batch => {num_batches} batches")
		log.info(f"Embedding the {num_images} images in the classification dataset...")
		if num_images > (1 << 21):
			log.warning(f"Current implementation loads entire dataset into memory and so may cause memory issues for giant datasets: {num_images} samples")

		dataset_batches = []
		with tqdm.tqdm(desc='Embedding cls images', total=num_images, unit='img', unit_scale=False, dynamic_ncols=True, smoothing=0.08) as progress_bar:
			for images, targets, paths in itertools.islice(loader, num_batches):
				with embedder.inference_mode():
					embeds = embedder.inference_image(images=images)
				dataset_batches.append((embeds := embeds.cpu(), targets.tolist(), paths))
				progress_bar.update(n=embeds.shape[0])
			assert progress_bar.n == progress_bar.total

	if not dataset_batches:
		log.warning("No dataset batches to process!")
		return None

	results_map = {}
	log.info('-' * 80)
	for model_path, model_path_tail in zip(model_paths, model_path_tails):

		results = eval_cls_decoding_single(
			cfg=cfg,
			use_wandb=use_wandb,
			embedder=embedder,
			vocab_id_map=vocab_id_map,
			model_path=model_path,
			model_path_tail=model_path_tail,
			dataset_str=dataset_str,
			num_images=num_images,
			cls_classes=dataset.cls_classes,
			dataset_batches=dataset_batches,
			data_config=data_config,
			gencfgs=gencfgs,
			device=device,
			device_is_cpu=device_is_cpu,
			amp_context=amp_context,
		)

		model_spec = (model_path, model_path_tail)
		results_map[model_spec] = results

		utils.release_cuda_memory(device=device)
		log.info('-' * 80)

	table_data = {}
	for model_spec, results in results_map.items():
		model_path, model_path_tail = model_spec
		for gencfg, topk, topk_guide, topk_vocab, topk_invalid in results:
			if (model_perf := table_data.get(gencfg, None)) is None:
				table_data[gencfg] = (model_perf := {})
			if model_spec in model_perf:
				log.warning(f"Ignoring duplicate data for generation configuration {gencfg.name} and model {model_path_tail}")
			else:
				model_perf[model_spec] = topk[:3]

	topk_tensor = torch.stack(tuple(torch.nn.utils.rnn.pad_sequence([model_perf[model_spec] for model_perf in table_data.values()], batch_first=True, padding_value=-torch.inf) for model_spec in results_map), dim=0).transpose(0, 1)
	mean_topk_tensor = topk_tensor.mean(dim=1)
	K = topk_tensor.shape[2]

	if use_wandb:
		table_rows = [[
			dataset_str.replace(' ', '/'),
			gencfg.name,
			*(correct * 100 if math.isfinite(correct) else None for correct in mean_corrects),
		] for gencfg, mean_corrects in zip(table_data.keys(), mean_topk_tensor)]
		if table_rows:
			table_rows = [row for order, row in sorted(zip((-mean_corrects[0] for mean_corrects in mean_topk_tensor), table_rows))]
			wandb_table = wandb.Table(data=table_rows, columns=['Dataset', 'Gen Cfg', *(f'Top-{k} Mean' for k in range(1, K + 1))])
			best_row = table_rows[0]
			dataset_str_ = dataset_str.replace(' ', '_')
			wandb.log({
				'table_per_dataset': wandb_table,
				f'{dataset_str_}/best_gen_cfg': best_row[1],
				**{f'{dataset_str_}/best_top{k}_mean': topk for k, topk in enumerate(itertools.islice(best_row, 2, None), 1)}
			})

	log.info(f"Top-1 < Top-2 < Top-3 correct scores of all models on {dataset_str}:")
	table_rows = tuple((
		f"\033[36m{gencfg.name}\033[0m",
		' < '.join("{best_color}{correct}{best_uncolor}".format(
			best_color='\033[92m' if best_correct else '',
			correct=format_ratio_str(correct),
			best_uncolor='\033[0m' if best_correct else '',
		) for correct, best_correct in itertools.islice(zip(mean_corrects, best_mean_corrects), mean_corrects.isfinite().sum())),
		*(' < '.join("{best_color}{correct}{best_uncolor}".format(
			best_color='\033[92m' if best_correct else '',
			correct=format_ratio_str(correct),
			best_uncolor='\033[0m' if best_correct else '',
		) for correct, best_correct in zip(model_perf[model_spec], best_model_perfs)) for model_spec, best_model_perfs in zip(results_map, best_results)),
	) for (gencfg, model_perf), best_results, mean_corrects, best_mean_corrects in zip(table_data.items(), topk_tensor.eq(topk_tensor.amax(dim=0)), mean_topk_tensor, mean_topk_tensor.eq(mean_topk_tensor.amax(dim=0))))
	table_rows = tuple(row for order, row in sorted(zip((-mean_corrects[0] for mean_corrects in mean_topk_tensor), table_rows)))
	print(tabulate.tabulate(table_rows, headers=('\nGen Cfg', '\nMean Top-k', *("\033[35m{0}\033[0m".format(model_spec[1].replace('/', '/\033[0m\n\033[35m')) for model_spec in results_map)), tablefmt='pretty', numalign='left', stralign='left'))

	log.info(f"Finished evaluating dataset: {dataset_str}")
	return dataset_str, tuple(table_data.keys()), topk_tensor, mean_topk_tensor

# Evaluate decoding strategies on a single classification dataset with a single model
def eval_cls_decoding_single(
	cfg: omegaconf.DictConfig,
	use_wandb: bool,
	embedder: embedders.Embedder,
	vocab_id_map: dict[str, set[int]],
	model_path: str,
	model_path_tail: str,
	dataset_str: str,
	num_images: int,
	cls_classes: Sequence[str],
	dataset_batches: list[tuple[torch.Tensor, list[int], Optional[list[str]]]],
	data_config: embedding_dataset.DataConfig,
	gencfgs: Sequence[infer.GenerationConfig],
	device: torch.device,
	device_is_cpu: bool,
	amp_context: ContextManager,
) -> tuple[tuple[infer.GenerationConfig, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], ...]:

	checkpoint, checkpoint_path, model_targets, model_targets_set, vocab_targets = load_decoder_checkpoint_generate(cfg=cfg, embedder=embedder, model_path=model_path, device=device, device_is_cpu=device_is_cpu)

	cls_class_lists, cls_targets = align_cls_class_targets(cfg=cfg, embedder=embedder, cls_classes=cls_classes, targets=model_targets, vocab_id_map=vocab_id_map)
	cls_targets_set = set(cls_targets)
	cls_main_classes = tuple(class_list[0] if class_list else 'UNKNOWN' for class_list in cls_class_lists)
	guide_targets = load_guide_targets(guide_targets=cls_targets, embedder=embedder, device=device, device_is_cpu=device_is_cpu)

	with torch.inference_mode():

		model = prepare_decoder_model_eval(cfg=cfg, embedder=embedder, data_config=data_config, checkpoint=checkpoint, device=device, device_is_cpu=device_is_cpu, eval_train=cfg.eval_train)
		del checkpoint
		gc.collect()

		max_topk = max(gencfg.topk for gencfg in gencfgs)
		gen_task_list = GenerationTaskList(gencfgs=gencfgs, model=model, vocab_targets_set=model_targets_set, vocab_targets=vocab_targets, guide_targets_set=cls_targets_set, guide_targets=guide_targets, class_lists=cls_class_lists)

		log.info("Evaluating performance of loaded model on classification dataset with multiple decoding strategies...")

		num_batches = 0
		num_samples = 0
		best_task = None

		with tqdm.tqdm(desc='Evaluating cls dataset', total=num_images, unit='img', unit_scale=False, dynamic_ncols=True, smoothing=0.08) as progress_bar:

			def update_tqdm_postfix():
				progress_bar.set_postfix_str(f"task {i}/{len(gen_task_list)}, best={best_task and best_task.gencfg.name}, top1={best_task.topk[0] if best_task else 0:.3%}")

			debug = cfg.eval_debug
			start_time = time.perf_counter()
			for embeds, targets, paths in dataset_batches:

				embeds = embeds.pin_memory().to(device=device, non_blocking=True)
				batch_samples = embeds.shape[0]
				num_batches += 1
				num_samples += batch_samples

				with amp_context:
					for i, task in gen_task_list.iter_generate(embeds=embeds, targets=targets):
						pass  # Note: update_tqdm_postfix() here is useful for better progress bar stats but is not kind to wandb console logs (it tends to lose data)
					best_task = max(gen_task_list, key=lambda tsk: tsk.topk[0])
					update_tqdm_postfix()

				if debug:
					if paths:
						log.info(f"Full path of first sample: {paths[0]}")
					log.info(f"Predictions on {dataset_str} of {model_path_tail}:")
					table_rows = tuple((
						task.gencfg.name,
						f"\033[36m{cls_main_classes[target]}\033[0m",
						*("{color}{noun}\033[0m = {score:.3g}".format(color=GenerationTask.COLOR_MAP[result], noun=noun[:22], score=score) for result, noun, score in zip(task.result[i], task.target_str[i], task.target_score[i])),
						*(None for _ in range(max_topk - task.gencfg.topk)),
						'None' if paths is None else os.path.join(os.path.basename(os.path.dirname(paths[i])), os.path.basename(paths[i])),
					) for i, target in enumerate(targets) for task in gen_task_list)
					with tqdm.tqdm.external_write_mode():
						print(tabulate.tabulate(table_rows, headers=('Gen Cfg', 'Label', *(f'Prediction {k}' for k in range(1, max_topk + 1)), 'Path'), tablefmt='pretty', numalign='left', stralign='left'))
					debug = False

				progress_bar.update(n=batch_samples)

			elapsed_time = time.perf_counter() - start_time
			assert progress_bar.n == progress_bar.total

	assert num_batches == len(dataset_batches) and num_samples == num_images
	log.info(f"Evaluated {num_batches} batches = {num_samples} samples in {elapsed_time:.1f}s")

	if use_wandb:
		table_rows = [[
			dataset_str.replace(' ', '/'),
			model_path_tail,
			task.gencfg.name,
			*(correct and correct * 100 for correct, _ in itertools.zip_longest(task.topk.tolist(), range(max_topk))),
			*(invalid and invalid * 100 for invalid, _ in itertools.zip_longest(task.topk_invalid.tolist(), range(max_topk))),
		] for task in gen_task_list]
		if table_rows:
			table_rows = [row for order, row in sorted(zip((-task.topk[0] for task in gen_task_list), table_rows))]
			wandb.log(dict(table_per_model=wandb.Table(data=table_rows, columns=['Dataset', 'Model', 'Gen Cfg', *(f'Top-{k}' for k in range(1, max_topk + 1)), *(f'ITop-{k}' for k in range(1, max_topk + 1))])))

	log.info(f"Top-k correct ~ invalid scores on {dataset_str} of {model_path_tail}:")
	all_topk = torch.nn.utils.rnn.pad_sequence([task.topk for task in gen_task_list], batch_first=True, padding_value=-torch.inf)
	all_topk_invalid = torch.nn.utils.rnn.pad_sequence([task.topk_invalid for task in gen_task_list], batch_first=True, padding_value=-torch.inf)
	table_rows = tuple((
		f"\033[36m{task.gencfg.name}\033[0m",
		*("{best_color}{correct}{best_uncolor} ~ {worst_color}{invalid}%{worst_uncolor}".format(
			best_color='\033[92m' if best_correct else '',
			correct=format_ratio_str(correct),
			best_uncolor='\033[0m' if best_correct else '',
			worst_color='\033[91m' if worst_invalid else '',
			invalid=utils.format_semifix(invalid, precision=3),
			worst_uncolor='\033[0m' if worst_invalid else '',
		) for correct, best_correct, invalid, worst_invalid in zip(task.topk, best_topk, task.topk_invalid, worst_topk_invalid)),
	) for task, best_topk, worst_topk_invalid in zip(gen_task_list, all_topk.eq(all_topk.amax(dim=0)), all_topk_invalid.eq(all_topk_invalid.amax(dim=0))))
	table_rows = tuple(row for order, row in sorted(zip((-task.topk[0] for task in gen_task_list), table_rows)))
	print(tabulate.tabulate(table_rows, headers=('Gen Cfg', *(f'Top-{k}' for k in range(1, max_topk + 1))), tablefmt='pretty', numalign='left', stralign='left'))

	log.info(f"Finished evaluating model: {model_path}")
	return tuple((task.gencfg, task.topk, task.topk_guide, task.topk_vocab, task.topk_invalid) for task in gen_task_list)

#
# Action: Inference
#

# Action: Inference
def action_infer(cfg: omegaconf.DictConfig, hydra_dir: str, use_wandb: bool):

	model_paths = load_checkpoint_paths(cfg=cfg, hydra_dir=hydra_dir, allow_empty=False)
	model_path_tails = tuple(os.path.join(os.path.basename(os.path.dirname(model_path)), os.path.basename(model_path)) for model_path in model_paths)
	gencfgs = load_generation_configs(cfg=cfg, guided=cfg.infer_guided)

	device, device_is_cpu, device_is_cuda = load_device(cfg=cfg)
	embedder = load_embedder(cfg=cfg, device=device)
	amp_context, amp_dtype = load_decoder_amp(cfg=cfg, device=device)
	data_config = embedding_dataset.DataConfig.create(data_config_dict=dict(use_weights=False, multi_target=False), use_targets=True)

	if cfg.infer_guide_targets:
		infer_targets = tuple(cfg.infer_guide_targets)
	elif cfg.infer_guide_dataset:
		log.info("Loading guiding dataset...")
		dataset = load_embedding_dataset(cfg=cfg, embedder=embedder, embed_dataset=cfg.infer_guide_dataset, use_targets=True, training=False, strict_embedder=cfg.strict_embedder)
		infer_targets = dataset.targets[dataset.num_invalid_targets:]
		del dataset
		log.info(f"Unloaded guiding dataset after obtaining {len(infer_targets)} guiding target nouns from it")
	else:
		infer_targets = None  # Note: Will use model vocabulary as infer targets

	data = load_infer_data(cfg=cfg, embedder=embedder)
	if not data:
		log.warning("Inference has nothing to do")
		return

	log.info('-' * 80)
	for model_path, model_path_tail in zip(model_paths, model_path_tails):

		infer_model(
			cfg=cfg,
			hydra_dir=hydra_dir,
			use_wandb=use_wandb,
			embedder=embedder,
			data=data,
			model_path=model_path,
			model_path_tail=model_path_tail,
			data_config=data_config,
			gencfgs=gencfgs,
			infer_targets=infer_targets,
			device=device,
			device_is_cpu=device_is_cpu,
			amp_context=amp_context,
		)

		utils.release_cuda_memory(device=device)
		log.info(f"Finished inference for model: {model_path_tail}")
		log.info('-' * 80)

# Load and embed texts and images for inference
def load_infer_data(cfg: omegaconf.DictConfig, embedder: embedders.Embedder) -> tuple[tuple[list[str], torch.Tensor], ...]:

	data_keys_raw = []
	data_embeds_raw = []

	with embedder.inference_model(), embedder.inference_mode():

		if cfg.infer_texts:
			infer_texts = (cfg.infer_texts,) if isinstance(cfg.infer_texts, str) else cfg.infer_texts
			log.info(f"Embedding {len(infer_texts)} texts for inference...")
			infer_texts_iter = iter(infer_texts)
			while texts := tuple(itertools.islice(infer_texts_iter, embedder.inference_batch_size)):
				data_keys_raw.extend(f'"{text}"' for text in texts)
				data_embeds_raw.append(embedder.inference_text(text=texts))
			if cfg.infer_ann_json_update:
				load_sample_annotations(ann_json=cfg.infer_ann_json, image_dir=None, update_samples=infer_texts)
		else:
			infer_texts = ()

		if cfg.infer_images:
			infer_images = (cfg.infer_images,) if isinstance(cfg.infer_images, str) else cfg.infer_images
			log.info(f"Embedding {len(infer_images)} images for inference...")
			image_transform = embedder.get_image_transform()
			infer_images_iter = iter(infer_images)
			image_dir = resolve_source_path(cfg.infer_image_dir)
			while image_paths := tuple(itertools.islice(infer_images_iter, embedder.image_batch_size)):
				data_keys_raw.extend(image_paths)
				data_embeds_raw.append(embedder.inference_image(torch.utils.data.default_collate(tuple(image_transform(PIL.Image.open(os.path.join(image_dir, image_path))) for image_path in image_paths))))
			if cfg.infer_ann_json_update:
				load_sample_annotations(ann_json=cfg.infer_ann_json, image_dir=image_dir, update_samples=infer_images)
		else:
			infer_images = ()

		if cfg.infer_all_images_dir:
			infer_all_images_dir = resolve_source_path(cfg.infer_all_images_dir)
			log.info(f"Collecting all image files in directory: {infer_all_images_dir}")
			filenames = os.listdir(infer_all_images_dir)
			infer_all_images = tuple(sorted(set().union(*([filename for filename in filenames if fnmatch.fnmatch(name=filename.lower(), pat=pattern)] for pattern in utils.IMAGE_PATTERNS))))
			log.info(f"Found {len(infer_all_images)} images and {len(filenames) - len(infer_all_images)} non-images in the directory of {len(filenames)} entries")
			log.info(f"Embedding {len(infer_all_images)} images for inference...")
			image_transform = embedder.get_image_transform()
			infer_all_images_iter = iter(infer_all_images)
			while image_paths := tuple(itertools.islice(infer_all_images_iter, embedder.image_batch_size)):
				data_keys_raw.extend(image_paths)
				data_embeds_raw.append(embedder.inference_image(torch.utils.data.default_collate(tuple(image_transform(PIL.Image.open(os.path.join(infer_all_images_dir, image_path))) for image_path in image_paths))))
			if cfg.infer_ann_json_update:
				load_sample_annotations(ann_json=cfg.infer_ann_json, image_dir=infer_all_images_dir, update_samples=infer_all_images)
		else:
			infer_all_images = ()

	num_samples = len(infer_texts) + len(infer_images) + len(infer_all_images)
	assert num_samples == len(data_keys_raw) == sum(embeds.shape[0] for embeds in data_embeds_raw)
	log.info(f"Finished generating embeddings for {len(data_keys_raw)} input texts/images")

	i = j = 0
	data_keys = []
	data_embeds = []
	text_index = 0
	pending_embeds = []
	pending_free = cfg.batch_size

	while i < len(data_embeds_raw):
		embed_chunk = data_embeds_raw[i]
		chunk = embed_chunk[j:] if j > 0 else embed_chunk
		chunk_size = chunk.shape[0]
		if chunk_size >= pending_free:
			pending_embeds.append(chunk[:pending_free])
			pending_embed = torch.concat(pending_embeds, dim=0) if len(pending_embeds) > 1 else pending_embeds[0]
			data_embeds.append(pending_embed)
			new_text_index = text_index + pending_embed.shape[0]
			data_keys.append(data_keys_raw[text_index:new_text_index])
			text_index = new_text_index
			if chunk_size == pending_free:
				i += 1
				j = 0
			else:
				j += pending_free
			pending_embeds.clear()
			pending_free = cfg.batch_size
		else:
			pending_embeds.append(chunk)
			pending_free -= chunk_size
			i += 1
			j = 0

	if pending_embeds:
		pending_embed = torch.concat(pending_embeds, dim=0) if len(pending_embeds) > 1 else pending_embeds[0]
		data_embeds.append(pending_embed)
		data_keys.append(data_keys_raw[text_index:text_index + pending_embed.shape[0]])

	assert (data_keys_lens := tuple(len(keys) for keys in data_keys)) == tuple(embeds.shape[0] for embeds in data_embeds) and sum(data_keys_lens) == num_samples
	assert all(keys_len == cfg.batch_size for keys_len in data_keys_lens[:-1]) and (not data_keys_lens or 1 <= data_keys_lens[-1] <= cfg.batch_size)
	assert data_keys_raw == list(itertools.chain.from_iterable(data_keys))
	log.info(f"Reorganised generated embeddings into {len(data_keys)} batches suitable for decoding")

	return tuple(zip(data_keys, data_embeds))

# Inference on key-embedding data
def infer_model(
	cfg: omegaconf.DictConfig,
	hydra_dir: str,
	use_wandb: bool,
	embedder: embedders.Embedder,
	data: tuple[tuple[list[str], torch.Tensor], ...],
	model_path: str,
	model_path_tail: str,
	data_config: embedding_dataset.DataConfig,
	gencfgs: tuple[infer.GenerationConfig, ...],
	infer_targets: Optional[tuple[str, ...]],
	device: torch.device,
	device_is_cpu: bool,
	amp_context: ContextManager,
):

	checkpoint, checkpoint_path, model_targets, model_targets_set, vocab_targets = load_decoder_checkpoint_generate(cfg=cfg, embedder=embedder, model_path=model_path, device=device, device_is_cpu=device_is_cpu)

	if infer_targets is None:
		infer_targets = model_targets
	if not ((infer_targets_set := set(infer_targets)) <= model_targets_set):
		log.warning(f"Some guide target nouns are not in the set of trained model target nouns: {', '.join(sorted(infer_targets_set - model_targets_set))}")
	guide_targets = load_guide_targets(guide_targets=infer_targets, embedder=embedder, device=device, device_is_cpu=device_is_cpu)

	with torch.inference_mode():

		model = prepare_decoder_model_eval(cfg=cfg, embedder=embedder, data_config=data_config, checkpoint=checkpoint, device=device, device_is_cpu=device_is_cpu, eval_train=False)
		model_cfg_flat = checkpoint['cfg_flat']
		del checkpoint
		gc.collect()

		samples = tuple(key for keys, embeds in data for key in keys)
		gen_task_list = GenerationTaskList(gencfgs=gencfgs, model=model, vocab_targets_set=model_targets_set, vocab_targets=vocab_targets, guide_targets_set=infer_targets_set, guide_targets=guide_targets)

		log.info("Inferencing loaded model on the generated embeddings...")

		num_batches = 0
		num_samples = 0
		predictions = {task: {} for task in gen_task_list}

		start_time = time.perf_counter()
		with amp_context:
			for keys, embeds in data:
				num_batches += 1
				num_samples += embeds.shape[0]
				gen_task_list.generate(embeds=embeds)
				for task, preds in predictions.items():
					for key, target_list, score_list, result_list in zip(keys, task.target_str, task.target_score, task.result.tolist(), strict=True):
						preds[key] = tuple((' '.join(target.split()), score, result) for target, score, result in zip(target_list, score_list, result_list))
			assert all(len(preds) == num_samples and tuple(preds.keys()) == samples and all(len(topk_list) == task.gencfg.topk for topk_list in preds.values()) for task, preds in predictions.items())

		elapsed_time = time.perf_counter() - start_time
		log.info(f"Inferenced {num_samples} samples in {num_batches} batches for {len(gen_task_list)} generation configs in {elapsed_time:.2f}s")

		log.info('\xB7' * 80)
		for task, preds in predictions.items():

			log.info(f"Generation config: {task.gencfg.name}")

			if cfg.infer_debug:
				log.info(f"Predictions of {model_path_tail} using {task.gencfg.name}:")
				table_rows = tuple((
					key,
					*("{color}{pred}\033[0m = {score:.3g}".format(color=GenerationTask.COLOR_MAP[result], pred=pred, score=score) for pred, score, result in topk_list),
				) for key, topk_list in preds.items())
				print(tabulate.tabulate(table_rows, headers=('Sample', *(f'Prediction {k}' for k in range(1, task.gencfg.topk + 1))), tablefmt='pretty', numalign='left', stralign='left'))

			if cfg.infer_log:
				for key, topk_list in preds.items():
					log.info(f"{key} --> {topk_list[0][0]}")

			if use_wandb:
				wandb.log({'Eval/' + key: value for key, value in dict(
					dataset='infer',
					model_path=model_path,
					model_dir=os.path.basename(os.path.dirname(model_path)),
					model_name=os.path.basename(model_path),
					gen_cfg=task.gencfg.name,
					valid_guide=task.topk_guide[0] * 100,
					valid_vocab=task.topk_vocab[0] * 100,
					valid=task.topk_valid[0] * 100,
					invalid=task.topk_invalid[0] * 100,
					num_samples=num_samples,
					num_batches=num_batches,
				).items()})

			log.info(f"Top-1 predictions: {task.topk_guide[0]:.3%} are valid guide nouns, {task.topk_vocab[0]:.3%} are valid model vocab, {task.topk_invalid[0]:.3%} are invalid, {task.topk_valid[0]:.3%} overall are valid")
			log.info('\xB7' * 80)

	if use_wandb:
		table_rows = [[model_path_tail, key, *(preds[key][0][0] for preds in predictions.values())] for key in samples]
		wandb.log(dict(table_per_model=wandb.Table(data=table_rows, columns=['Model', 'Sample', *(task.gencfg.name for task in predictions.keys())])))

	if cfg.infer_pred_json:
		pred_json_path = os.path.join(hydra_dir, f"infer_{model_path_tail.replace('/', '_').replace('.', '_')}.json")
		with open(pred_json_path, 'w') as file:
			utils.json_dump(dict(
				version=1,
				model=model_path_tail,
				model_path=model_path,
				model_dir=os.path.basename(os.path.dirname(model_path)),
				model_name=os.path.basename(model_path),
				model_cfg=model_cfg_flat,
				infer_cfg=utils_config.flatten_config(cfg),
				guide_targets=sorted(infer_targets_set),
				vocab_targets=sorted(model_targets_set),
				samples=samples,
				predictions={task.gencfg.name: dict(
					gen_cfg=dataclasses.asdict(task.gencfg),
					valid_guide=(task.topk_guide * 100).tolist(),
					valid_vocab=(task.topk_vocab * 100).tolist(),
					valid=(task.topk_valid * 100).tolist(),
					invalid=(task.topk_invalid * 100).tolist(),
					pred=tuple(tuple(item[0] for item in topk_list) for topk_list in preds.values()),
					score=tuple(tuple(item[1] for item in topk_list) for topk_list in preds.values()),
					result=tuple(tuple(item[2] for item in topk_list) for topk_list in preds.values()),  # Note: Result is never 0 (correct) as no ground truth annotations are available
				) for task, preds in predictions.items()},
			), file, indent=2)
		log.info(f"Saved predictions JSON: {pred_json_path}")

#
# Action: Format predictions data
#

# Action: Format predictions data
def action_format_preds(cfg: omegaconf.DictConfig):

	pred_jsons = load_predictions(cfg=cfg)
	class_annotations, categories = load_sample_annotations(ann_json=cfg.pred_ann_json, image_dir=cfg.pred_image_dir)
	correct_targets = {sample: set().union(ann.get('correct_primary', ()), ann.get('correct_secondary', ())) for sample, ann in class_annotations.items()}

	pfmt_type = cfg.pfmt_type.lower()
	if pfmt_type == 'nouns_v1':
		format_nouns_v1(cfg=cfg, pred_jsons=pred_jsons, correct_targets=correct_targets)
	elif pfmt_type == 'model_topk_v1':
		format_model_topk_v1(cfg=cfg, pred_jsons=pred_jsons, class_annotations=class_annotations, categories=categories)
	elif pfmt_type == 'model_max_v1':
		format_model_max_v1(cfg=cfg, pred_jsons=pred_jsons, class_annotations=class_annotations, categories=categories)
	elif pfmt_type == 'gencfg_model_v1':
		format_gencfg_model_v1(cfg=cfg, pred_jsons=pred_jsons, class_annotations=class_annotations, categories=categories)
	else:
		raise ValueError(f"Unsupported predictions format type: {cfg.pfmt_type}")

# Format table of noun predictions (version 1)
def format_nouns_v1(cfg: omegaconf.DictConfig, pred_jsons: dict[str, dict[str, Any]], correct_targets: dict[str, set[str]]):

	K = cfg.pfmt_topk
	compare_pred_jsons(pred_jsons=pred_jsons)

	for path, pred_json in pred_jsons.items():
		file = os.path.basename(path)
		for gencfg, predictions in pred_json['predictions'].items():
			log.info(f"Top-{min(K, predictions['gen_cfg']['topk'])} predictions for {file} decoded with {gencfg}:")
			table_rows = []
			for sample, preds, scores, results in zip(pred_json['samples'], predictions['pred'], predictions['score'], predictions['result'], strict=True):
				correct = correct_targets.get(sample, None)
				table_rows.append((sample, *("{color}{pred}\033[0m = {score:.3g}".format(color=GenerationTask.COLOR_MAP[0 if correct is not None and pred in correct else result], pred=pred, score=score) for pred, score, result, _ in zip(preds, scores, results, range(K)))))
			print(tabulate.tabulate(table_rows, headers=('Sample', *(f'Prediction {k}' for k in range(1, K + 1))), tablefmt='pretty', numalign='left', stralign='left'))

# Format table of model top-k results (version 1)
def format_model_topk_v1(cfg: omegaconf.DictConfig, pred_jsons: dict[str, dict[str, Any]], class_annotations: dict[str, dict[str, set[str]]], categories: dict[str, None]):

	K = cfg.pfmt_topk
	compare_pred_jsons(pred_jsons=pred_jsons)
	scorer = PredictionScorer(class_annotations=class_annotations, categories=categories)
	score_data, totals = calc_prediction_scores(pred_jsons=pred_jsons, scorer=scorer, topk=K)

	for gencfg, gencfg_score_data in sorted(score_data.items()):
		log.info(f"Top-k (up to {K}) prediction scores when models are decoded with {gencfg}:")
		topk = max(len(topk_scores) for topk_scores, topk_counters in gencfg_score_data.values())
		table_headers = ('Model' if cfg.pfmt_model_spec else 'JSON', 'Top-1 Details', *(f'Top-{k}' for k in range(1, topk + 1)), *(f'Top-{k}%' for k in range(1, topk + 1)))
		table_rows = [(
			pred_jsons[path]['model'] if cfg.pfmt_model_spec else os.path.basename(path),
			scorer.format_counter(counter=topk_counters[0]),
			*(scorer.format_score(score=score, total=counter.total()) for score, counter in zip(topk_scores, topk_counters, strict=True)),
			*(scorer.format_score_pct(score=score, total=counter.total()) for score, counter in zip(topk_scores, topk_counters, strict=True)),
		) for path, (topk_scores, topk_counters) in gencfg_score_data.items()]
		if cfg.pfmt_sort:
			sort_table_rows(table_rows=table_rows, default_order=None, table_headers=table_headers, sort_spec=cfg.pfmt_sort)
		print(tabulate.tabulate(table_rows, headers=table_headers, tablefmt='pretty', numalign='left', stralign='left'))

# Format table of maximum model results (version 1)
def format_model_max_v1(cfg: omegaconf.DictConfig, pred_jsons: dict[str, dict[str, Any]], class_annotations: dict[str, dict[str, set[str]]], categories: dict[str, None]):

	compare_pred_jsons(pred_jsons=pred_jsons)
	scorer = PredictionScorer(class_annotations=class_annotations, categories=categories)
	score_data, totals = calc_prediction_scores(pred_jsons=pred_jsons, scorer=scorer, topk=1)

	max_score_data = {path: max(((gencfg_score_data[path][0][0], gencfg_score_data[path][1][0]) for gencfg_score_data in score_data.values() if path in gencfg_score_data), key=operator.itemgetter(0), default=-math.inf) for path in pred_jsons}

	log.info(f"Maximum top-1 prediction scores per-model across decoding strategies:")
	table_headers = ('Model' if cfg.pfmt_model_spec else 'JSON', 'Top-1 Details', 'Top-1', 'Top-1%')
	table_rows = [(
		pred_jsons[path]['model'] if cfg.pfmt_model_spec else os.path.basename(path),
		scorer.format_counter(counter=counter),
		scorer.format_score(score=score, total=counter.total()),
		scorer.format_score_pct(score=score, total=counter.total()),
	) for path, (score, counter) in max_score_data.items()]
	sort_table_rows(table_rows=table_rows, default_order=tuple(-score / counter.total() for score, counter in max_score_data.values()), table_headers=table_headers, sort_spec=None)
	print(tabulate.tabulate(table_rows, headers=table_headers, tablefmt='pretty', numalign='left', stralign='left'))

# Format table of generation configuration results (version 1)
def format_gencfg_model_v1(cfg: omegaconf.DictConfig, pred_jsons: dict[str, dict[str, Any]], class_annotations: dict[str, dict[str, set[str]]], categories: dict[str, None]):

	K = cfg.pfmt_topk
	compare_pred_jsons(pred_jsons=pred_jsons)
	scorer = PredictionScorer(class_annotations=class_annotations, categories=categories)
	score_data, totals = calc_prediction_scores(pred_jsons=pred_jsons, scorer=scorer, topk=K)
	K = max(len(topk_scores) for gencfg_score_data in score_data.values() for topk_scores, topk_counters in gencfg_score_data.values())

	mean_score_data = {}
	for gencfg, gencfg_score_data in score_data.items():
		topk_scores = tuple(tuple(gencfg_score_data[path][0][k] for path in pred_jsons if path in gencfg_score_data and len(gencfg_score_data[path][0]) > k) for k in range(K))
		topk_scores_pct = tuple(tuple(gencfg_score_data[path][0][k] / totals[path] for path in pred_jsons if path in gencfg_score_data and len(gencfg_score_data[path][0]) > k) for k in range(K))
		mean_scores = tuple(sum(scores) / len(scores) if scores else None for scores in topk_scores)
		mean_scores_pct = tuple(sum(scores) / len(scores) if scores else None for scores in topk_scores_pct)
		mean_score_data[gencfg] = (mean_scores, mean_scores_pct)

	max_score_data = {path: tuple(max(topk_data, key=operator.itemgetter(0), default=-math.inf) for topk_data in itertools.zip_longest(*(zip(*gencfg_score_data[path], strict=True) for gencfg_score_data in score_data.values() if path in gencfg_score_data), fillvalue=(-math.inf, None))) for path in pred_jsons}
	max_mean_scores: tuple[Optional[float]] = tuple(max((score for score in scores if score is not None), default=None) for scores in zip(*(mean_scores for mean_scores, mean_scores_pct in mean_score_data.values()), strict=True))  # noqa
	max_mean_scores_pct: tuple[Optional[float]] = tuple(max((score for score in scores if score is not None), default=None) for scores in zip(*(mean_scores_pct for mean_scores, mean_scores_pct in mean_score_data.values()), strict=True))  # noqa

	log.info(f"Top-k (up to {K}) prediction scores across models and decoding strategies:")
	if len(set(totals.values())) > 1:
		log.warning(f"Not all prediction JSONs have the same number of samples, so treat means in the table below with caution: {sorted(set(totals.values()))}")
	paths_set = set(pred_jsons.keys())
	if any(set(gencfg_score_data) != paths_set for gencfg_score_data in score_data.values()):
		log.warning("Not all combinations of gencfg and model have data available, so not all means in the table below are actually directly comparable")

	table_headers = ['\nGen Cfg', '\nMean Top-k']
	if cfg.pfmt_model_spec:
		table_headers.extend(pred_json['model'].replace('/', '/\n') for pred_json in pred_jsons.values())
	else:
		for path in pred_jsons:
			file = os.path.basename(path)
			if file.endswith('.json'):
				file = file[:-5]
			table_headers.append('\n'.join(file[i:(i + 32)] for i in range(0, len(file), 32)))

	max_total = max(counter.total() for gencfg_score_data in score_data.values() for topk_scores, topk_counters in gencfg_score_data.values() for counter in topk_counters)
	score_width = len(format(max_total, 'd'))
	score_seq_width = K * (score_width + 5) - 3

	table_rows = []
	for gencfg, gencfg_score_data in sorted(score_data.items()):
		mean_scores, mean_scores_pct = mean_score_data[gencfg]
		table_row = [gencfg, f"{' < '.join(scorer.format_score(score=score, total=max_total, width=score_width) for score in mean_scores if score is not None):<{score_seq_width}s} = {' < '.join(scorer.format_score_pct(score=score_pct, total=1) for score_pct in mean_scores_pct if score_pct is not None)}"]
		for path in pred_jsons:
			if path in gencfg_score_data:
				topk_scores, topk_counters = gencfg_score_data[path]
				table_row.append(f"{' < '.join(scorer.format_score(score=score, total=counter.total(), width=score_width) for score, counter in zip(topk_scores, topk_counters)):<{score_seq_width}s} = {' < '.join(scorer.format_score_pct(score=score, total=counter.total()) for score, counter in zip(topk_scores, topk_counters))}")
			else:
				table_row.append(None)
		table_rows.append(table_row)
	sort_table_rows(table_rows=table_rows, default_order=tuple(tuple(-math.inf if score is None else -score for score in mean_score_data[table_row[0]][1]) for table_row in table_rows), table_headers=table_headers, sort_spec=None)
	table_rows.insert(0, [
		'MAXIMUM (mixed gencfg)',
		f"{' < '.join(scorer.format_score(score=score, total=max_total, width=score_width) for score in max_mean_scores if score is not None):<{score_seq_width}s} = {' < '.join(scorer.format_score_pct(score=score_pct, total=1) for score_pct in max_mean_scores_pct if score_pct is not None)}",
		*(f"{' < '.join(scorer.format_score(score=score, total=counter.total(), width=score_width) for score, counter in max_scores):<{score_seq_width}s} = {' < '.join(scorer.format_score_pct(score=score, total=counter.total()) for score, counter in max_scores)}" for max_scores in max_score_data.values()),
	])
	print(tabulate.tabulate(table_rows, headers=table_headers, tablefmt='pretty', numalign='left', stralign='left'))

	best_score_pct, best_path, best_gencfg = min((-topk_scores[0] / topk_counters[0].total(), path, gencfg) for gencfg, gencfg_score_data in score_data.items() for path, (topk_scores, topk_counters) in gencfg_score_data.items())
	best_score_pct = -best_score_pct
	best_header = pred_jsons[best_path]['model'] if cfg.pfmt_model_spec else os.path.basename(best_path)
	log.info(f"Best seen top-1 prediction score is {scorer.format_score_pct(score=best_score_pct, total=1)} by {best_header} with {best_gencfg}")

# Compare the loaded prediction JSONs in terms of their configuration
def compare_pred_jsons(pred_jsons: dict[str, dict[str, Any]]):

	if not pred_jsons:
		return

	samples = next(iter(pred_jsons.values()))['samples']
	if any(pred_json['samples'] != samples for pred_json in pred_jsons.values()):
		log.warning("CAREFUL: Not all loaded prediction JSONs have the same list of samples")
	else:
		log.info(f"All loaded prediction JSONs were inferenced on the same {len(samples)} samples")

	flat_pred_jsons = {path: utils.flatten_dict({key: value for key, value in pred_json.items() if key != 'predictions'}) for path, pred_json in pred_jsons.items()}
	flat_keys = sorted(set().union(*(flat_pred_json.keys() for flat_pred_json in flat_pred_jsons.values())))

	table_rows = []
	for flat_key in flat_keys:
		values = tuple(flat_pred_json.get(flat_key, None) for flat_pred_json in flat_pred_jsons.values())
		if any(value != values[0] for value in values):
			table_rows.append((flat_key, *(format(value)[:40] if value is not None else None for value in values)))

	if table_rows:
		log.info("Comparison of loaded prediction JSONs (only values that differ):")
		table_headers = []
		for path in flat_pred_jsons.keys():
			file = os.path.basename(path)
			if file.endswith('.json'):
				file = file[:-5]
			table_headers.append('\n'.join(file[i:(i + 32)] for i in range(0, len(file), 32)))
		print(tabulate.tabulate(table_rows, headers=(('\n' * max(header.count('\n') for header in table_headers)) + 'Key', *table_headers), tablefmt='pretty', numalign='left', stralign='left'))

# Collect prediction scores from predictions JSONs
def calc_prediction_scores(pred_jsons: dict[str, dict[str, Any]], scorer: PredictionScorer, topk: int) -> tuple[dict[str, dict[str, tuple[tuple[float, ...], tuple[Counter[Optional[str]], ...]]]], dict[str, int]]:
	# Note: The first output is dict[gencfg str, dict[JSON path, ((score float, score float, ...), (category counter, category counter, ...))]] where the number of scores/counters in each case is at most topk
	# Note: All counters for a particular JSON path are guaranteed to have exactly the same total() (totals are returned as second argument)
	score_data = {}
	totals = {}
	for path, pred_json in pred_jsons.items():
		totals[path] = len(pred_json['samples'])
		for gencfg, predictions in pred_json['predictions'].items():
			if (gencfg_score_data := score_data.get(gencfg, None)) is None:
				score_data[gencfg] = (gencfg_score_data := {})
			gencfg_score_data[path] = scorer.categorise_topk(predictions=dict(zip(pred_json['samples'], predictions['pred'], strict=True)), topk=min(topk, predictions['gen_cfg']['topk']))
	scorer.finalise()
	return score_data, totals

#
# Action: Format Wandb data
#

# Run row information class
@dataclasses.dataclass(frozen=True)
class RunRowInfo:
	model_info: ModelInfo
	wandb_run: wandb.apis.public.runs.Run
	wandb_log: Optional[str]
	train_tag: str
	command: Sequence[str]
	action: str
	row: dict[str, Any]

# Action: Format Wandb data
def action_format_wandb(cfg: omegaconf.DictConfig):

	log.info("Connecting to wandb API...")
	api = wandb.Api()
	wandb_path = cfg.wandb_project if cfg.wandb_entity is None else f'{cfg.wandb_entity}/{cfg.wandb_project}'
	log.info(f"Using wandb project: {wandb_path}")

	filters = [{'state': 'finished'}]
	now = datetime.datetime.now(tz=datetime.timezone.utc)
	if created_at_min := parse_duration_str(duration=cfg.fmt_max_ago, now=now):
		filters.append({'createdAt': {'$gt': created_at_min}})
	if created_at_max := parse_duration_str(duration=cfg.fmt_min_ago, now=now):
		filters.append({'createdAt': {'$lt': created_at_max}})
	if created_at_min := parse_datetime_str(stamp=cfg.fmt_min_stamp):
		filters.append({'createdAt': {'$gt': created_at_min}})
	if created_at_max := parse_datetime_str(stamp=cfg.fmt_max_stamp):
		filters.append({'createdAt': {'$lt': created_at_max}})

	need_logs = {'infer'}
	fmt_type = cfg.fmt_type.lower()
	if fmt_type == 'eval_gen_cls_v1':
		filters.append({'config.action': {'$in': ['eval', 'eval_cls']}})
	elif fmt_type == 'infer_v1':
		filters.append({'config.action': 'infer'})
	elif fmt_type in ('all_v1', 'all_v2'):
		filters.append({'config.action': {'$in': ['eval', 'eval_cls', 'infer']}})
	else:
		raise ValueError(f"Unsupported wandb format type: {cfg.fmt_type}")

	if fmt_type in ('infer_v1', 'all_v1', 'all_v2'):
		filters.append({'$or': [
			{'config.action': {'$ne': 'infer'}},
			{'$and': [
				{'$or': [{'config.infer_log': {'$exists': False}}, {'config.infer_log': True}]},
				{'config.infer_texts': '[]'},
				{'config.infer_image_dir': cfg.infer_image_dir},
				{'$or': [{'config.infer_ann_json': {'$exists': False}}, {'config.infer_ann_json': cfg.infer_ann_json}, {'config.infer_ann_json': cfg.infer_ann_json.replace(IMAGE_DIR_TAG, '$INFER_IMAGE_DIR')}]},
				{'config.infer_guided': False},
				{'$and': [{'config.gencfg': {'$exists': True}}, {'config.gencfg': cfg.gencfg}]} if cfg.gencfg else {'$or': [{'config.gencfg': {'$exists': False}}, {'config.gencfg': cfg.gencfg}]},
				{'$or': [{'config.gencfgs': {'$exists': False}}, {'config.gencfgs': '[]'}]},
				{'$or': [{'config.gencfgs_grid': {'$exists': False}}, {'config.gencfgs_grid': False}]},
			]},
		]})
		if cfg.gencfg:
			log.info(f"Limiting inference results to those with gencfg={cfg.gencfg}")

	train_tag_map = {}
	if fmt_model_hosts := set(cfg.fmt_model_hosts.split()):
		host_regexes = []
		for host_regex in fmt_model_hosts:
			host_regex = host_regex.replace('/', r'\/')
			host_regexes.append(host_regex)
			if not (host_regex.startswith('ovod-') and host_regex.endswith('-mlpod')):
				host_regexes.append(f'ovod-{host_regex}-mlpod')
		host_regexes.sort()
		train_runs = list(api.runs(path=wandb_path, filters={'$and': [{'state': 'finished'}, {'config.action': 'train'}, {'host': {'$regex': f"^(({')|('.join(host_regexes)}))$"}}]}))
		with tqdm.tqdm(desc='Looking up model hosts', total=len(train_runs), unit='run', unit_scale=False, dynamic_ncols=True, smoothing=0.08) as progress_bar:
			for train_run in train_runs:
				if not (model_dir := train_run.config['hydra_name']):
					continue
				metadata = train_run.metadata
				train_tag = metadata['host']
				if train_tag.startswith('ovod-') and train_tag.endswith('-mlpod'):
					train_tag = train_tag[5:-6]
				command = metadata['args']
				train_tag_map[model_dir] = (train_tag, command)
				progress_bar.update(n=1)
			assert progress_bar.n == progress_bar.total

	fmt_models = set(cfg.fmt_models.split())
	fmt_models.update(train_tag_map.keys())
	if fmt_models:
		fmt_models = sorted(fmt_models)
		fmt_models_spec = {model for model in fmt_models if '/' in model}
		fmt_models_dir = {model for model in fmt_models if '/' not in model}
		re_model_specs = '|'.join(re.escape(model).replace('/', r'\/') for model in fmt_models)
		if re_model_dirs := sorted({re.escape(model.split('/', maxsplit=1)[0]).replace('/', r'\/') for model in fmt_models_spec}):
			models_regex = rf"\b(({re_model_specs})\b|({'|'.join(re_model_dirs)})\b(?!\/))"
		else:
			models_regex = rf"\b({re_model_specs})\b"
		filters.append({'$or': [
			{'config.load_model': {'$regex': models_regex}},
			{'config.load_models': {'$regex': models_regex}},
		]})
	elif fmt_model_hosts:
		log.warning("No runs matched fmt_model_hosts specification and no fmt_models were provided either => Nothing to format!")
		return

	if fmt_hosts := set(cfg.fmt_hosts.split()):
		in_hosts = []
		number_hosts = []
		for host in fmt_hosts:
			in_hosts.append(host)
			if not (host.startswith('ovod-') and host.endswith('-mlpod')):
				in_hosts.append(f'ovod-{host}-mlpod')
			if re.fullmatch(r'[1-9][0-9]*', host):
				number_hosts.append(host)
		in_hosts.sort()
		host_filters = [{'host': {'$in': in_hosts}}]
		if number_hosts:
			number_hosts.sort()
			host_filters.append({'host': {'$regex': f"^ovod-[a-zA-Z-]*({'|'.join(number_hosts)})-mlpod$"}})
			filters.append({'$or': host_filters})
		else:
			filters.append(host_filters[0])

	wandb_runs = list(api.runs(path=wandb_path, filters={'$and': filters}))
	log.info(f"Retrieved {len(wandb_runs)} possibly relevant runs from server")

	row_infos = []
	run_log_map = {}
	wandb_run = train_runs = None  # noqa
	with tqdm.tqdm(desc='Retrieving run data', total=len(wandb_runs), unit='run', unit_scale=False, dynamic_ncols=True, smoothing=0.08) as progress_bar:
		for wandb_run in wandb_runs:
			action = wandb_run.config['action']
			if action in ('eval', 'eval_cls', 'infer'):
				for row in wandb_run.scan_history():
					try:
						model_path, model_dir, model_name = row['Eval/model_path'], row['Eval/model_dir'], row['Eval/model_name']
						model_info = ModelInfo(model_path=model_path, model_spec=f'{model_dir}/{model_name}', model_dir=model_dir, model_name=model_name)
					except KeyError:
						continue
					if fmt_models and model_info.model_spec not in fmt_models_spec and model_info.model_dir not in fmt_models_dir:  # noqa
						continue
					if (run_log := run_log_map.get(wandb_run, None)) is None and action in need_logs:
						if need_logs:
							try:
								response = requests.get(wandb_run.file('output.log').directUrl)
								if response.status_code == 200:
									run_log = response.text
								else:
									with tqdm.tqdm.external_write_mode():
										log.warning(f"Failed to retrieve run output log for {model_info.model_dir}: Unsuccessful request status code {response.status_code}")
							except requests.RequestException as e:
								with tqdm.tqdm.external_write_mode():
									log.warning(f"Failed to retrieve run output log for {model_info.model_dir}: {e}")
						run_log_map[wandb_run] = run_log
					if (entry := train_tag_map.get(model_info.model_dir, None)) is None:
						train_runs = list(api.runs(path=wandb_path, filters={'$and': [{'state': 'finished'}, {'config.action': 'train'}, {'config.hydra_name': model_info.model_dir}]}))
						if len(train_runs) >= 1:
							if len(train_runs) > 1:
								with tqdm.tqdm.external_write_mode():
									log.warning(f"Found more than one training run for model dir: {model_info.model_dir}")
							metadata = train_runs[0].metadata
							train_tag = metadata['host']
							if train_tag.startswith('ovod-') and train_tag.endswith('-mlpod'):
								train_tag = train_tag[5:-6]
							command = metadata['args']
						else:
							train_tag = '-'
							command = ['-']
						train_tag_map[model_info.model_dir] = (train_tag, command)
					else:
						train_tag, command = entry
					row_infos.append(RunRowInfo(model_info=model_info, wandb_run=wandb_run, wandb_log=run_log, train_tag=train_tag, command=command, action=action, row=row))
			else:
				raise ValueError(f"Unsupported action: {action}")
			progress_bar.set_postfix_str(f'rows={len(row_infos)}, train_tags={len(train_tag_map)}, logs={sum(1 for value in run_log_map.values() if value is not None)}', refresh=False)
			progress_bar.update(n=1)
		assert progress_bar.n == progress_bar.total
	log.info(f"Collected {len(row_infos)} relevant data rows from {len(wandb_runs)} considered runs")
	del wandb_runs, wandb_run, train_runs

	if row_infos:
		fmt_type = cfg.fmt_type.lower()
		if fmt_type == 'eval_gen_cls_v1':
			format_eval_gen_cls_v1(cfg=cfg, row_infos=row_infos)
		elif fmt_type == 'infer_v1':
			format_infer_v1(cfg=cfg, row_infos=row_infos)
		elif fmt_type == 'all_v1':
			format_eval_gen_cls_v1(cfg=cfg, row_infos=row_infos)
			format_infer_v1(cfg=cfg, row_infos=row_infos)
		elif fmt_type == 'all_v2':
			format_all_v2(cfg=cfg, row_infos=row_infos)
	else:
		log.warning("No data available!")

# Collect eval/gen/cls results data
def collect_eval_gen_cls_data(row_infos: Sequence[RunRowInfo]) -> dict[ModelInfo, dict[str, Any]]:

	log.info("Collecting eval/gen/cls data...")

	data = {}
	for RI in row_infos:
		data_model: dict[str, Any]
		if (data_model := data.get(RI.model_info, None)) is None:
			data[RI.model_info] = (data_model := dict(train_tag=RI.train_tag, command=RI.command, eval={}, eval_cls=('-', '-', '-', '-')))
		if RI.train_tag != data_model['train_tag']:
			log.warning(f"Inconsistent training tag: {RI.train_tag} vs {data_model['train_tag']}")
		if RI.command != data_model['command']:
			log.warning(f"Inconsistent command: {RI.command} vs {data_model['command']}")
		if RI.action == 'eval':
			if dataset_bin := RI.row.get('Eval/dataset', None):
				eval_data = (RI.row.get('Eval/noun_top1', '-'), RI.row.get('Eval/top1', '-'))
				if dataset_bin not in data_model['eval']:
					data_model['eval'][dataset_bin] = eval_data
		elif RI.action == 'eval_cls' and RI.row.get('Eval/dataset', None) == 'ImageNet1K':
			eval_cls_data = (RI.row.get('Eval/valid', '-'), RI.row.get('Eval/direct_correct', '-'), RI.row.get('Eval/DirectValidGuidedCorrect', '-'), RI.row.get('Eval/noun_top1', '-'))
			if all(item == '-' for item in data_model['eval_cls']):
				data_model.update(eval_cls=eval_cls_data)

	return data

# Collect infer results data
def collect_infer_data(cfg: omegaconf.DictConfig, row_infos: Sequence[RunRowInfo]) -> tuple[dict[ModelInfo, dict[str, Any]], tuple[str, ...]]:

	log.info("Collecting infer data...")

	class_annotations, categories = load_sample_annotations(ann_json=cfg.infer_ann_json, image_dir=cfg.infer_image_dir)

	preline_regex = re.compile(r'] Inferencing loaded model on the generated embeddings\.\.\.$', flags=re.MULTILINE)
	postline_regex = re.compile(r'] -{40,}$', flags=re.MULTILINE)
	prediction_regex = re.compile(r'] (.*) -->(.*)$', flags=re.MULTILINE)

	data = {}
	all_predictions = {}
	scorer = PredictionScorer(class_annotations=class_annotations, categories=categories)

	for RI in row_infos:

		if RI.action != 'infer':
			continue

		predictions = {}
		if RI.wandb_log is not None:
			model_matches = tuple(re.finditer(rf'] Loading model: .*/{re.escape(RI.model_info.model_spec)}$', RI.wandb_log, flags=re.MULTILINE))
			if not model_matches:
				log.warning(f"Failed to locate inference result for model despite having a result row: {RI.model_info.model_spec}")
			else:
				if len(model_matches) > 1:
					log.warning(f"Encountered multiple inference results within the same log for the same model: {RI.model_info.model_spec}")
				model_match_end = model_matches[0].end()
				preline_match_end = preline_regex.search(RI.wandb_log, pos=model_match_end).end()
				postline_match_start = postline_regex.search(RI.wandb_log, pos=model_match_end).start()
				if preline_match_end > postline_match_start:
					log.warning(f"Failed to parse inference results due to pre/post line order mismatch: {RI.model_info.model_spec}")
				else:
					predictions = {src: ' '.join(pred.split()) for src, pred in prediction_regex.findall(RI.wandb_log, pos=preline_match_end, endpos=postline_match_start)}
					if not predictions:
						log.warning(f"Failed to locate any prediction lines for the model: {RI.model_info.model_spec}")
		all_predictions.update(predictions)

		score, score_counter = scorer.categorise(predictions=predictions)  # Note: Outside the if statement so that missing classes etc are still collected from all infer results
		if RI.model_info not in data:  # Note: Keep only the first infer results you encounter (should be the most recent infer results)
			data[RI.model_info] = dict(train_tag=RI.train_tag, command=RI.command, valid=RI.row.get('Eval/valid', '-'), predictions=predictions, score=score, score_counter=score_counter)

	scorer.finalise()
	return data, tuple(all_predictions.keys())

# Format table of eval/gen/cls results (version 1)
def format_eval_gen_cls_v1(cfg: omegaconf.DictConfig, row_infos: Sequence[RunRowInfo]):

	data = collect_eval_gen_cls_data(row_infos=row_infos)
	log.info("Generating eval/gen/cls table (version 1)...")

	table_rows = []
	for model_info, data_model in data.items():
		table_rows.append((
			data_model['train_tag'],
			model_info.model_spec,
			format_eval_gen_cls_top1(r'noun_dataset_cache_vt[0-9]+.bin', model_info=model_info, data_model=data_model),
			format_eval_gen_cls_top1(r'noun_dataset2_cache_vt[0-9]+.bin', model_info=model_info, data_model=data_model),
			format_eval_gen_cls_top1(r'noun_dataset3_cache_vt[0-9]+.bin', model_info=model_info, data_model=data_model),
			format_eval_gen_cls_top1(r'captions2_cache_vt[0-9]+.bin', model_info=model_info, data_model=data_model),
			format_percent_str(data_model['eval_cls'][0]),
			format_percent_str(data_model['eval_cls'][1]),
			format_percent_str(data_model['eval_cls'][2]),
			format_percent_str(data_model['eval_cls'][3]),
			format_percent_str(get_eval_gen_cls_top1(r'cls_imagenet1k_valid_guide_vt[0-9]+.bin', model_info=model_info, data_model=data_model)[0]),
			format_multi_guide_cls('cls_fashionmnist_valid_$TYPE_vt[0-9]+.bin', model_info=model_info, data_model=data_model, precision=2),
			format_multi_guide_cls('cls_cifar10_valid_$TYPE_vt[0-9]+.bin', model_info=model_info, data_model=data_model, precision=2),
			format_multi_guide_cls('cls_cifar100_valid_$TYPE_vt[0-9]+.bin', model_info=model_info, data_model=data_model, precision=2),
			format_multi_guide_cls('cls_food101_valid_$TYPE_vt[0-9]+.bin', model_info=model_info, data_model=data_model, precision=3),
			format_multi_guide_cls('cls_imagenette_valid_$TYPE_vt[0-9]+.bin', model_info=model_info, data_model=data_model, precision=3),
			format_multi_guide_cls('cls_imagewoof_valid_$TYPE_vt[0-9]+.bin', model_info=model_info, data_model=data_model, precision=3),
		))

	log.info(f"Collected data for {len(table_rows)} models")
	table_headers = ('Tag', 'Ovod', 'Eval dataset', 'Eval dataset2', 'Eval dataset3', 'Eval captions2', 'Valid', 'Direct', 'Fixed', 'Guided', 'ImageNet1K', 'FashionMNIST', 'CIFAR10', 'CIFAR100', 'Food101', 'Imagenette', 'Imagewoof')
	sort_table_rows(table_rows=table_rows, default_order=None, table_headers=table_headers, sort_spec=cfg.fmt_sort)
	print(tabulate.tabulate(table_rows, headers=table_headers, tablefmt='pretty', numalign='left', stralign='left'))

# Format table of infer results (version 1)
def format_infer_v1(cfg: omegaconf.DictConfig, row_infos: Sequence[RunRowInfo]):

	data, all_predictions = collect_infer_data(cfg=cfg, row_infos=row_infos)
	log.info("Generating infer table (version 1)...")

	table_rows = []
	PRED_WIDTH = 22
	for model_info, data_model in data.items():
		pred_data = tuple(data_model['predictions'].get(src, '-') for src in all_predictions)
		table_rows.append((
			data_model['train_tag'],
			model_info.model_spec,
			format_percent_str(data_model['valid']),
			*(pred[:PRED_WIDTH] for pred in pred_data),
		))

	log.info(f"Collected data for {len(table_rows)} models")
	table_headers = ('Tag', 'Ovod', 'Valid', *(format(os.path.splitext(src)[0].replace('_', ' '), f'{PRED_WIDTH}s') for src in all_predictions))
	sort_table_rows(table_rows=table_rows, default_order=None, table_headers=table_headers, sort_spec=cfg.fmt_sort)
	print(tabulate.tabulate(table_rows, headers=table_headers, tablefmt='pretty', numalign='left', stralign='left'))

# Format table of all results (version 2)
def format_all_v2(cfg: omegaconf.DictConfig, row_infos: Sequence[RunRowInfo]):

	data_eval = collect_eval_gen_cls_data(row_infos=row_infos)
	data_infer, all_predictions = collect_infer_data(cfg=cfg, row_infos=row_infos)
	log.info("Generating all table (version 2)...")

	table_rows = []
	table_rows_order = []
	for model_info in set().union(data_eval, data_infer):
		if (deval := data_eval.get(model_info, None)) is None:
			deval = dict(train_tag='-', command=['-'], eval={}, eval_cls=('-', '-', '-', '-'))
		if (dinfer := data_infer.get(model_info, None)) is None:
			dinfer = dict(train_tag='-', command=['-'], valid='-', predictions={}, score=0.0, score_counter=collections.Counter())
		assert deval['train_tag'] == '-' or dinfer['train_tag'] == '-' or deval['train_tag'] == dinfer['train_tag']
		train_tag = deval['train_tag'] if deval['train_tag'] != '-' else dinfer['train_tag']
		assert deval['command'] == ['-'] or dinfer['command'] == ['-'] or deval['command'] == dinfer['command']
		command = deval['command'] if deval['command'] != ['-'] else dinfer['command']
		score: Counter[Optional[str]] = dinfer['score_counter']
		assert score['correct_primary'] + score['correct_secondary'] + score['close_primary'] + score['close_secondary'] + score['incorrect'] + score[None] == len(dinfer['predictions'])
		table_rows.append((
			train_tag,
			model_info.model_spec,
			format_eval_gen_cls_top1(r'captions2_cache_vt[0-9]+.bin', model_info=model_info, data_model=deval),
			format_percent_str(deval['eval_cls'][0]),
			format_percent_str(deval['eval_cls'][1]),
			format_percent_str(deval['eval_cls'][2]),
			format_percent_str(deval['eval_cls'][3]),
			format_multi_guide_cls('cls_imagenet1k_valid_$TYPE_vt[0-9]+.bin', model_info=model_info, data_model=deval, precision=3),
			format_multi_guide_cls('cls_food101_valid_$TYPE_vt[0-9]+.bin', model_info=model_info, data_model=deval, precision=3),
			PredictionScorer.format_counter(counter=score),
			PredictionScorer.format_score(score=dinfer['score'], total=score.total()),
			PredictionScorer.format_score_pct(score=dinfer['score'], total=score.total()),
			format_percent_str(dinfer['valid']),
			' '.join(reversed(command)),
		))
		table_rows_order.append((tuple(command), train_tag, model_info.model_spec))

	log.info(f"Collected data for {len(table_rows)} models")
	table_headers = ('Tag', 'Model', 'Eval captions2', 'Valid', 'Direct', 'Fixed', 'Guided', 'ImageNet1K', 'Food101', 'Infer stats', 'Score', 'Score %', 'Valid', 'Command')
	sort_table_rows(table_rows=table_rows, default_order=table_rows_order, table_headers=table_headers, sort_spec=cfg.fmt_sort)
	print(tabulate.tabulate(table_rows, headers=table_headers, tablefmt='pretty', numalign='left', stralign='left'))

# Parse a string duration into an absolute string datetime specification
def parse_duration_str(duration: str, now: datetime.datetime) -> Optional[str]:
	if not duration:
		return None
	if not (match := re.fullmatch(r'(\d+y)?(\d+w)?(\d+d)?(\d+h)?(\d+m)?(\d+s)?', duration, flags=re.IGNORECASE)):
		raise ValueError(f"Invalid duration specification: {duration}")
	years, weeks, days, hours, minutes, seconds = match.groups()
	ago = datetime.timedelta(
		days=(365 * int(years[:-1]) if years else 0) + (7 * int(weeks[:-1]) if weeks else 0) + (int(days[:-1]) if days else 0),
		hours=int(hours[:-1]) if hours else 0,
		minutes=int(minutes[:-1]) if minutes else 0,
		seconds=int(seconds[:-1]) if seconds else 0,
	)
	return (now - ago).isoformat(timespec='seconds')

# Parse a string datetime stamp into an absolute string datetime specification
def parse_datetime_str(stamp: str) -> Optional[str]:
	if not stamp:
		return None
	elif len(stamp) == 9:
		datetime_format = "D%Y%m%d"
	elif len(stamp) == 16:
		datetime_format = "D%Y%m%d_%H%M%S"
	else:
		raise ValueError(f"Invalid datetime stamp: {stamp}")
	return datetime.datetime.strptime(stamp, datetime_format).astimezone().isoformat(timespec='seconds')

# Get the eval/gen/cls noun/token top-1 results for a particular dataset regex
def get_eval_gen_cls_top1(dataset: str, model_info: ModelInfo, data_model: dict[str, Any]) -> tuple[Union[float, str], Union[float, str]]:
	options = tuple(top1 for dset, top1 in data_model['eval'].items() if re.fullmatch(dataset, dset))
	if options:
		if len(options) > 1:
			log.warning(f"There were multiple options for '{dataset}': {model_info.model_spec}")
		return options[0]
	else:
		return '-', '-'

# Format to a string the eval/gen/cls noun/token top-1 results for a particular dataset regex
def format_eval_gen_cls_top1(dataset: str, model_info: ModelInfo, data_model: dict[str, Any]) -> str:
	noun_top1, token_top1 = get_eval_gen_cls_top1(dataset=dataset, model_info=model_info, data_model=data_model)
	if not isinstance(noun_top1, str):
		noun_top1 = format(noun_top1, '.3f')
	if not isinstance(token_top1, str):
		token_top1 = format(token_top1, '.4f')
	return f"{noun_top1:>6s}% + {token_top1:>6s}%"

# Format to a string the eval/gen/cls multi/guide top-1 results for a particular dataset regex
def format_multi_guide_cls(dataset: str, model_info: ModelInfo, data_model: dict[str, Any], precision: int = 3) -> str:
	noun_top1_multi = get_eval_gen_cls_top1(dataset=dataset.replace('$TYPE', 'multi'), model_info=model_info, data_model=data_model)[0]
	noun_top1_guide = get_eval_gen_cls_top1(dataset=dataset.replace('$TYPE', 'guide'), model_info=model_info, data_model=data_model)[0]
	if not isinstance(noun_top1_multi, str):
		noun_top1_multi = format(noun_top1_multi, f'.{precision}f')
	if not isinstance(noun_top1_guide, str):
		noun_top1_guide = format(noun_top1_guide, f'.{precision}f')
	return f"{noun_top1_multi:>{precision + 3}s}% + {noun_top1_guide:>{precision + 3}s}%"

#
# Action: Collect Wikipedia images
#

# Action: Collect Wikipedia images
def action_collect_wiki_images(cfg: omegaconf.DictConfig, hydra_dir: str):

	device, device_is_cpu, device_is_cuda = load_device(cfg=cfg)
	embedder = load_embedder(cfg=cfg, device=device)
	dataset = load_embedding_dataset(cfg=cfg, embedder=embedder, embed_dataset='NounDataset', use_targets=True, use_cache=False)

	log.info(f"Collecting Wikipedia images based on {len(dataset.targets)} target nouns from noun dataset of vocab threshold {dataset.vocab_thres}...")

	wiki_collect_dir = os.path.abspath(resolve_source_path(cfg.wiki_collect_dir))
	with contextlib.suppress(OSError):
		os.mkdir(wiki_collect_dir)
	image_dir = os.path.join(wiki_collect_dir, f"{os.path.basename(hydra_dir)}-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
	os.mkdir(image_dir)
	log.info(f"Output image directory: {image_dir}")

	wiki_titles = []
	with tqdm.tqdm(desc='Collecting Wikipedia page titles', total=len(dataset.targets), unit='noun', unit_scale=False, dynamic_ncols=True, smoothing=0.08) as progress_bar:
		for target_noun in dataset.targets:
			progress_bar.set_postfix(noun=target_noun, titles=len(wiki_titles))
			response_data = make_wiki_request(action='query', list='search', srsearch=target_noun, format='json')
			if response_data is None:
				with tqdm.tqdm.external_write_mode():
					log.warning(f"Wikipedia API requests failed for: {target_noun}")
			else:
				try:
					search_results = response_data['query']['search']
					if len(search_results) <= 0:
						with tqdm.tqdm.external_write_mode():
							log.warning(f"No Wikipedia page found for: {target_noun}")
					else:
						for result in search_results:
							title = result['title']
							if 'disambiguation' not in title.lower():
								wiki_titles.append(title)
								break
						else:
							with tqdm.tqdm.external_write_mode():
								log.warning(f"Only found a disambiguation page for: {target_noun}")
				except (TypeError, KeyError):
					with tqdm.tqdm.external_write_mode():
						log.warning("Failed to retrieve Wikipedia page title from JSON")
			progress_bar.update(n=1)
		progress_bar.set_postfix(titles=len(wiki_titles))
		assert progress_bar.n == progress_bar.total

	wiki_title_counts = collections.Counter(wiki_titles)
	log.info(f"Wikipedia pages found multiple times (>= 4): {dict(sorted(tuple((title, count) for title, count in wiki_title_counts.items() if count >= 4), key=lambda item: (-item[1], item[0])))}")
	log.info(f"Collect {len(wiki_titles)} page titles ({len(wiki_title_counts)} unique) from the {len(dataset.targets)} target nouns")

	num_wiki_images = 0
	wiki_titles = list(wiki_title_counts.keys())
	random.shuffle(wiki_titles)
	with tqdm.tqdm(desc='Collecting Wikipedia images', total=len(wiki_titles), unit='title', unit_scale=False, dynamic_ncols=True, smoothing=0.08) as progress_bar:

		it = iter(wiki_titles)
		while titles := tuple(itertools.islice(it, 16)):

			response_data: Any = make_wiki_request(action='query', prop='pageimages', format='json', piprop='original', titles='|'.join(titles))
			if response_data is None:
				with tqdm.tqdm.external_write_mode():
					log.warning(f"Wikipedia API requests failed for batch: {titles}")
			else:
				for image_dict in response_data['query']['pages'].values():
					if original_dict := image_dict.get('original', None):

						original_area = original_dict['width'] * original_dict['height']
						if original_area < 20000:
							with tqdm.tqdm.external_write_mode():
								log.warning(f"Ignoring too-small {original_dict['width']}\xD7{original_dict['height']} image: {original_dict['source']}")
						elif original_area > PIL.Image.MAX_IMAGE_PIXELS:
							with tqdm.tqdm.external_write_mode():
								log.warning(f"Ignoring too-large {original_dict['width']}\xD7{original_dict['height']} image: {original_dict['source']}")
						else:

							success = False
							image_url = original_dict['source']
							if not image_url.lower().endswith(('.svg', '.pdf', '.ogv')):
								for attempt in range(3):
									try:
										response = requests.get(url=image_url, headers=WIKIMEDIA_USER_AGENT)
										response.raise_for_status()
										image = PIL.Image.open(io.BytesIO(response.content))
										image.verify()  # Note: This 'consumes' the image, so to use the image again you need to reopen it
										success = True
										break
									except requests.RequestException as e:
										with tqdm.tqdm.external_write_mode():
											log.warning(f"Network error downloading {image_url}: {e} (retrying in 10s)")
										time.sleep(10)
									except (OSError, SyntaxError, ValueError) as e:
										with tqdm.tqdm.external_write_mode():
											log.warning(f"Invalid image {image_url}: {e}")
										break

							if success:
								num_wiki_images += 1
								image_filename = urllib.parse.unquote(os.path.basename(image_url)).replace('\0', '')
								image_filename = f"{num_wiki_images:06d}-{image_filename}"
								name, ext = os.path.splitext(image_filename)
								if len(name) > 250:
									with tqdm.tqdm.external_write_mode():
										log.warning(f"Truncating too-long filename stem: {name}")
									name = name[:250]
								image_path = os.path.join(image_dir, f'{name}.jpg')
								try:
									image = PIL.Image.open(io.BytesIO(response.content)).convert('RGB')
									image_area = image.width * image.height
									if image_area > 800000:
										scale_factor = (800000 / image_area) ** 0.5
										new_width = round(image.width * scale_factor)
										new_height = round(image.height * scale_factor)
										image = image.resize((new_width, new_height), resample=PIL.Image.Resampling.LANCZOS)  # noqa
									image.save(image_path)
								except (OSError, SyntaxError, ValueError) as e:
									num_wiki_images -= 1
									with tqdm.tqdm.external_write_mode():
										log.warning(f"Failed to open/convert/save image {image_url}: {e}")

			progress_bar.set_postfix(images=num_wiki_images, refresh=False)
			progress_bar.update(n=len(titles))

		progress_bar.set_postfix(images=num_wiki_images)
		assert progress_bar.n == progress_bar.total

	log.info(f"Successfully downloaded {num_wiki_images} Wikipedia images from {len(wiki_titles)} pages from {len(dataset.targets)} target nouns")

# Robustly make a Wikipedia request
def make_wiki_request(**params):

	retry_time = 1
	while True:

		time.sleep(0.1)  # Note: Try to stay below any rate limits

		try:
			response = requests.get(url='https://en.wikipedia.org/w/api.php', headers=WIKIMEDIA_USER_AGENT, params=params)
			response.raise_for_status()
			if params.get('format', '') == 'json':
				if 'application/json' in response.headers.get('Content-Type', ''):
					return response.json()
				else:
					with tqdm.tqdm.external_write_mode():
						log.warning("Received response was not in JSON format")
			else:
				return response
		except requests.ConnectionError:
			log.warning("Failed to establish a connection")
		except requests.HTTPError as e:
			log.warning(f"HTTP error occurred: {e.response.status_code}")
		except requests.Timeout:
			log.warning("The request timed out")
		except requests.TooManyRedirects:
			log.warning("Too many redirects")
		except requests.RequestException as e:
			log.warning(f"An error occurred: {e}")
		except json.decoder.JSONDecodeError:
			log.warning("Failed to parse received JSON")

		time.sleep(retry_time)
		if retry_time < 120:
			retry_time *= 2
		else:
			return None

#
# Action: Sample images
#

# Action: Samples images
def action_sample_images(cfg: omegaconf.DictConfig, hydra_dir: str):

	N = cfg.sample_count
	input_dir = os.path.abspath(resolve_source_path(cfg.sample_input_dir))
	if not input_dir:
		raise ValueError("Need to specify an input image directory using sample_input_dir")
	log.info(f"Sampling up to {N} images from directory: {input_dir}")

	log.info("Collecting all image files from input directory...")
	filenames = os.listdir(input_dir)
	input_images = tuple(sorted(set().union(*([filename for filename in filenames if fnmatch.fnmatch(name=filename.lower(), pat=pattern)] for pattern in utils.IMAGE_PATTERNS))))
	num_images = len(input_images)
	log.info(f"Found {num_images} images and {len(filenames) - num_images} non-images in the directory of {len(filenames)} entries")
	N = min(N, num_images)

	output_parent_dir = os.path.abspath(resolve_source_path(cfg.sample_output_dir))
	with contextlib.suppress(OSError):
		os.mkdir(output_parent_dir)
	output_dir = os.path.join(output_parent_dir, f"{os.path.basename(hydra_dir)}-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
	os.mkdir(output_dir)
	log.info(f"Sampled output image directory: {output_dir}")

	if cfg.sample_special:

		device, device_is_cpu, device_is_cuda = load_device(cfg=cfg)
		embedder = load_embedder(cfg=cfg, device=device)
		prompts = classification_dataset.load_image_dataset_prompts(name='imagenet1k', variant='clip')  # Note: ImageNet1K CLIP prompts are generally suitable for open vocabulary images

		special = tuple(cfg.sample_special)
		special_mean = torch.tensor(cfg.sample_special_mean, dtype=embedder.embed_dtype, device=device)
		special_factor = torch.tensor(cfg.sample_special_factor, dtype=embedder.embed_dtype, device=device).unsqueeze(dim=0)
		if special_factor.shape != (1, len(special)):
			raise ValueError("The lists sample_special and sample_special_factor must have the same lengths")

		with torch.inference_mode(), embedder.inference_model():

			log.info(f"Using embedder zero-shot to weight images related to special nouns: {', '.join(special)}")
			text_embeds, text_embeds_T = compute_text_embeddings(embedder=embedder, nouns=special, prompts=prompts)

			log.info(f"Embedding {num_images} input images and calculating special weights...")
			input_weights = []
			input_images_iter = iter(input_images)
			image_transform = embedder.get_image_transform()
			with tqdm.tqdm(desc='Embedding images', total=num_images, unit='img', unit_scale=False, dynamic_ncols=True, smoothing=0.08) as progress_bar:
				while image_paths := tuple(itertools.islice(input_images_iter, embedder.image_batch_size)):
					with embedder.inference_mode():
						image_embeds = embedder.inference_image(torch.utils.data.default_collate(tuple(image_transform(PIL.Image.open(os.path.join(input_dir, image_path))) for image_path in image_paths)))
					logits = image_embeds @ text_embeds_T
					input_weights.append((logits - special_mean).clamp_(min=0).mul_(special_factor).sum(dim=1).exp().cpu())
					progress_bar.update(n=image_embeds.shape[0])
				assert sum(weights.numel() for weights in input_weights) == len(input_images) == num_images == progress_bar.total == progress_bar.n

			input_weights = torch.concat(input_weights, dim=0)
			input_weights.div_(input_weights.mean())
			sample_indices = torch.multinomial(input_weights, num_samples=N, replacement=False)
			sampled_images = [input_images[i] for i in sample_indices]

	else:
		sampled_images = random.sample(input_images, N)

	assert len(sampled_images) == N
	log.info(f"Copying {N} randomly sampled images to output directory...")
	for img in sampled_images:
		shutil.copy2(src=os.path.join(input_dir, img), dst=os.path.join(output_dir, img))
	log.info(f"Finished copying {N} sampled images to output directory")

#
# Common action parts
#

# Load the PyTorch device
def load_device(cfg: omegaconf.DictConfig, device: Optional[str] = None) -> tuple[torch.device, bool, bool]:
	return infer.load_device(cfg.device if device is None else device)

# Load the embedder
def load_embedder(cfg: omegaconf.DictConfig, device: torch.device, load_model: bool = False, check_consistent: bool = False) -> embedders.Embedder:
	log.info(f"Creating embedder of specification {cfg.embedder_spec}{' with checking' if check_consistent else ''}...")
	embedder = embedders.Embedder.create(
		spec=cfg.embedder_spec,
		amp=cfg.embedder_amp,
		amp_bf16=cfg.embedder_amp_bf16,
		tokenizer_batch_size=cfg.batch_size_token,
		inference_batch_size=cfg.batch_size_embed,
		image_batch_size=cfg.batch_size_image,
		load_model=load_model,
		compile_model=cfg.embedder_compile,
		use_optimum=cfg.embedder_optimum,
		device=device,
		check=check_consistent,
	)
	log.info(f"Created embedder of class type {type(embedder).__qualname__}")
	return embedder

# Compute text embeddings based on prompt templates
def compute_text_embeddings(embedder: embedders.Embedder, nouns: Sequence[str], prompts: Sequence[tuple[str, bool]], release: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
	# The prompts are a sequence of pairs of the string prompt template (incorporates '{noun}') and a boolean whether the noun should be articled
	# Note: This function must be called with embedder.inference_model() entered

	with tqdm.tqdm(desc='Computing text embeddings', total=len(nouns), unit='noun', unit_scale=False, dynamic_ncols=True, smoothing=0.08) as progress_bar:
		text_embeds = []
		for noun in nouns:
			articled_noun = noun_dataset.NounDataset.make_indefinite(noun)
			noun_embeds = []
			it = iter(prompts)
			with embedder.inference_mode():
				while prompts_chunk := tuple(itertools.islice(it, embedder.inference_batch_size)):
					noun_embeds.append(embedder.inference_text(text=tuple(prompt.format(noun=articled_noun if article else noun) for prompt, article in prompts_chunk)))
			text_embeds.append(torch.nn.functional.normalize((noun_embeds[0] if len(noun_embeds) <= 1 else torch.concat(tensors=noun_embeds, dim=0)).mean(dim=0), dim=-1))
			progress_bar.update()
		text_embeds = torch.stack(text_embeds)
		text_embeds_T = text_embeds.T

	if release:
		del noun_embeds
		utils.release_cuda_memory(device=embedder.device)

	return text_embeds, text_embeds_T

# Load the embedding dataset
def load_embedding_dataset(
	cfg: omegaconf.DictConfig,            # Hydra configuration
	embedder: embedders.Embedder,         # The embedder does not need a target_config yet for this function
	embed_dataset: Optional[str] = None,  # Optional manual specification of which embedding dataset to load (default: cfg.embedding_dataset)
	use_targets: Optional[bool] = None,   # Whether to use targets in the embedding dataset (False/True) or let the dataset automatically decide based on availability (None)
	vocab_thres: Optional[int] = None,    # NounDataset: Optional manual specification of vocab threshold (default: cfg.vocab_thres)
	use_cache: Optional[bool] = None,     # NounDataset: Optional manual specification of whether to use a cache (default: cfg.noun_cache)
	check_consistent: bool = False,       # NounDataset: Whether to perform consistency checking of the noun dataset
	check_print: int = 0,                 # NounDataset: How many lines of the noun dataset to print for testing (0 = None)
	training: bool = False,               # EmbeddingCache: Whether the dataset should be loaded in training mode
	strict_embedder: bool = True,         # EmbeddingCache: Whether to perform strict embedder compatibility checking (recommended)
) -> embedding_dataset.EmbeddingDataset:

	if embed_dataset is None:
		embed_dataset = cfg.embedding_dataset
	if not embed_dataset:
		raise ValueError("Cannot load embedding dataset given by empty string")

	if embed_dataset.lower() == 'noundataset':

		if vocab_thres is None:
			vocab_thres = cfg.vocab_thres
		if use_cache is None:
			use_cache = cfg.noun_cache

		log.info(f"Loading {'cached' if use_cache else 'uncached'} noun dataset {'with any available' if use_targets is None else 'with' if use_targets else 'without'} targets{' and with checking' if check_consistent or check_print != 0 else ''}...")
		dataset = noun_dataset.NounDataset(
			embedder=embedder,
			vocab_path=resolve_source_path(cfg.vocab_path),
			prompt_path=resolve_source_path(cfg.prompt_path),
			prompt_collection=cfg.prompt_collection,
			hypernym_collection=cfg.hypernym_collection,
			vocab_thres=vocab_thres,
			cache_dir=resolve_source_path(cfg.noun_cache_dir) if use_cache else None,
			force_recache=cfg.noun_recache,
			check_consistent=check_consistent,
			check_print=check_print,
			use_targets=use_targets,
		)
		log.info(f"Loaded noun dataset of class type {type(dataset).__qualname__}")

	else:

		if cfg.embedding_cache_dir and not os.path.isabs(embed_dataset) and not os.path.exists(embed_dataset) and os.path.exists(embed_dataset_alt := os.path.join(resolve_source_path(cfg.embedding_cache_dir), embed_dataset)):
			embed_dataset = embed_dataset_alt

		log.info(f"Loading embedding cache {'with any available' if use_targets is None else 'with' if use_targets else 'without'} targets{'' if strict_embedder else ' and non-strict embedder check'}...")
		embed_cache = embedding_cache.EmbeddingCache(cache_path=embed_dataset, embedder=embedder, use_targets=use_targets, strict_embedder=strict_embedder)
		dataset = embed_cache.create_dataset(batch_size=cfg.batch_size, training=training)
		log.info(f"Loaded embedding cache dataset of class type {type(dataset).__qualname__}")

	return dataset

# Load multiple embedding datasets
def load_embedding_datasets(cfg: omegaconf.DictConfig, embedder: embedders.Embedder, embed_datasets: Optional[Iterable[str]] = None, **kwargs) -> tuple[embedding_dataset.EmbeddingDataset, ...]:
	# The embedder does not need a target_config yet for this function
	# cfg, embedder = See load_embedding_dataset()
	# embed_datasets = Optional manual specification of which embedding datasets to load (default: cfg.embedding_datasets)
	# kwargs = See keyword arguments of load_embedding_dataset() beyond embed_dataset
	if embed_datasets is None:
		embed_datasets = cfg.embedding_datasets
	return tuple(load_embedding_dataset(cfg=cfg, embedder=embedder, embed_dataset=embed_dataset, **kwargs) for embed_dataset in embed_datasets)

# Get the path to save any newly create embedding cache as (empty => error)
def get_cache_save_path(cfg: omegaconf.DictConfig):

	save_embedding_cache = cfg.save_embedding_cache
	if not save_embedding_cache:
		raise ValueError("Embedding cache save path is required but empty")

	basename = os.path.basename(save_embedding_cache)
	if not basename:
		raise ValueError(f"Embedding cache save path cannot look like a directory: {save_embedding_cache}")

	if basename == save_embedding_cache:
		embedding_cache_dir = resolve_source_path(cfg.embedding_cache_dir)
		with contextlib.suppress(OSError):
			os.mkdir(embedding_cache_dir)
		return os.path.join(embedding_cache_dir, basename)
	else:
		return save_embedding_cache

# Generate a target configuration based on given target nouns and the configured model specification
def gen_target_config(cfg: omegaconf.DictConfig, embedder: embedders.Embedder, targets: Sequence[str], num_invalid_targets: int) -> embedders.TargetConfig:
	# Note: The targets sequence should contain all targets (including invalid ones), except possibly omitting a leading empty string invalid target (num_invalid_targets must also be adjusted in that case)

	log.info(f"Generating target configuration for loaded target nouns and model of specification {cfg.model}...")
	model_class: Type[embedding_decoder.EmbeddingDecoder] = getattr(embedding_decoder, cfg.model)

	cfg_target_kwargs = dict(
		with_start_token=cfg.with_start_token,
		with_end_token=cfg.with_end_token,
		compact_ids=cfg.compact_ids,
		fixed_token_length=cfg.fixed_token_length,
		auto_fixed_token_length=cfg.auto_fixed_token_length,
		use_masks=cfg.use_masks,
	)

	target_kwargs = model_class.get_target_config_kwargs(**cfg_target_kwargs)
	if target_kwargs.keys() != cfg_target_kwargs.keys():
		raise ValueError("Model unexpectedly changed the target configuration keys")
	if changed_keys := tuple(key for key, value in target_kwargs.items() if hasattr(cfg, key) and (value != cfg_target_kwargs[key] or type(value) != type(cfg_target_kwargs[key]))):  # noqa
		log.warning(f"Model changed the following target configuration keys: {{{', '.join(f'{key}: {cfg_target_kwargs[key]} -> {target_kwargs[key]}' for key in changed_keys)}}}")

	target_config = embedder.create_target_config(targets=targets, **target_kwargs)
	embedder.configure_target(target_config=target_config, target_vocab=targets[num_invalid_targets:] if num_invalid_targets > 0 else targets)

	return target_config

# Generate a data configuration based on a given embedding dataset and the configured model specification
def gen_data_config(cfg: omegaconf.DictConfig, dataset: embedding_dataset.EmbeddingDataset, **kwargs) -> embedding_dataset.DataConfig:
	# Target config must be available in the dataset embedder
	# The kwargs allow an action to set custom limitations explicitly (e.g. multi_target=False) that can override the cfg

	log.info(f"Resolving data configuration for loaded embedding dataset and model of specification {cfg.model}...")
	model_class: Type[embedding_decoder.EmbeddingDecoder] = getattr(embedding_decoder, cfg.model)

	cfg_data_kwargs = dict(
		use_weights=cfg.use_weights,
		unit_weights=None,
		multi_target=cfg.multi_target,
		multi_first=cfg.multi_first,
		full_targets=None,
		fixed_multi_length=cfg.fixed_multi_length,
		multi_length=None,
	)
	cfg_data_kwargs.update(kwargs)

	data_kwargs = model_class.get_data_config_kwargs(**cfg_data_kwargs)
	if not (set(data_kwargs.keys()) == set(cfg_data_kwargs.keys()) == {field.name for field in dataclasses.fields(embedding_dataset.DataConfig)}):
		raise ValueError("Model unexpectedly changed the data configuration keys or some keys are unexpected")
	if changed_keys := tuple(key for key, value in data_kwargs.items() if cfg_data_kwargs[key] is not None and (value != cfg_data_kwargs[key] or type(value) != type(cfg_data_kwargs[key]))):  # noqa
		log.warning(f"Model changed the following data configuration keys: {{{', '.join(f'{key}: {cfg_data_kwargs[key]} -> {data_kwargs[key]}' for key in changed_keys)}}}")

	data_config = dataset.resolve_data_config(**data_kwargs)
	dataset.configure_data(data_config)

	log.info(f"Dataset is configured to use {'multiple' if data_config.multi_target else 'single'} targets per embedding, and {'' if data_config.use_weights else 'NOT '}to use target weights ({'normalized' if data_config.unit_weights else 'unnormalized'})")
	if data_config.multi_target:
		log.info(f"Dataset is {'full ' if data_config.full_targets else ''}multi-target with a {'fixed M of' if data_config.fixed_multi_length else 'dynamic M of up to'} {data_config.multi_length}, and the M-dim is {'before' if data_config.multi_first else 'after'} the batch dimension B")

	return data_config

# Load the required generation configuration
def load_generation_config(cfg: omegaconf.DictConfig, **default_kwargs) -> infer.GenerationConfig:

	if cfg.gencfg:
		gencfg_spec = cfg.gencfg
	else:
		default_gencfg = dict(method='greedy', topk=1, vocab_prior=False, vocab_per_token=False, vocab_scaler=0, guided=False, guide_renorm=False, temperature=1, length_alpha=0)
		gencfg_spec = infer.GenerationConfig(**{**default_gencfg, **default_kwargs}).name

	gencfg = infer.GenerationConfig.from_name(name=gencfg_spec)
	log.info(f"Using generation config: {gencfg.name}")
	return gencfg

# Load the required list of generation configurations with some deduplication and preprocessing
def load_generation_configs(cfg: omegaconf.DictConfig, **default_kwargs) -> tuple[infer.GenerationConfig, ...]:

	gencfg_specs = []  # Note: We go from values to string to generation config in order to ensure that the result is exactly equivalent to specifying the generation config by name in future runs
	if cfg.gencfgs:
		gencfg_specs.extend(cfg.gencfgs)
	if cfg.gencfgs_grid:
		for method in cfg.gencfg_method:
			is_greedy = (method == 'greedy')
			for topk in cfg.gencfg_topk:
				if is_greedy:
					topk = 1
				for prior in cfg.gencfg_prior:
					if prior == 'none' or is_greedy:
						vocab_prior, vocab_per_token, vocab_scaler = False, False, 0
					else:
						vocab_prior = True
						match = re.fullmatch(r'(tok|tgt)(.*)', prior)
						try:
							vocab_per_token = (match.group(1) == 'tok')
							vocab_scaler = float(match.group(2))
						except (AttributeError, ValueError):
							raise ValueError(f"Failed to parse generation configuration prior: {prior}")
					for guide in cfg.gencfg_guide:
						if guide not in ('none', 'plain', 'renorm'):
							raise ValueError(f"Invalid generation configuration guiding specification: {guide}")
						if is_greedy and guide == 'renorm':
							guide = 'plain'  # Note: Changes target noun scores but NOT which target noun is decoded
						guided = (guide != 'none' or method == 'all')
						guide_renorm = (guide == 'renorm')
						for tau in cfg.gencfg_tau:
							if is_greedy:
								tau = 1  # Note: Changes target noun scores but NOT which target noun is decoded
							for alpha in cfg.gencfg_alpha:
								if is_greedy:
									alpha = 0  # Note: Changes target noun scores but NOT which target noun is decoded
								gencfg_specs.append(infer.GenerationConfig(method=method, topk=topk, vocab_prior=vocab_prior, vocab_per_token=vocab_per_token, vocab_scaler=vocab_scaler, guided=guided, guide_renorm=guide_renorm, temperature=tau, length_alpha=alpha).name)

	if gencfg_specs:
		gencfg_specs = dict.fromkeys(gencfg_specs)  # Deduplicate without losing order (all dict values are None)
		gencfgs = tuple(infer.GenerationConfig.from_name(name=gencfg_spec) for gencfg_spec in gencfg_specs)
	else:
		gencfgs = (load_generation_config(cfg=cfg, **default_kwargs),)

	log.info(f"Collected {len(gencfgs)} generation configurations")
	return gencfgs

# Precompute data required for generation
def model_precompute(model: embedding_decoder.EmbeddingDecoder, gencfg: infer.GenerationConfig, vocab_targets: Optional[torch.Tensor], guide_targets: Optional[torch.Tensor], precompute_cache: Optional[dict[tuple[Any, ...], Any]] = None) -> Any:

	if gencfg.vocab_prior and vocab_targets is None:
		raise ValueError("Generation config specifies to use vocab priors but no vocab targets were provided")
	if gencfg.guided and guide_targets is None:
		raise ValueError("Generation config is guided but no guide targets were provided")

	if gencfg.method == 'all':
		if not gencfg.guided:
			raise ValueError("The 'all' generation method must always be guided")
		precompute_method = model.precompute_generate_all
		precompute_kwargs = dict(length_alpha=gencfg.length_alpha, vocab_targets=vocab_targets if gencfg.vocab_prior else None, vocab_per_token=gencfg.vocab_per_token, vocab_scaler=gencfg.vocab_scaler, guide_targets=guide_targets, guide_renorm=gencfg.guide_renorm)
	else:
		raise ValueError(f"Unsupported generation method for precomputation: {gencfg.method}")

	if precompute_cache is None:
		precompute = precompute_method(**precompute_kwargs)
	else:
		precompute_key = (gencfg.method, *precompute_kwargs.values())  # Note: Tensors hash by identity
		if (precompute := precompute_cache.get(precompute_key, None)) is None:
			precompute = precompute_method(**precompute_kwargs)
			precompute_cache[precompute_key] = precompute

	return precompute

# Generate the output of a model given a particular generation configuration
def model_generate(model: embedding_decoder.EmbeddingDecoder, embeds: torch.Tensor, gencfg: infer.GenerationConfig, vocab_targets: Optional[torch.Tensor], guide_targets: Optional[torch.Tensor], precompute: Optional[Any] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	# Returns BxKxC tensor of token IDs, BxKxC tensor of token paddings, and BxK tensor of scores in descending order for each individual K (K = 1 for greedy)
	# Note: vocab_targets and/or guide_targets can be None if gencfg does not require them
	# Note: precompute can only be provided for the 'all' generation method
	# Note: It is assumed that this method is called within torch inference mode, and an appropriate entered model AMP context

	if gencfg.vocab_prior and vocab_targets is None:
		raise ValueError("Generation config specifies to use vocab priors but no vocab targets were provided")
	if gencfg.guided and guide_targets is None:
		raise ValueError("Generation config is guided but no guide targets were provided")
	if precompute is not None and gencfg.method != 'all':
		raise ValueError("Precomputed generation data can only be provided for the 'all' generation method")

	if gencfg.method == 'greedy':
		if gencfg.topk != 1:
			raise ValueError(f"Top-k must be 1 for greedy generation: {gencfg.topk}")
		if gencfg.vocab_prior:
			raise ValueError("Greedy generation does not support vocab priors")
		target, target_padding, _, _, _, target_score = model.generate(embed=embeds, collect_logits=False, calc_loss=True, temperature=gencfg.temperature, length_alpha=gencfg.length_alpha, sample_weight=None, guide_targets=guide_targets if gencfg.guided else None, guide_renorm=gencfg.guide_renorm)
		target = target.unsqueeze(dim=1)
		target_padding = target_padding.unsqueeze(dim=1)
		target_score = target_score.unsqueeze(dim=1)
	elif gencfg.method == 'beam':
		target, target_padding, target_score = model.generate_beam(embed=embeds, topk=gencfg.topk, temperature=gencfg.temperature, length_alpha=gencfg.length_alpha, vocab_targets=vocab_targets if gencfg.vocab_prior else None, vocab_per_token=gencfg.vocab_per_token, vocab_scaler=gencfg.vocab_scaler, guide_targets=guide_targets if gencfg.guided else None, guide_renorm=gencfg.guide_renorm)
	elif gencfg.method == 'all':
		if not gencfg.guided:
			raise ValueError("The 'all' generation method must always be guided")
		target, target_padding, target_score = model.generate_all(embed=embeds, topk=gencfg.topk, temperature=gencfg.temperature, length_alpha=gencfg.length_alpha, vocab_targets=vocab_targets if gencfg.vocab_prior else None, vocab_per_token=gencfg.vocab_per_token, vocab_scaler=gencfg.vocab_scaler, guide_targets=guide_targets, guide_renorm=gencfg.guide_renorm, precompute=precompute)
	else:
		raise ValueError(f"Unsupported generation method: {gencfg.method}")

	return target, target_padding, target_score

# Load the required list of embedding decoder model checkpoint paths (reverse-sorted absolute paths that cannot be empty strings)
def load_checkpoint_paths(cfg: omegaconf.DictConfig, hydra_dir: Optional[str] = None, allow_empty: bool = False) -> list[str]:

	hydra_dirname = os.path.dirname(hydra_dir) if hydra_dir is not None else None

	model_paths = set()
	for path in itertools.chain((cfg.load_model,), (cfg.load_models,) if isinstance(cfg.load_models, str) else cfg.load_models):

		if not path:
			continue

		if hydra_dirname is not None and not os.path.isabs(path) and not os.path.exists(path) and os.path.exists(hydra_path := os.path.join(hydra_dirname, path)):
			path = hydra_path
		path = os.path.abspath(path)

		if os.path.isdir(path):
			dir_model_paths = set()
			for entry in os.listdir(path):
				entry_path = os.path.join(path, entry)
				if os.path.isfile(entry_path) and (entry.endswith('.model') or entry.endswith('.train')):
					dir_model_paths.add(entry_path)
			if cfg.load_models_dirnum > 0:
				model_paths.update(sorted(dir_model_paths, reverse=True)[:cfg.load_models_dirnum])
			else:
				model_paths.update(dir_model_paths)
		else:
			model_paths.add(path)

	if not model_paths:
		msg = "Specify model checkpoint files and/or directories to process using 'load_model' and 'load_models'"
		if allow_empty:
			log.warning(msg)
		else:
			raise ValueError(msg)

	log.info(f"Collected {len(model_paths)} model paths")
	return sorted(model_paths, reverse=True)

# Load an embedding decoder model checkpoint
def load_decoder_checkpoint(cfg: omegaconf.DictConfig, hydra_dir: Optional[str] = None, checkpoint_path: Optional[str] = None, target_config: Optional[embedders.TargetConfig] = None, data_config: Optional[embedding_dataset.DataConfig] = None) -> tuple[Optional[dict[str, Any]], Optional[str]]:

	if checkpoint_path is None:
		checkpoint_path = cfg.load_model
		if not checkpoint_path:
			return None, None
	elif not checkpoint_path:
		raise ValueError("Cannot explicitly load a decoder checkpoint corresponding to an empty string")

	if hydra_dir is not None and not os.path.isabs(checkpoint_path) and not os.path.exists(checkpoint_path) and os.path.exists(hydra_checkpoint_path := os.path.join(os.path.dirname(hydra_dir), checkpoint_path)):
		checkpoint_path = hydra_checkpoint_path
	checkpoint_path = os.path.abspath(checkpoint_path)

	checkpoint = torch.load(checkpoint_path, map_location='cpu')  # Note: We load to CPU for more control of GPU memory spikes, and because target configuration has tensors that need to stay on CPU

	check_loaded_config(name='hydra config', using=utils_config.flatten_config(cfg), loaded=checkpoint['cfg_flat'], ignore=IGNORE_CFG_DIFFS)
	if target_config is not None:
		check_loaded_config(name='target config', using=dataclasses.asdict(target_config), loaded=checkpoint['target_config'])
	if data_config is not None:
		check_loaded_config(name='data config', using=dataclasses.asdict(data_config), loaded=checkpoint['data_config'])

	return checkpoint, checkpoint_path

# Load an embedding decoder model checkpoint for generation
def load_decoder_checkpoint_generate(cfg: omegaconf.DictConfig, embedder: embedders.Embedder, model_path: str, device: torch.device, device_is_cpu: bool) -> tuple[dict[str, Any], str, tuple[str, ...], set[str], torch.Tensor]:

	log.info(f"Loading model: {model_path}")
	checkpoint, checkpoint_path = load_decoder_checkpoint(cfg=cfg, checkpoint_path=model_path)
	load_target_config(checkpoint=checkpoint, embedder=embedder)

	model_targets = embedder.target_vocab
	model_targets_set = set(model_targets)
	vocab_targets = embedder.tokenize_target(model_targets)[0]
	if not device_is_cpu:
		vocab_targets = vocab_targets.to(device=device, non_blocking=True)

	return checkpoint, checkpoint_path, model_targets, model_targets_set, vocab_targets

# Check a configuration dict against a loaded one and warn if anything is different
def check_loaded_config(name: str, using: dict[str, Any], loaded: dict[str, Any], ignore: Optional[Iterable[str]] = None):

	issue = False

	using_keys = set(using)
	loaded_keys = set(loaded)
	common_keys = using_keys & loaded_keys
	using_keys.difference_update(common_keys)
	loaded_keys.difference_update(common_keys)
	if ignore:
		common_keys.difference_update(ignore)

	if using_keys:
		log.warning(f"Loaded {name}s did not include configs: {', '.join(sorted(using_keys))}")
		issue = True
	if loaded_keys:
		log.warning(f"Loaded {name}s included unused configs: {', '.join(sorted(loaded_keys))}")
		issue = True

	name_cap = name[0].upper() + name[1:]
	for key in sorted(common_keys):

		using_value = using[key]
		loaded_value = loaded[key]
		using_value_type = type(using_value)
		loaded_value_type = type(loaded_value)

		if using_value_type != loaded_value_type:
			log.warning(f"{name_cap} {key} has type mismatch: Loaded {utils.get_class_str(loaded_value_type)} vs Using {utils.get_class_str(using_value_type)}")
			issue = True

		if isinstance(using_value, torch.Tensor) or isinstance(loaded_value, torch.Tensor):
			unequal = not (using_value_type == loaded_value_type and using_value.dtype == loaded_value.dtype and torch.equal(using_value, loaded_value))
		else:
			unequal = (using_value != loaded_value)

		if unequal:
			loaded_value_str = repr(loaded_value) if isinstance(loaded_value, str) else format(loaded_value)
			using_value_str = repr(using_value) if isinstance(using_value, str) else format(using_value)
			log.warning(f"{name_cap} {key} has value mismatch: Loaded {loaded_value_str} vs Using {using_value_str}")
			issue = True

	if issue:
		log.warning(f"Mismatches were detected in loaded and in-use {name}s => Verify this is okay!")
	else:
		log.info(f"Loaded and in-use {name}s are identical")

# Load the target configuration from an already loaded embedding decoder model checkpoint
def load_target_config(checkpoint: dict[str, Any], embedder: embedders.Embedder) -> embedders.TargetConfig:
	log.info(f"Loading {len(checkpoint['target_config'])} target configuration items from checkpoint...")
	target_config = utils.dataclass_from_dict(cls=embedders.TargetConfig, state=checkpoint['target_config'])
	embedder.configure_target(target_config=target_config, target_vocab=checkpoint['target_nouns'][checkpoint['num_invalid_target_nouns']:])
	return target_config

# Load the data configuration from an already loaded embedding decoder model checkpoint
def load_data_config(checkpoint: dict[str, Any], dataset: Optional[embedding_dataset.EmbeddingDataset] = None) -> embedding_dataset.DataConfig:
	# Target config must be available in the dataset embedder if a dataset is provided
	log.info(f"Loading {len(checkpoint['data_config'])} data configuration items from checkpoint...")
	data_config = utils.dataclass_from_dict(cls=embedding_dataset.DataConfig, state=checkpoint['data_config'])
	if dataset:
		dataset.configure_data(data_config)
	return data_config

# Load a data loader for an embedding dataset
def load_embedding_dataset_loader(cfg: omegaconf.DictConfig, dataset: embedding_dataset.EmbeddingDataset, training: bool, device: torch.device, patch: bool = True) -> tuple[torch.utils.data.DataLoader, embedding_dataset.LoaderInfo]:
	# The embedder needs a valid target_config for this function, and the dataset must have a valid data_config

	num_workers = cfg.dataset_workers
	if cfg.determ or utils.debugger_attached():
		log.warning("Not using any dataset workers due to determinism or debugger")
		num_workers = 0
	elif num_workers > cfg.batch_size:
		num_workers = cfg.batch_size  # Note: Just to clarify, each worker produces complete batches and not just individual samples that are later merged into batches cross-process, so this is a pure heuristic not a fundamental limit

	log.info(f"Creating {'' if patch else 'unpatched '}embedding dataset loader in {'TRAIN' if training else 'EVAL'} mode with batch size {cfg.batch_size} and {num_workers} workers...")
	loader, loader_info = dataset.create_loader(batch_size=cfg.batch_size, num_workers=num_workers, training=training, device=device, patch=patch)

	log.info(f"Dataset: {dataset.num_embeds} embeddings across {dataset.num_items} items")
	log.info(f"Loader: {loader_info.epoch_samples}/{loader_info.available_samples} samples used in {loader_info.complete_batches}+{int(loader_info.incomplete_batch)} = {loader_info.epoch_batches} batches of size {loader_info.batch_size}+{loader_info.batch_size_last}")
	log.info(f"Loader: {loader_info.num_workers} workers prefetching {loader_info.prefetch_factor} {'pinned' if loader_info.pin_memory else 'unpinned'} {device.type.upper() if loader_info.on_device else 'CPU'} batches each")

	return loader, loader_info

# Load an image classification dataset
def load_cls_dataset(cfg: omegaconf.DictConfig, embedder: embedders.Embedder, device_is_cpu: bool, cls_dataset: Optional[str] = None, variant: Optional[str] = None, clean: Optional[bool] = None, paths: bool = False, path_optional: bool = False) -> tuple[classification_dataset.ClassificationDataset, torch.utils.data.DataLoader, str, bool]:
	if cls_dataset is None:
		cls_dataset = cfg.cls_dataset
	cls_variant = cfg.class_names_variant if variant is None else variant
	cls_clean = (cls_variant == 'clip') if clean is None else clean
	dataset = classification_dataset.load_image_dataset(name=cls_dataset, root_path=cfg.cls_dataset_root, split=cfg.cls_split, variant=cls_variant, clean=cls_clean, image_transform=embedder.get_image_transform(), paths=paths, path_optional=path_optional)
	loader = classification_dataset.load_image_dataset_loader(dataset=dataset, batch_size=embedder.image_batch_size, num_workers=cfg.dataset_workers, device_is_cpu=device_is_cpu)
	return dataset, loader, cls_variant, cls_clean

# Align the classes in an image classification dataset with the noun dataset
def align_cls_classes(cfg: omegaconf.DictConfig, embedder: embedders.Embedder, dataset: classification_dataset.ClassificationDataset, dataset_noun: Optional[noun_dataset.NounDataset] = None) -> tuple[list[list[str]], tuple[str, ...]]:

	if loaded_nouns := (dataset_noun is None):
		dataset_noun = load_embedding_dataset(cfg=cfg, embedder=embedder, embed_dataset='NounDataset', use_targets=False, use_cache=False)  # Note: Vocab JSON is affected by the current vocab threshold

	error = False
	cls_class_lists = []
	min_cls_class = min_cls_freq = None
	log.info(f"Aligning classification dataset classes with noun dataset of vocab threshold {dataset_noun.vocab_thres}...")
	with tqdm.tqdm(desc='Aligning classes', total=len(dataset.cls_classes), unit='class', unit_scale=False, dynamic_ncols=True, smoothing=0.08) as progress_bar:  # noqa

		for raw_cls_classes in dataset.cls_classes:

			cls_classes: list[str] = []
			for raw_cls_class in raw_cls_classes.split(','):
				raw_cls_class = raw_cls_class.strip()
				cls_class = utils.get_canon(raw_cls_class, sanitize=True)
				cls_class_gen = ((vocab['target_noun'], vocab['singulars_freq_sum'] + vocab['plurals_freq_sum']) for vocab in dataset_noun.vocab_json if cls_class in vocab['singulars'] or cls_class in vocab['plurals'])
				try:
					cls_class, cls_freq = next(cls_class_gen)
					if cls_class not in cls_classes:
						cls_classes.append(cls_class)
					if min_cls_freq is None or cls_freq < min_cls_freq:
						min_cls_class = cls_class
						min_cls_freq = cls_freq
					try:
						cls_class_alt, cls_freq_alt = next(cls_class_gen)
						log.error(f"Class '{raw_cls_class}' cannot uniquely be resolved to a target noun in the noun dataset vocabulary: {cls_class} vs {cls_class_alt}")
						error = True
					except StopIteration:
						pass
				except StopIteration:
					log.warning(f"Ignoring class '{raw_cls_class}' as it is not present in the noun dataset vocabulary")

			if not cls_classes:
				log.error(f"Empty class names entry because none of the following are in the noun dataset vocabulary: {raw_cls_classes}")
				error = True

			cls_class_lists.append(cls_classes)
			progress_bar.update(n=1)

		assert progress_bar.n == progress_bar.total

	log.info(f"Minimum encountered target frequency was {min_cls_freq} for '{min_cls_class}' (and possibly other target nouns)")
	if error:
		raise ValueError("Failed to match classification dataset class names to noun dataset target nouns")

	if loaded_nouns:
		log.info("Unloading noun dataset")

	targets = tuple(sorted(set(cls for cls_classes in cls_class_lists for cls in cls_classes)))
	return cls_class_lists, targets

# Align the classes of an image classification dataset with a collection of available target nouns (e.g. available model targets) using a vocab mapping
def align_cls_class_targets(cfg: omegaconf.DictConfig, embedder: embedders.Embedder, cls_classes: Sequence[str], targets: Sequence[str], vocab_id_map: Optional[dict[str, set[int]]] = None) -> tuple[list[list[str]], tuple[str, ...]]:

	if vocab_id_map is None:
		vocab_id_map = load_vocab_id_map(cfg=cfg, embedder=embedder)

	id_targets_map = {}
	missing_targets = set()
	duplicate_targets = set()
	for target in targets:
		if (vid_set := vocab_id_map.get(target, None)) is None:
			missing_targets.add(target)
		else:
			for vid in vid_set:
				if (target_set := id_targets_map.get(vid, None)) is None:
					id_targets_map[vid] = {target}
				elif target in target_set:
					duplicate_targets.add(target)
				else:
					target_set.add(target)
	if missing_targets:
		log.warning(f"{len(missing_targets)} targets failed to map to a vocab ID: {', '.join(sorted(missing_targets))}")
	if duplicate_targets:
		log.warning(f"{len(duplicate_targets)} targets were duplicate: {', '.join(sorted(duplicate_targets))}")

	cls_class_lists = []
	missing_raw_cls_classes = set()
	missing_raw_class_targets = set()
	for csv_classes in cls_classes:
		main_class = None
		aligned_classes: set[str] = set()  # noqa
		for raw_cls_class in csv_classes.split(','):
			raw_cls_class = raw_cls_class.strip()
			cls_class = utils.get_canon(raw_cls_class, sanitize=True)
			if (vid_set := vocab_id_map.get(cls_class, None)) is None:
				missing_raw_cls_classes.add(raw_cls_class)
			else:
				matched_target = False
				for vid in vid_set:
					if target_set := id_targets_map.get(vid, None):
						if not aligned_classes:
							main_class = min(target_set)
						aligned_classes.update(target_set)
						matched_target = True
				if not matched_target:
					missing_raw_class_targets.add(raw_cls_class)
		cls_class_lists.append(sorted(aligned_classes, key=lambda x: (x != main_class, x)))
	if missing_raw_cls_classes:
		log.warning(f"{len(missing_raw_cls_classes)} cls classes failed to map to a vocab ID: {', '.join(sorted(missing_raw_cls_classes))}")
	if missing_raw_class_targets:
		log.warning(f"{len(missing_raw_class_targets)} cls classes failed to map to an available target noun: {', '.join(sorted(missing_raw_class_targets))}")

	cls_targets = tuple(sorted(set(cls for cls_classes in cls_class_lists for cls in cls_classes)))
	return cls_class_lists, cls_targets

# Load the vocab ID map (unthresholded unless a thresholded dataset_noun is explicitly passed)
def load_vocab_id_map(cfg: omegaconf.DictConfig, embedder: embedders.Embedder, dataset_noun: Optional[noun_dataset.NounDataset] = None) -> dict[str, set[int]]:
	# Returns a dictionary that maps each known noun (whether it's a main target noun or not, whether singular or plural) to the set of integer vocab IDs that contain it (almost always unique, but not always e.g. axes --> ax/axis)

	if dataset_noun is None:
		vocab_json = load_embedding_dataset(cfg=cfg, embedder=embedder, embed_dataset='NounDataset', use_targets=False, vocab_thres=0, use_cache=False).vocab_json
		log.info("Unloaded noun dataset")
	else:
		vocab_json = dataset_noun.vocab_json

	vocab_id_map = {}
	for vocab in vocab_json:
		vocab_id = vocab['id']
		for noun in itertools.chain(vocab['singulars'], vocab['plurals']):
			if (vid_set := vocab_id_map.get(noun, None)) is None:
				vocab_id_map[noun] = {vocab_id}
			else:
				vid_set.add(vocab_id)
	assert sum(len(vid_set) for vid_set in vocab_id_map.values()) == sum(len({vocab['target_noun']}.union(vocab['singulars'], vocab['plurals'])) for vocab in vocab_json)

	return vocab_id_map

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

# Load AMP for the embedding decoder model
def load_decoder_amp(cfg: omegaconf.DictConfig, device: torch.device) -> tuple[ContextManager, Optional[torch.dtype]]:

	amp_enabled = cfg.amp
	if amp_enabled and (cfg.determ or device.type == 'cpu'):
		log.warning("Decoder AMP was automatically disabled due to determinism or CPU device")
		amp_enabled = False

	if amp_enabled:
		amp_context = torch.autocast(device.type, dtype=torch.bfloat16 if cfg.amp_bf16 else None)
		amp_dtype = amp_context.fast_dtype
		log.info(f"Decoder AMP is enabled with dtype {amp_dtype}")
	else:
		amp_context = contextlib.nullcontext()
		amp_dtype = None
		log.info("Decoder AMP is disabled")

	return amp_context, amp_dtype

# Load the embedding decoder model
def load_decoder_model(cfg: omegaconf.DictConfig, embedder: embedders.Embedder, data_config: embedding_dataset.DataConfig, checkpoint: Optional[dict[str, Any]]) -> embedding_decoder.EmbeddingDecoder:

	if checkpoint is not None and (model_cfg_flat := {key: value for key, value in checkpoint['cfg_flat'].items() if key in MODEL_CFGS and key in cfg and getattr(cfg, key) != value}):
		log.warning(f"Overriding config values based on checkpoint: {' '.join('{key}={value}'.format(key=key, value=format(value, '.3g') if isinstance(value, float) else value) for key, value in model_cfg_flat.items())}")
		cfg = copy.deepcopy(cfg)
		for key, value in model_cfg_flat.items():
			omegaconf.OmegaConf.update(cfg=cfg, key=key, value=value, merge=False)

	model_class: Type[embedding_decoder.EmbeddingDecoder] = getattr(embedding_decoder, cfg.model)
	log.info(f"Creating model of class {model_class.__qualname__}...")

	assert embedder.target_config is not None  # Note: If this triggers then you forgot to call embedder.configure_target() prior to calling the current function

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

# Finalise the embedding decoder model
def finalise_decoder_model(cfg: omegaconf.DictConfig, model: embedding_decoder.EmbeddingDecoder) -> embedding_decoder.EmbeddingDecoder:
	if cfg.compile:
		log.info("Will compile decoder model when it is used")
		model = torch.compile(model)
	return model

# Prepare the embedding decoder model for evaluation
def prepare_decoder_model_eval(cfg: omegaconf.DictConfig, embedder: embedders.Embedder, data_config: embedding_dataset.DataConfig, checkpoint: Optional[dict[str, Any]], device: torch.device, device_is_cpu: bool, eval_train: bool) -> embedding_decoder.EmbeddingDecoder:

	model = load_decoder_model(cfg=cfg, embedder=embedder, data_config=data_config, checkpoint=checkpoint)
	if not device_is_cpu:
		log.info(f"Moving model to {device.type.upper()}...")
		model.to(device=device)
	model = finalise_decoder_model(cfg=cfg, model=model)

	if eval_train:
		log.info("Evaluating model in TRAIN mode (e.g. dropout may be active)")
		model.train()
	else:
		log.info("Evaluating model in EVAL mode")
		model.eval()

	return model

# Load prediction JSON files
def load_predictions(cfg: omegaconf.DictConfig, pred_json_paths: Optional[Sequence[str]] = None) -> dict[str, dict[str, Any]]:

	if pred_json_paths is None:
		pred_json_paths = cfg.load_pred_jsons

	pred_json_files = set()
	for path in pred_json_paths:
		if path:
			path = os.path.abspath(resolve_source_path(path))
			if os.path.isdir(path):
				pred_json_files.update(os.path.join(path, json_file) for json_file in fnmatch.filter(os.listdir(path), '*.json'))
			else:
				if not path.endswith('.json'):
					raise ValueError(f"Predictions must be JSON files: {path}")
				pred_json_files.add(path)
	pred_json_files = sorted(pred_json_files)

	pred_jsons = {}
	for path in pred_json_files:
		log.info(f"Loading predictions JSON: {path}")
		with open(path, 'r') as file:
			pred_jsons[path] = json.load(file)
	log.info(f"Loaded {len(pred_jsons)} predictions JSONs")

	return pred_jsons

# Load a sample annotations file
def load_sample_annotations(ann_json: str, image_dir: Optional[str], update_samples: Optional[Iterable[str]] = None) -> tuple[dict[str, dict[str, set[str]]], dict[str, None]]:

	if image_dir is None:
		log.info(f"Loading sample annotations...")
	else:
		log.info(f"Loading sample annotations for: {resolve_source_path(image_dir)}")

	if not ann_json:
		return {}, {}

	if image_dir is not None and not os.path.isabs(ann_json) and os.path.commonpath(paths=(ann_json, IMAGE_DIR_TAG)):
		ann_json = image_dir + ann_json[len(IMAGE_DIR_TAG):]
	ann_json = resolve_source_path(ann_json)

	with open(ann_json, 'r') as file:
		class_annotations = json.load(file)
	categories = {category: None for annotation in class_annotations.values() for category in annotation.keys()}  # Dict so that 'in' operation is efficient while maintaining order

	if update_samples is not None:
		new_samples = set()
		for sample in update_samples:
			if sample not in class_annotations:
				class_annotations[sample] = {category: [] for category in categories}
				new_samples.add(sample)
		if new_samples:
			class_annotations = dict(sorted(class_annotations.items()))
			with open(ann_json, 'w') as file:
				utils.json_dump(class_annotations, file, indent=2)
			log.info(f"Updated annotations file with {len(new_samples)} new samples: {ann_json}")

	class_annotations = {sample: {category: set(classes) for category, classes in annotation.items()} for sample, annotation in class_annotations.items()}
	log.info(f"Loaded {sum(len(classes) for annotation in class_annotations.values() for classes in annotation.values())} class annotations in {len(categories)} categories for {len(class_annotations)} infer images from: {ann_json}")

	return class_annotations, categories

# Apply a custom sorting specification to table rows
def sort_table_rows(table_rows: list[Sequence[Any]], default_order: Optional[Sequence[Any]], table_headers: Sequence[str], sort_spec: Optional[str]):

	if default_order is None:
		table_rows.sort()
	else:
		assert len(default_order) == len(table_rows)
		table_rows[:] = [row for order, row in sorted(zip(default_order, table_rows))]

	if sort_spec:

		sort_spec = sort_spec.lower()
		if sort_spec[0] in ('+', '-'):
			reverse = (sort_spec[0] == '-')
			sort_spec = sort_spec[1:]
		else:
			reverse = False
		if sort_spec[0] == '0':
			numparse = True
			sort_spec = sort_spec[1:]
		else:
			numparse = False

		col_index = next((i for i, header in enumerate(table_headers) if header.lower() == sort_spec), None)
		if col_index is None:
			log.warning(f"Failed to sort by missing case-insensitive column header: {sort_spec}")
			col_index = 0

		if numparse:
			float_regex = re.compile(r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?')
			table_rows.sort(key=lambda x: (float(match.group(0)) if (match := re.search(float_regex, x[col_index])) else -math.inf if reverse else math.inf) if isinstance(x[col_index], str) else x[col_index], reverse=reverse)
		else:
			table_rows.sort(key=lambda x: x[col_index], reverse=reverse)

#
# Miscellaneous
#

# Resolve a path that might contain a source tag in it as the leading path component
def resolve_source_path(path: str) -> str:
	if not path:
		raise ValueError("Path cannot be empty")
	if not os.path.isabs(path) and os.path.commonpath(paths=(path, SOURCE_TAG)):
		return os.path.dirname(os.path.realpath(__file__)) + path[len(SOURCE_TAG):]
	return path

# Make an embedder specification path-safe
def safe_embedder_spec(embedder_spec: str) -> str:
	return embedder_spec.replace(':', '_').replace('/', '_')

# Helper function to format a numeric value 0-100 as a constant width percent string
def format_percent_str(value: Union[float, str]) -> str:
	if not isinstance(value, str):
		value_str = format(value, '.3f')
		if len(value_str) > 6:
			value_str = format(value, '.2f')
		value = value_str
	return f"{value:>6s}%"

# Helper function to format a ratio value 0-1 as a constant width percent string
def format_ratio_str(value: float) -> str:
	return format_percent_str(value * 100)

#
# Run
#

# Run main function
if __name__ == '__main__':
	main()
# EOF
