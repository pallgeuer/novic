# Writers of cached datasets of embeddings and target nouns

# Imports
import os
import json
import random
import fnmatch
import itertools
import contextlib
import collections
import dataclasses
from typing import Sequence, Iterable, Optional, Union, Any
import tqdm
import torch.nn.functional
import torch.utils.data
from logger import log
import utils
import embedders
import embedding_cache
import noun_dataset

# Random cache writer => Writes a given number of random embedding vectors into a cache without any targets
class RandomCacheWriter(embedding_cache.EmbeddingCacheWriter):

	def __init__(self, cache_path: str, embedder: embedders.Embedder, num_embed: int, batch_size: int = 2048):
		self.batch_size = batch_size
		super().__init__(
			cache_path=cache_path,
			embedder=embedder,
			num_embed=num_embed,
			shuffle=False,
			use_targets=False,
			embedder_strict=False,
		)

	def generate(self):

		log.info(f"Generating random cache file with {self.num_embed} embeddings...")
		with self:

			num_left = self.header.embed_num
			while num_left > 0:
				embeds = torch.nn.functional.normalize(torch.randn(min(self.batch_size, num_left), self.header.embed_dim, dtype=self.meta.embed_dtype), dim=-1)
				self.write(embeds=embeds)
				num_left -= embeds.shape[0]

		log.info("Finished generating random cache file")

# Photo cache writer => Writes one embedding per target noun using the prompt template 'a photo of a NOUN'
class PhotoCacheWriter(embedding_cache.EmbeddingCacheWriter):

	def __init__(self, cache_path: str, embedder: embedders.Embedder, target_nouns: Sequence[str], debug: bool = False):
		self.debug = debug
		super().__init__(
			cache_path=cache_path,
			embedder=embedder,
			num_embed=len(target_nouns),
			shuffle=True,
			use_targets=True,
			full_targets=True,
			target_nouns=target_nouns,
			num_embed_targets=1,
			default_weights=True,
			unit_weights=True,
		)

	def generate(self) -> Optional[tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:

		log.info(f"Generating photo prompt cache file with {self.num_embed} embeddings (one for each target noun)...")
		with self.embedder.inference_model(), self:

			if self.debug:
				all_embeds = torch.full(size=(self.num_embed, self.embedder.embed_dim), fill_value=torch.nan, dtype=self.embedder.embed_dtype)
			all_embed_targets = torch.arange(1, self.num_target_nouns, dtype=self.meta.embed_targets_dtype).unsqueeze(dim=1)

			count = 0
			it = itertools.islice(self.target_nouns, 1, None)  # Note: Skip the first item which is just the empty string
			while target_nouns := tuple(itertools.islice(it, self.embedder.inference_batch_size)):

				with self.embedder.inference_mode():
					embeds = self.embedder.inference_text(text=tuple(f'a photo of a {target_noun}' for target_noun in target_nouns))
				batch_size = len(target_nouns)
				new_count = count + batch_size
				embed_targets = all_embed_targets[count:new_count, :]
				embeds = embeds.cpu()  # GPU-CPU synchronization point

				if self.debug:
					assert torch.equal(embed_targets, self.tensorize_embed_targets(embed_targets_str=target_nouns))
					all_embeds[count:new_count, :] = embeds
					print(f"{embeds[:, :8]}\n{embed_targets}")

				self.write(embeds=embeds, embed_targets=embed_targets)
				count = new_count

			if self.debug:
				print(f"{self.target_token_ids}\n{self.target_mask}")
				assert count == self.num_embed == all_embeds.shape[0] == self.target_token_ids.shape[0] - 1 and not all_embeds.isnan().any() and self.target_token_ids.shape[1] == self.embedder.target_config.token_length
				ret = (all_embeds, self.target_token_ids[1:, :], self.target_mask[1:, :] if self.embedder.target_config.use_masks else None)
			else:
				ret = None

		log.info("Finished generating photo prompt cache file")
		return ret

# Index cache writer => Writes dud data for the given target nouns so that cache indexing can be tested
class IndexCacheWriter(embedding_cache.EmbeddingCacheWriter):

	def __init__(self, cache_path: str, embedder: embedders.Embedder, target_nouns: Sequence[str]):
		super().__init__(
			cache_path=cache_path,
			embedder=embedder,
			num_embed=len(target_nouns),
			shuffle=False,
			use_targets=True,
			full_targets=True,
			target_nouns=target_nouns,
			num_embed_targets=1,
			default_weights=True,
			unit_weights=True,
		)

	def generate(self):

		log.info(f"Generating index cache file with {self.num_embed} embeddings (one for each target noun in order)...")
		with self:

			all_embed_targets = torch.arange(1, self.num_target_nouns, dtype=self.meta.embed_targets_dtype).unsqueeze(dim=1)

			count = 0
			it = itertools.islice(self.target_nouns, 1, None)  # Note: Skip the first item which is just the empty string
			while target_nouns := tuple(itertools.islice(it, self.embedder.tokenizer_batch_size)):

				batch_size = len(target_nouns)
				new_count = count + batch_size

				embeds = torch.nn.functional.normalize(torch.randn(batch_size, self.header.embed_dim, dtype=self.meta.embed_dtype), dim=-1)
				embed_targets = all_embed_targets[count:new_count, :]

				self.write(embeds=embeds, embed_targets=embed_targets)
				count = new_count

		log.info("Finished generating index cache file")

# Test multi-target cache writer => Writes random data with random multi-targets and weights
class TestMultiCacheWriter(embedding_cache.EmbeddingCacheWriter):

	def __init__(self, cache_path: str, embedder: embedders.Embedder, num_embed: int, target_nouns: Sequence[str], num_embed_targets: int, batch_size: int = 2048, debug: bool = True):
		self.batch_size = batch_size
		self.debug = debug
		super().__init__(
			cache_path=cache_path,
			embedder=embedder,
			num_embed=num_embed,
			shuffle=False,
			use_targets=True,
			full_targets=False,
			target_nouns=target_nouns,
			num_embed_targets=num_embed_targets,
			default_weights=False,
			unit_weights=False,
		)

	def generate(self):

		log.info(f"Generating random test multi-target cache file with {self.num_embed} embeddings...")
		with self:

			range_tensor = torch.arange(start=self.header.embed_targets_dim, end=0, step=-1)
			num_left = self.header.embed_num
			while num_left > 0:

				batch_size = min(self.batch_size, num_left)
				embeds = torch.nn.functional.normalize(torch.randn(batch_size, self.header.embed_dim, dtype=self.meta.embed_dtype), dim=-1)
				embed_targets = torch.randint(low=1, high=self.num_target_nouns, size=(batch_size, self.header.embed_targets_dim), dtype=self.meta.embed_targets_dtype)
				embed_targets *= ((num_fully_padded := torch.randint(self.header.embed_targets_dim, size=(batch_size, 1))) < range_tensor)
				embed_target_weights = torch.rand(size=(batch_size, self.header.embed_targets_dim), dtype=self.meta.embed_dtype).sort(dim=1, descending=True)[0]
				embed_target_weights *= ((num_fully_padded + torch.randint_like(num_fully_padded, 4).logical_not()).clamp_max_(self.header.embed_targets_dim - 1) < range_tensor)

				if self.debug:
					print("WRITTEN EMBEDS {shape}:".format(shape='\xD7'.join(str(dim) for dim in embeds.shape)))
					print(embeds[:, :8])
					print("WRITTEN EMBED TARGETS {shape}:".format(shape='\xD7'.join(str(dim) for dim in embed_targets.shape)))
					print(embed_targets)
					print("WRITTEN EMBED TARGET WEIGHTS {shape}:".format(shape='\xD7'.join(str(dim) for dim in embed_target_weights.shape)))
					print(embed_target_weights)

				self.write(embeds=embeds, embed_targets=embed_targets, embed_target_weights=embed_target_weights)
				num_left -= embeds.shape[0]

			if self.debug:
				print("WRITTEN TOKEN IDS {shape}:".format(shape='\xD7'.join(str(dim) for dim in self.target_token_ids.shape)))
				print(self.target_token_ids)
				print("WRITTEN TOKEN MASK {shape}:".format(shape='\xD7'.join(str(dim) for dim in self.target_mask.shape)))
				print(self.target_mask)

		log.info("Finished generating random test multi-target cache file")

# Noun dataset cache writer => Writes the contents of a noun dataset to a cache file in shuffled order
class NounDatasetCacheWriter(embedding_cache.EmbeddingCacheWriter):

	def __init__(self, cache_path: str, dataset: noun_dataset.NounDataset):
		self.dataset = dataset
		super().__init__(
			cache_path=cache_path,
			embedder=self.dataset.embedder,
			num_embed=self.dataset.num_embeds,
			shuffle=True,
			use_targets=True,
			full_targets=True,
			target_nouns=self.dataset.targets,
			num_embed_targets=1,
			default_weights=True,
			unit_weights=True,
		)

	def generate(self):

		log.info(f"Converting noun dataset with {self.num_embed} embeddings to an embedding cache file...")
		with self.embedder.inference_model(), self, tqdm.tqdm(desc='Converting noun dataset', total=self.dataset.num_usids, unit='id', unit_scale=False, dynamic_ncols=True, smoothing=0.08) as progress_bar:

			total_usids = 0
			total_fsids = 0
			it = iter(self.dataset.unique_sample())
			while unique_samples := tuple(itertools.islice(it, self.embedder.inference_batch_size)):

				with self.embedder.inference_mode():
					embed_tensor = self.embedder.inference_text(text=tuple(unique_sample.text for unique_sample in unique_samples))
				embed_targets = self.tensorize_embed_targets(tuple(unique_sample.target for unique_sample in unique_samples))
				embed_freq = tuple(unique_sample.freq for unique_sample in unique_samples)
				embed_tensor = embed_tensor.cpu()

				for embed, target, freq in zip(embed_tensor, embed_targets, embed_freq):
					self.write(embeds=embed.expand(freq, -1), embed_targets=target.expand(freq, -1))

				num_samples = len(unique_samples)
				total_usids += num_samples
				total_fsids += sum(embed_freq)
				progress_bar.set_postfix_str(f"{total_fsids}/{self.dataset.num_fsids} FSID", refresh=False)
				progress_bar.update(n=num_samples)

			assert total_usids == self.dataset.num_usids == progress_bar.n and total_fsids == self.dataset.num_fsids

		log.info("Finished converting noun dataset to an embedding cache")

# Noun multiset cache writer => Constructs and writes multi-target embedded prompts based on a noun dataset to a cache file in shuffled order (ignores hypernyms)
class NounMultisetCacheWriter(embedding_cache.EmbeddingCacheWriter):

	def __init__(self, cache_path: str, dataset: noun_dataset.NounDataset, multi_target_freq: Sequence[int]):

		self.dataset = dataset
		self.multi_target_freq = tuple(reversed(tuple(itertools.dropwhile(lambda x: x == 0, reversed(multi_target_freq)))))  # Remove any trailing zero frequencies, and ensure sequence is a tuple

		if self.dataset.hypernym_prompts or self.dataset.total_freq_hypernyms != 1:
			raise ValueError("Noun multiset does not support hypernyms")
		if self.dataset.use_cache:
			raise ValueError("Noun dataset should be configured not to use an internal cache")
		if not self.dataset.use_targets:
			raise ValueError("Noun dataset needs targets in order to generate noun multiset")

		num_multi_target_freqs = len(self.multi_target_freq)
		multi_target_freq_sum = sum(self.multi_target_freq)
		if num_multi_target_freqs <= 0:
			raise ValueError("Non-empty list of multi-target frequencies must be provided")
		if not all(freq >= 0 for freq in self.multi_target_freq):
			raise ValueError(f"All multi-target frequencies must be non-negative: {self.multi_target_freq}")
		if multi_target_freq_sum <= 0:
			raise ValueError("At least one multi-target frequency must be non-zero")

		self.singular_prompts = tuple(prompt for prompt in self.dataset.singular_prompts for _ in range(prompt.freq))
		self.plural_prompts = tuple(prompt for prompt in self.dataset.plural_prompts for _ in range(prompt.freq))
		self.singular_samples = tuple((singular, vocab['target_noun']) for vocab in self.dataset.vocab_json for singular in vocab['singulars'])
		self.plural_samples = tuple((plural, vocab['target_noun']) for vocab in self.dataset.vocab_json for plural in vocab['plurals'])
		self.singular_counts = tuple(count for vocab in self.dataset.vocab_json for count in vocab['singulars_freq'])
		self.plural_counts = tuple(count for vocab in self.dataset.vocab_json for count in vocab['plurals_freq'])
		self.singular_counts_total = sum(self.singular_counts)
		self.plural_counts_total = sum(self.plural_counts)
		assert len(self.singular_samples) == len(self.singular_counts) and len(self.plural_samples) == len(self.plural_counts)
		assert self.singular_counts_total * len(self.singular_prompts) + self.plural_counts_total * len(self.plural_prompts) == self.dataset.num_fsids

		super().__init__(
			cache_path=cache_path,
			embedder=self.dataset.embedder,
			num_embed=self.dataset.num_fsids * multi_target_freq_sum,
			shuffle=True,
			use_targets=True,
			full_targets=(multi_target_freq_sum == self.multi_target_freq[-1]),
			target_nouns=self.dataset.targets,
			num_embed_targets=num_multi_target_freqs,
			default_weights=True,
			unit_weights=True,
		)

	def generate(self):

		def embed_and_write():
			nonlocal total_embed
			with self.embedder.inference_mode():
				embed_tensor = self.embedder.inference_text(text=texts)
			embed_targets = self.tensorize_embed_targets(targets)
			if multi_num == 1:
				verify_counter.update(zip(texts, targets))
			texts.clear()
			targets.clear()
			num_embed = embed_tensor.shape[0]
			total_embed += num_embed
			embed_tensor = embed_tensor.cpu()
			self.write(embeds=embed_tensor, embed_targets=embed_targets)
			progress_bar.set_postfix_str(f"repetition {repetition}/{multi_freq}, total {total_embed}/{self.num_embed}", refresh=False)
			progress_bar.update(n=num_embed)

		log.info(f"Caching noun multiset with freqs {self.multi_target_freq} and {self.num_embed} embeddings to an embedding cache file...")
		with self.embedder.inference_model(), self:

			total_embed = 0

			for multi_num, multi_freq in enumerate(self.multi_target_freq, 1):

				if multi_freq <= 0:
					continue

				with tqdm.tqdm(desc=f'Caching {multi_num}-set', total=self.dataset.num_fsids * multi_freq, unit='embed', unit_scale=False, dynamic_ncols=True, smoothing=0.08) as progress_bar:

					texts = []
					targets = []
					if multi_num == 1:
						verify_counter = collections.Counter()

					for repetition in range(1, multi_freq + 1):
						for prompts, samples, counts, counts_total in ((self.singular_prompts, self.singular_samples, self.singular_counts, self.singular_counts_total), (self.plural_prompts, self.plural_samples, self.plural_counts, self.plural_counts_total)):
							for prompt in prompts:
								for noun_targets in zip(*(random.sample(samples, k=counts_total, counts=counts) for _ in range(multi_num))):

									if prompt.need_article:
										nouns = (self.dataset.make_indefinite(noun_target[0]) for noun_target in noun_targets)
									else:
										nouns = (noun_target[0] for noun_target in noun_targets)
									texts.append(prompt.template.format(noun=' and '.join(nouns)))
									targets.append(tuple(noun_target[1] for noun_target in noun_targets))

									if len(texts) == self.embedder.inference_batch_size:
										embed_and_write()

					if texts:
						embed_and_write()

					assert progress_bar.n == progress_bar.total
					if multi_num == 1:
						dataset_counter = collections.Counter((text, (target_noun,)) for text, target_noun, _, _ in self.dataset)
						for text in dataset_counter:
							dataset_counter[text] *= multi_freq
						assert verify_counter == dataset_counter
						del verify_counter, dataset_counter

			assert total_embed == self.num_embed

		log.info("Finished caching noun multiset")

# Captions cache writer => Writes captions data backed by a noun dataset in shuffled order
class CaptionsCacheWriter(embedding_cache.EmbeddingCacheWriter):

	@dataclasses.dataclass(frozen=True)
	class CaptionsEntry:
		noun_vocab: dict[str, Any]   # Vocab entry from the noun dataset
		singular_prompts: list[str]  # List of all singular prompts ready for formatting and inference (terminating fullstop, correct number as per template multiplier)
		plural_prompts: list[str]    # List of all plural captions ready for formatting and inference (terminating fullstop, correct number as per template multiplier)

	def __init__(self, cache_path: str, captions_path: str, dataset: noun_dataset.NounDataset, template_multiplier: int, sample_multiplier: int, print_approx: int):

		self.captions_path = captions_path
		self.dataset = dataset
		self.sample_multiplier = sample_multiplier
		self.template_multiplier = min(template_multiplier, self.sample_multiplier)
		self.print_approx = max(print_approx, 0)

		assert self.sample_multiplier >= self.template_multiplier
		if self.template_multiplier < 1:
			raise ValueError(f"Multipliers must be at least 1: {self.template_multiplier} and {self.sample_multiplier}")

		target_map = {noun_vocab['target_noun']: noun_vocab for noun_vocab in self.dataset.vocab_json}
		log.info(f"Loaded {len(target_map)} entries from noun dataset")

		with open(self.captions_path, 'r') as file:
			captions_json = json.load(file)
		log.info(f"Temporary memory use for captions JSON is {utils.get_size_mb(captions_json):.1f}MiB")
		if not isinstance(captions_json, list):
			raise TypeError(f"Captions JSON should contain a list: {type(captions_json)}")
		log.info(f"Loaded {len(captions_json)} entries from captions JSON: {self.captions_path}")

		captions_vocab_map = {}
		captions_targets_unused = set()
		count_total_sing = count_total_plural = count_nonnoun_sing = count_nonnoun_plural = 0
		for captions_vocab in captions_json:
			if not isinstance(captions_vocab, dict):
				raise TypeError(f"Entries in the captions JSON should be dicts: {type(captions_vocab)}")
			target = captions_vocab['target_noun']
			if target in captions_vocab_map or target in captions_targets_unused:
				raise ValueError(f"Captions JSON has duplicate target noun: {target}")
			captions_vocab['singular_captions'] = (singular_captions := tuple((caption if caption.endswith('.') else caption + '.') for caption in captions_vocab['singular_captions']))
			captions_vocab['plural_captions'] = (plural_captions := tuple((caption if caption.endswith('.') else caption + '.') for caption in captions_vocab['plural_captions']))
			if not all('{singular}' in caption for caption in singular_captions):
				raise ValueError(f"All singular captions must contain {{singular}}: {target}")
			if not all('{plural}' in caption for caption in plural_captions):
				raise ValueError(f"All plural captions must contain {{plural}}: {target}")
			if len(set(singular_captions)) != len(singular_captions):
				raise ValueError(f"Target noun in captions JSON contains duplicate singular captions: {target}")
			if len(set(plural_captions)) != len(plural_captions):
				raise ValueError(f"Target noun in captions JSON contains duplicate plural captions: {target}")
			count_total_sing += (count_sing := len(singular_captions))
			count_total_plural += (count_plural := len(plural_captions))
			if target in target_map:
				captions_vocab_map[target] = captions_vocab
			else:
				captions_targets_unused.add(target)
				count_nonnoun_sing += count_sing
				count_nonnoun_plural += count_plural

		log.info(f"Captions JSON contains {len(captions_vocab_map)} relevant target nouns and {len(captions_targets_unused)} target nouns that are NOT in the noun dataset")
		missing_targets = set(target_map)
		missing_targets.difference_update(captions_vocab_map)
		if missing_targets:
			log.warning(f"The following {len(missing_targets)} noun dataset target nouns are completely missing in the captions JSON and will NOT occur in the cache: {sorted(missing_targets)}")

		self.captions_entries = []
		bad_empty_sing = set()
		bad_empty_plural = set()
		count_used_sing = count_used_plural = count_unused_sing = count_unused_plural = 0
		for target, captions_vocab in captions_vocab_map.items():
			noun_vocab = target_map[target]
			num_singular_templates = noun_vocab['singulars_freq_sum'] * self.template_multiplier
			num_plural_templates = noun_vocab['plurals_freq_sum'] * self.template_multiplier
			assert num_singular_templates >= 0 and num_plural_templates >= 0 and (num_singular_templates > 0 or num_plural_templates > 0)
			singular_captions = captions_vocab['singular_captions']
			plural_captions = captions_vocab['plural_captions']
			singular_prompts = random.sample(population=singular_captions, k=min(len(singular_captions), num_singular_templates))
			plural_prompts = random.sample(population=plural_captions, k=min(len(plural_captions), num_plural_templates))
			count_used_sing += len(singular_prompts)
			count_used_plural += len(plural_prompts)
			count_unused_sing += len(singular_captions) - len(singular_prompts)
			count_unused_plural += len(plural_captions) - len(plural_prompts)
			if not singular_prompts and num_singular_templates > 0:
				bad_empty_sing.add(target)
			if not plural_prompts and num_plural_templates > 0:
				bad_empty_plural.add(target)
			if singular_prompts or plural_prompts:  # Note: If this is not true, then the target noun must have ended in either bad_empty_sing or bad_empty_plural and is therefore warned about
				self.captions_entries.append(self.CaptionsEntry(noun_vocab=noun_vocab, singular_prompts=singular_prompts, plural_prompts=plural_prompts))

		if bad_empty_sing:
			log.warning(f"The following {len(bad_empty_sing)} target nouns have no singular captions despite needing them: {sorted(bad_empty_sing)}")
		if bad_empty_plural:
			log.warning(f"The following {len(bad_empty_plural)} target nouns have no plural captions despite needing them: {sorted(bad_empty_plural)}")
		self.unique_captions = count_used_sing + count_used_plural
		log.info(f"Of {count_total_sing} + {count_total_plural} = {count_total_sing + count_total_plural} captions in the JSON, {count_nonnoun_sing} + {count_nonnoun_plural} = {count_nonnoun_sing + count_nonnoun_plural} are for target nouns NOT in the noun dataset, {count_used_sing} + {count_used_plural} = {self.unique_captions} will be used, {count_unused_sing} + {count_unused_plural} = {count_unused_sing + count_unused_plural} will NOT be used because certain targets have too many singular/plural captions")
		assert count_total_sing == count_nonnoun_sing + count_used_sing + count_unused_sing and count_total_plural == count_nonnoun_plural + count_used_plural + count_unused_plural
		num_embed = self.sample_multiplier * sum((entry.noun_vocab['singulars_freq_sum'] if entry.singular_prompts else 0) + (entry.noun_vocab['plurals_freq_sum'] if entry.plural_prompts else 0) for entry in self.captions_entries)
		self.print_prob = self.print_approx / num_embed

		super().__init__(
			cache_path=cache_path,
			embedder=self.dataset.embedder,
			num_embed=num_embed,
			shuffle=True,
			use_targets=True,
			full_targets=True,
			target_nouns=self.dataset.targets,
			num_embed_targets=1,
			default_weights=True,
			unit_weights=True,
		)

		log.info(f"Total memory use for captions cache writer is {utils.get_size_mb(self):.1f}MiB")

	def generate(self):

		log.info(f"Converting captions JSON to an embedding cache file with {self.num_embed} embeddings...")
		with self.embedder.inference_model(), self, tqdm.tqdm(desc='Creating captions cache', total=self.num_embed, unit='embed', unit_scale=False, dynamic_ncols=True, smoothing=0.08) as progress_bar:

			pending_embeds = []
			template_dup_counts = collections.Counter()
			caption_dup_counts = collections.Counter()

			entry_iter = iter(enumerate(self.captions_entries, 1))
			while True:

				try:
					entry_index, entry = next(entry_iter)
				except StopIteration:
					entry = None

				if entry is not None:

					orig_pending_count = len(pending_embeds)

					if entry.singular_prompts:
						prompt_index = 0
						num_prompts = len(entry.singular_prompts)
						num_pending_orig = len(pending_embeds)
						for singular, freq in zip(entry.noun_vocab['singulars'], entry.noun_vocab['singulars_freq']):
							for _ in range(freq * self.sample_multiplier):
								pending_embeds.append((entry.singular_prompts[prompt_index].format(singular=singular), entry.noun_vocab['target_noun']))
								prompt_index = (prompt_index + 1) % num_prompts
						dup, rem = divmod(len(pending_embeds) - num_pending_orig, num_prompts)
						template_dup_counts[dup] += num_prompts - rem
						template_dup_counts[dup + 1] += rem

					if entry.plural_prompts:
						prompt_index = 0
						num_prompts = len(entry.plural_prompts)
						num_pending_orig = len(pending_embeds)
						for plural, freq in zip(entry.noun_vocab['plurals'], entry.noun_vocab['plurals_freq']):
							for _ in range(freq * self.sample_multiplier):
								pending_embeds.append((entry.plural_prompts[prompt_index].format(plural=plural), entry.noun_vocab['target_noun']))
								prompt_index = (prompt_index + 1) % num_prompts
						dup, rem = divmod(len(pending_embeds) - num_pending_orig, num_prompts)
						template_dup_counts[dup] += num_prompts - rem
						template_dup_counts[dup + 1] += rem

					caption_dup_counts.update(collections.Counter(itertools.islice(pending_embeds, orig_pending_count, None)).values())

				while pending_embeds and (len(pending_embeds) >= self.embedder.inference_batch_size or entry is None):
					texts, targets = zip(*itertools.islice(pending_embeds, self.embedder.inference_batch_size))
					with self.embedder.inference_mode():
						embed_tensor = self.embedder.inference_text(text=texts)
					embed_targets = self.tensorize_embed_targets(targets)
					chosen_count = len(texts)
					if self.print_approx > 0:
						with tqdm.tqdm.external_write_mode():
							for index in (torch.rand(chosen_count) < self.print_prob).nonzero().squeeze(dim=1).tolist():
								print(f"{targets[index]} = {texts[index]}")
					pending_embeds = pending_embeds[chosen_count:]
					embed_tensor = embed_tensor.cpu()
					self.write(embeds=embed_tensor, embed_targets=embed_targets)
					progress_bar.set_postfix_str(f"entry={entry_index}/{len(self.captions_entries)}", refresh=False)
					progress_bar.update(n=embed_tensor.shape[0])

				if entry is None:
					assert not pending_embeds
					break

			assert progress_bar.n == progress_bar.total and entry_index == len(self.captions_entries)

		unique_captions = sum(template_dup_counts.values())
		log.info(f"Distribution of caption template reuse frequencies: {dict(sorted(item for item in template_dup_counts.items() if item[1] != 0))}")
		log.info(f"Distribution of sample reuse frequencies: {dict(sorted(item for item in caption_dup_counts.items() if item[1] != 0))}")
		log.info(f"Captions embedding cache contains {unique_captions} unique caption templates and {sum(caption_dup_counts.values())} unique samples, for a total of {self.num_embed} samples")
		assert unique_captions == self.unique_captions
		log.info("Finished converting captions JSON to an embedding cache")

# Image classification dataset cache writer => Writes an image classification dataset as a single or multi-target cache in shuffled order
class ClassificationCacheWriter(embedding_cache.EmbeddingCacheWriter):

	def __init__(self, cache_path: str, embedder: embedders.Embedder, loader: torch.utils.data.DataLoader, targets: Sequence[str], class_targets: Sequence[Sequence[str]]):
		self.loader = loader
		self.class_targets = class_targets
		if not self.class_targets or any(not tgts for tgts in self.class_targets):
			raise ValueError("Classification targets argument must be a non-empty sequence of non-empty sequences of target nouns")
		num_embed_targets = max(len(tgts) for tgts in self.class_targets)
		super().__init__(
			cache_path=cache_path,
			embedder=embedder,
			num_embed=len(self.loader.dataset),  # noqa
			shuffle=True,
			use_targets=True,
			full_targets=all(len(tgts) == num_embed_targets for tgts in self.class_targets),
			target_nouns=targets,
			num_embed_targets=num_embed_targets,
			default_weights=True,
			unit_weights=True,
		)

	def generate(self):

		log.info(f"Converting image classification dataset with {self.num_embed} embeddings to an embedding cache file...")
		with self.embedder.inference_model(), self:

			class_embed_targets = self.tensorize_embed_targets(self.class_targets)
			with tqdm.tqdm(desc='Converting cls dataset', total=self.num_embed, unit='img', unit_scale=False, dynamic_ncols=True, smoothing=0.08) as progress_bar:
				for images, cls_indices in self.loader:
					with self.embedder.inference_mode():
						embeds = self.embedder.inference_image(images=images)
					embed_targets = class_embed_targets[cls_indices, :]
					embeds = embeds.cpu()
					self.write(embeds=embeds, embed_targets=embed_targets)
					progress_bar.update(n=embeds.shape[0])
				assert progress_bar.n == progress_bar.total

		log.info("Finished converting image classification dataset to an embedding cache")

# Image cache writer => Writes a collection of images as target-less embeddings in shuffled order
class ImageCacheWriter(embedding_cache.EmbeddingCacheWriter):

	def __init__(self, cache_path: str, embedder: embedders.Embedder, images: Iterable[str], num_workers: int = 8):

		self.num_workers = num_workers

		manual_images = 0
		self.image_paths = []
		for image in images:
			image = os.path.abspath(image)
			if os.path.isdir(image):
				filenames = os.listdir(image)
				orig_count = len(self.image_paths)
				for pattern in utils.IMAGE_PATTERNS:
					self.image_paths.extend(os.path.join(image, filename) for filename in filenames if fnmatch.fnmatch(name=filename.lower(), pat=pattern))
				log.info(f"Found {len(self.image_paths) - orig_count} images in directory: {image}")
			else:
				self.image_paths.append(image)
				manual_images += 1
		if manual_images > 0:
			log.info(f"Found {manual_images} manual images")
		log.info(f"Found a total of {len(self.image_paths)} images")

		super().__init__(
			cache_path=cache_path,
			embedder=embedder,
			num_embed=len(self.image_paths),
			shuffle=True,
			use_targets=False,
		)

	def generate(self):

		log.info(f"Embedding {self.num_embed} images as an embedding cache file...")
		with self.embedder.inference_model(), self:

			image_dataset = utils.ImageDataset(image_paths=self.image_paths, transform=self.embedder.get_image_transform())
			image_loader = image_dataset.create_loader(batch_size=self.embedder.image_batch_size, num_workers=self.num_workers, device_is_cpu=(self.embedder.device.type == 'cpu'))

			with tqdm.tqdm(desc='Embedding images', total=self.num_embed, unit='img', unit_scale=False, dynamic_ncols=True, smoothing=0.08) as progress_bar:
				for image_paths, images in image_loader:
					with self.embedder.inference_mode():
						embeds = self.embedder.inference_image(images=images)
					embeds = embeds.cpu()
					self.write(embeds=embeds)
					progress_bar.update(n=embeds.shape[0])
				assert len(self.image_paths) == self.num_embed == progress_bar.total == progress_bar.n

		log.info("Finished embedding images as an embedding cache file")

# Merge caches writer => Writes the contents of multiple compatible cache files to a new cache file in shuffled order
class MergeCachesWriter(embedding_cache.EmbeddingCacheWriter):

	def __init__(self, cache_path: str, embedder: embedders.Embedder, caches: Sequence[embedding_cache.EmbeddingCache], freqs: Optional[Sequence[int]] = None, use_targets: Optional[bool] = None, multi_mode: Union[str, int] = 'max', batch_size: int = 2048):

		self.caches = caches
		if not self.caches:
			raise ValueError("Need at least one cache to merge")
		if any(cache.embedder is not embedder for cache in self.caches):
			raise ValueError("Caches to merge must be created from the same embedder")
		first_cache = self.caches[0]
		self.batch_size = batch_size

		self.freqs = tuple(1 for _ in self.caches) if freqs is None else freqs
		if len(self.freqs) != len(self.caches) or any(freq < 1 for freq in self.freqs):
			raise ValueError("Mismatch between number of caches to merge and the specified frequencies")

		if use_targets is None:
			use_targets = first_cache.use_targets
		if any(cache.use_targets != use_targets for cache in self.caches):
			raise ValueError("Mismatch between caches to merge in terms of whether to use targets")

		if use_targets:

			if any(cache.target_nouns != first_cache.target_nouns for cache in self.caches):
				raise ValueError("Inconsistent target nouns across caches to merge")
			target_nouns = first_cache.target_nouns[1:]

			with contextlib.suppress(ValueError):
				multi_mode = int(multi_mode)
			if multi_mode == 'min':
				num_embed_targets = min(cache.header.embed_targets_dim for cache in self.caches)
			elif multi_mode == 'max':
				num_embed_targets = max(cache.header.embed_targets_dim for cache in self.caches)
			elif isinstance(multi_mode, int):
				num_embed_targets = multi_mode
			else:
				raise ValueError(f"Unknown multi-mode for merging: {multi_mode}")

			full_targets = all(cache.header.full_targets and num_embed_targets <= cache.header.embed_targets_dim for cache in self.caches)
			unit_weights = all(cache.header.unit_weights and num_embed_targets >= cache.header.embed_targets_dim for cache in self.caches)
			default_weights = False

		else:

			full_targets = True
			target_nouns = None
			num_embed_targets = 0
			default_weights = True
			unit_weights = True

		super().__init__(
			cache_path=cache_path,
			embedder=embedder,
			num_embed=sum(cache.header.embed_num * freq for cache, freq in zip(self.caches, self.freqs)),
			shuffle=True,
			use_targets=use_targets,
			full_targets=full_targets,
			target_nouns=target_nouns,
			num_embed_targets=num_embed_targets,
			default_weights=default_weights,
			unit_weights=unit_weights,
		)

		assert self.target_nouns == first_cache.target_nouns

	def generate(self):

		num_caches = len(self.caches)
		log.info(f"Shuffle-merging {sum(self.freqs)} caches ({num_caches} unique) into a single embedding cache for a total of {self.num_embed} embeddings...")

		with self:

			for cache_num, (cache, freq) in enumerate(zip(self.caches, self.freqs), 1):

				cache_embed_num = cache.header.embed_num
				complete_batches, incomplete_samples = divmod(cache_embed_num, self.batch_size)
				num_batches = complete_batches + (incomplete_samples > 0)

				with cache, tqdm.tqdm(desc=f'Merging input cache {cache_num}/{num_caches}', total=cache_embed_num * freq, unit='embed', unit_scale=False, dynamic_ncols=True, smoothing=0.08, postfix=dict(freq=freq)) as progress_bar:

					for batch_id in range(num_batches):

						start = batch_id * self.batch_size
						stop = min(start + self.batch_size, cache_embed_num)
						batch_size = stop - start
						assert batch_size >= 1

						embeds = torch.frombuffer(cache.cache.view, dtype=cache.meta.embed_dtype, count=batch_size * cache.header.embed_dim, offset=cache.meta.embed_offset + start * cache.meta.embed_stride).view(batch_size, cache.header.embed_dim)
						if self.use_targets:
							embed_targets = cache.embed_targets[start:stop, :]
							embed_target_weights = cache.embed_target_weights[start:stop, :]
							if cache.header.embed_targets_dim < self.header.embed_targets_dim:
								embed_targets_padded = torch.zeros((batch_size, self.header.embed_targets_dim), dtype=embed_targets.dtype)
								embed_targets_padded[:, :cache.header.embed_targets_dim] = embed_targets
								embed_targets = embed_targets_padded
								embed_target_weights_padded = torch.zeros((batch_size, self.header.embed_targets_dim), dtype=embed_target_weights.dtype)
								embed_target_weights_padded[:, :cache.header.embed_targets_dim] = embed_target_weights
								embed_target_weights = embed_target_weights_padded
							elif cache.header.embed_targets_dim > self.header.embed_targets_dim:
								embed_targets = embed_targets[:, :self.header.embed_targets_dim]
								embed_target_weights = embed_target_weights[:, :self.header.embed_targets_dim]
							for _ in range(freq):
								self.write(embeds=embeds, embed_targets=embed_targets, embed_target_weights=embed_target_weights)
						else:
							for _ in range(freq):
								self.write(embeds=embeds)

						progress_bar.update(n=embeds.shape[0] * freq)

					assert progress_bar.n == progress_bar.total
					del embeds, embed_targets, embed_target_weights  # Note: Do not want to have these tensors around once the cache has been closed (e.g. SIGSEGV during debugging)

		log.info("Finished shuffle-merging embedding caches")
# EOF
