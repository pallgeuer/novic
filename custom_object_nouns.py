# Customize object nouns for a particular scenario

# Imports
import os
import re
import json
import argparse
import functools
import unidecode
import dataclasses
from typing import Any, Union, Optional
import pydantic
from gpt_batch_api import task_manager, gpt_requester, utils as gba_utils

#
# Customize object nouns task
#

# Object noun information class (structured outputs schema)
class ObjectNounInfo(pydantic.BaseModel):
	model_config = pydantic.ConfigDict(strict=True)
	object_noun: str = pydantic.Field(title='Object noun', description="The object noun in question.")
	formatted_noun: str = pydantic.Field(title='Formatted noun', description="The object noun in question, dictionary formatted with MINIMUM possible changes (no substantive changes) to have correct spacing (e.g. for compound nouns), correct punctuation (e.g. hyphenation, apostrophe, other), and correct dictionary capitalization (i.e. in case of proper nouns, otherwise lowercase).")
	dictionary_gloss: str = pydantic.Field(title='Dictionary gloss', description="A dictionary gloss for the object noun (covering its meaning in terms of it being an object or physical entity that can potentially be seen in a camera image).")
	is_valid_object_noun: bool = pydantic.Field(title='Is valid object noun', description="Whether the object noun is valid, i.e. an object, thing, or physical entity that may occur in a camera image.")
	can_conceivably_occur_discussion: str = pydantic.Field(title='Can conceivably occur (discussion)', description="In the specific described scenario, discuss in two sentences how conceivable it is that an instance of the object noun in question could ever be seen in a captured camera image.")
	can_conceivably_occur_rating: int = pydantic.Field(title='Can conceivably occur (rating 1-10)', description="In the specific described scenario, rate from 1 to 10 how conceivable it is that an instance of the object noun in question could ever be seen in a captured camera image.")
	will_likely_occur_discussion: str = pydantic.Field(title='Will likely occur (discussion)', description="In the specific described scenario, discuss in two sentences how likely it is in practice that an instance of the object noun in question will at some point be seen in a captured camera image.")
	will_likely_occur_rating: int = pydantic.Field(title='Will likely occur (rating 1-10)', description="In the specific described scenario, rate from 1 to 10 how likely it is in practice that an instance of the object noun in question will at some point be seen in a captured camera image.")

# Object nouns data class (task output entry)
@dataclasses.dataclass
class ObjectNounsData:
	target_noun: str
	pretty_noun: str
	opinions: list[ObjectNounInfo]

# Object nouns file class (task output file)
class ObjectNounsFile(task_manager.DataclassListOutputFile):
	Dataclass = ObjectNounsData

# Customize object nouns task class (task manager)
class CustomizeObjectNounsTask(task_manager.TaskManager):

	# Sample key: 'TARGET' where TARGET is the target noun
	# Task state:
	#  - committed_samples: dict[str, int] => Maps a sample key to the number of times it has been committed
	#  - failed_samples:    dict[str, int] => Maps a sample key to the number of times it has failed
	#  - succeeded_samples: dict[str, Union[list[~ObjectNounInfo], int]] => Maps a sample key to the list of succeeded ObjectNounInfo opinions, or just how many there were if an output entry has already been generated from them
	# Task output: Single JSONL file of JSON-serialized ObjectNounsData dataclass instances

	def __init__(self, cfg: gba_utils.Config):
		super().__init__(
			task_dir=cfg.working_dir,
			name_prefix=f'object_nouns_{cfg.scenario_safe}',
			output_factory=ObjectNounsFile.output_factory(),
			init_meta=dict(
				model=gba_utils.resolve(cfg.model, default='gpt-4o-mini-2024-07-18'),
				max_completion_tokens=gba_utils.resolve(cfg.max_completion_tokens, default=200),  # TODO: UPDATE
				completion_ratio=gba_utils.resolve(cfg.completion_ratio, default=0.35),  # TODO: UPDATE
				temperature=gba_utils.resolve(cfg.temperature, default=0.8),
				top_p=gba_utils.resolve(cfg.top_p, default=0.8),
				opinions=gba_utils.resolve(cfg.opinions, 3),  # Note: If cfg.opinions is changed then wipe_failed must also be set on the first run where it changed (otherwise expect an exception to be raised)!
			),
			**gba_utils.get_init_kwargs(cls=task_manager.TaskManager, cfg=cfg),
			**gba_utils.get_init_kwargs(cls=gpt_requester.GPTRequester, cfg=cfg, endpoint='/v1/chat/completions', assumed_completion_ratio=None)
		)
		self.cfg = cfg

	def on_task_enter(self):
		self.GR.set_assumed_completion_ratio(self.T.meta['completion_ratio'])

	def wipe_unfinished(self, wipe_failed: bool, rstack: gba_utils.RevertStack) -> bool:

		self.T.committed_samples.clear()
		for sample_key, opinions in self.T.succeeded_samples.items():
			self.T.committed_samples[sample_key] = opinions if isinstance(opinions, int) else len(opinions)

		if wipe_failed:
			self.T.failed_samples.clear()
			err_type, err_count = [gba_utils.LogSummarizer(log_fn=log.error, show_msgs=self.GR.show_errors) for _ in range(2)]
			for entry in self.output.rewrite(rstack=rstack):
				opinions: Union[list[dict[str, Any]], int] = self.T.succeeded_samples.get(entry.target_noun, [])
				if not isinstance(opinions, int):
					err_type.log(f"Outputted sample unexpectedly has non-int opinions stored in succeeded_samples: {entry.target_noun}")
					self.output.data.entries.append(entry)
				elif opinions != len(entry.opinions):
					err_count.log(f"Unexpected mismatch in the number of opinions stored in the output file vs succeeded_samples: {len(entry.opinions)} vs {opinions}")
					self.output.data.entries.append(entry)
				elif opinions >= self.cfg.opinions:
					self.output.data.entries.append(entry)
				else:
					self.T.succeeded_samples[entry.target_noun] = entry.opinions.copy()
			self.D = self.output.data
			err_type.finalize(msg_fn=lambda num_omitted, num_total: f"Got {num_omitted} further outputted samples with non-int opinions stored in succeeded_samples (total {num_total} errors)")
			err_count.finalize(msg_fn=lambda num_omitted, num_total: f"Got {num_omitted} further unexpected mismatches in the number of opinions stored in the output file vs succeeded_samples (total {num_total} errors)")
		else:
			for sample_key, num_failed in self.T.failed_samples.items():
				self.T.committed_samples[sample_key] = self.T.committed_samples.get(sample_key, 0) + num_failed

		return False

	def validate_state(self, *, clean: bool):
		super().validate_state(clean=clean)
		if clean:
			if unclean_sample_keys := {sample_key for sample_key, num_committed in self.T.committed_samples.items() if (opinions if isinstance((opinions := self.T.succeeded_samples.get(sample_key, [])), int) else len(opinions)) + self.T.failed_samples.get(sample_key, 0) != num_committed}:
				raise ValueError(f"Unexpected unclean sample keys: {sorted(unclean_sample_keys)}")

	def generate_requests(self) -> bool:

		input_nouns_path = os.path.abspath(self.cfg.input_nouns)
		with open(input_nouns_path, 'r') as file:
			object_noun_entries = json.load(file)

		seen_target_nouns = set()
		for entry in object_noun_entries:

			target_noun = entry['target_noun']
			if target_noun in seen_target_nouns:
				raise ValueError(f"There are multiple instances of the target noun '{target_noun}' in: {input_nouns_path}")

			pretty_noun = entry['pretty_noun']
			pretty_noun_canon = get_canon(noun=pretty_noun, sanitize=False)
			if pretty_noun_canon != target_noun:
				raise ValueError(f"Canonical form of the pretty noun is not equal to the target noun: {pretty_noun} ~ {pretty_noun_canon} != {target_noun}")

			if sum(entry['singulars_freq']) + sum(entry['plurals_freq']) <= self.cfg.input_ft:
				continue

			num_required = self.cfg.opinions - self.T.committed_samples.get(target_noun, 0)
			if num_required > 0:
				if isinstance(self.T.succeeded_samples.get(target_noun, None), int):
					raise ValueError(f"Cannot generate requests for a target noun that has int opinions stored in succeeded_samples: {target_noun}")
				request = gpt_requester.GPTRequest(
					payload=dict(
						model=self.T.meta['model'],
						max_completion_tokens=self.T.meta['max_completion_tokens'],
						temperature=self.T.meta['temperature'],
						top_p=self.T.meta['top_p'],
						messages=[
							dict(role='system', content=(
								"Given an English object noun by the user, it is your sole and only task to provide information about that object noun, in a structured JSON format that follows a predefined schema. "
								"'Object nouns' are simple or compound English nouns that correspond to an object, thing, or physical entity that may occur in a camera image. "
								"The information you provide about each object noun will be used to compile a scenario-specific dictionary of nouns that includes metadata like how conceivable or likely such an object would be to occur in the specific stated scenario. "
								"The exact scenario to consider for all information you provide about object nouns is described below.\n\n"
								f"SCENARIO: {self.cfg.description}"
							)),
							dict(role='user', content=f"OBJECT NOUN:\n{pretty_noun}"),
						],
						response_format=ObjectNounInfo,
					),
					meta=dict(
						entry=entry,
					),
				)
				self.GR.add_requests(request for _ in range(num_required))

			if self.GR.PQ.pool_len >= 6000 and self.commit_requests():
				return False

		return True

	def commit_cached_request(self, cached_req: gpt_requester.CachedGPTRequest):
		sample_key = cached_req.item.req.meta['entry']['target_noun']
		self.T.committed_samples[sample_key] = self.T.committed_samples.get(sample_key, 0) + 1

	def cached_request_keys(self, cached_reqs: list[gpt_requester.CachedGPTRequest]) -> Optional[set[str]]:
		return {cached_req.item.req.meta['entry']['target_noun'] for cached_req in cached_reqs}

	def process_batch_result(self, result: gpt_requester.BatchResult, rstack: gba_utils.RevertStack) -> bool:
		raise NotImplementedError

	# TODO: generate_requests should raise an exception if we have an int that needs another request committed to it
	# TODO: process_batch_result should raise an exception if we have a result for something that has an int opinions
	# TODO: process_batch_result needs to GUARANTEE that ANY thing written to output file is guaranteed to have int in succeeded_samples, even if pure fails (then int of 0)
	# TODO: CONTINUE (# Scenario / # Task)
	# TODO: Verify that canonical form of new pretty form is unchanged, otherwise warn and use old pretty form?
	# TODO: TEST changing opinions higher/lower in combination with setting wipe_failed or not
	# TODO: TEST unclean_sample_keys in validate_state()

#
# Miscellaneous
#

# Convert a noun to its canonical form (as per object noun dictionary)
def get_canon(noun: str, sanitize: bool) -> str:
	if sanitize:
		noun = unidecode.unidecode(' '.join(noun.split()))
	canon = noun.lower()
	canon = canon.replace("'", "").replace('.', '')
	canon = ' '.join(part for part in re.split(r'[\s/-]+', canon) if part)
	return canon

#
# Run
#

# Main function
def main():

	parser = argparse.ArgumentParser(description="Customize object nouns for a particular scenario.", add_help=False, formatter_class=functools.partial(argparse.HelpFormatter, max_help_position=36))
	parser.add_argument('--help', '-h', action='help', default=argparse.SUPPRESS, help="Show this help message and exit")
	parser.add_argument('--scenario', type=str, required=True, metavar='NAME', help="Scenario name to customize the object nouns for")
	parser.add_argument('--description', type=str, required=True, metavar='DESC', help="Scenario description")
	parser.add_argument('--input_nouns', type=str, metavar='PATH', help="Input object nouns JSON file (default: 'data/object_nouns.json' relative to this script)")
	parser.add_argument('--input_ft', type=int, default=0, metavar='FT', help="Frequency threshold (integer) non-strictly below which to ignore input nouns (default: %(default)s)")
	parser.add_argument('--output_nouns', type=str, metavar='PATH', help="Output object nouns JSON file (default: 'data/object_nouns_{scenario}.json' relative to this script)")
	parser.add_argument('--working_dir', type=str, metavar='PATH', help="GPT Batch API task directory (default: {output_nouns dir}/gba_tasks)")

	parser_meta = parser.add_argument_group(title='Task metadata')
	parser_meta.add_argument('--model', type=str, help="LLM model to use")
	parser_meta.add_argument('--max_completion_tokens', type=int, metavar='NUM', help="Maximum number of generated output tokens per request (including both reasoning and visible tokens)")
	parser_meta.add_argument('--completion_ratio', type=float, metavar='RATIO', help="How many output tokens (including both reasoning and visible tokens) to assume will be generated for each request on average, as a ratio of max_completion_tokens")
	parser_meta.add_argument('--temperature', type=float, metavar='TEMP', help="What sampling temperature to use")
	parser_meta.add_argument('--top_p', type=float, metavar='MASS', help="Nucleus sampling probability mass")
	parser_meta.add_argument('--opinions', type=int, metavar='NUM', help="Number of opinions required per object noun")

	task_manager.TaskManager.configure_argparse(parser=parser)
	gpt_requester.GPTRequester.configure_argparse(parser=parser)

	cfg = parser.parse_args()

	if not cfg.scenario:
		raise ValueError("Please provide a valid scenario")
	if not cfg.description:
		raise ValueError("Please provide a valid description")
	cfg.scenario_safe = cfg.scenario.replace('/', '_')

	script_dir = os.path.dirname(os.path.abspath(__file__))
	if cfg.input_nouns is None:
		cfg.input_nouns = os.path.join(script_dir, 'data', 'object_nouns.json')
	if cfg.output_nouns is None:
		cfg.output_nouns = os.path.join(script_dir, 'data', f'object_nouns_{cfg.scenario_safe}.json')
	if cfg.working_dir is None:
		cfg.working_dir = os.path.join(os.path.dirname(cfg.output_nouns), 'gba_tasks')
	if cfg.wandb is None:
		cfg.wandb = not cfg.dryrun

	with gpt_requester.GPTRequester.wandb_init(config=cfg):

		log.info(f"Scenario: {cfg.scenario}")
		log.info(f"Scenario description:\n{cfg.description}")
		log.info(f"Input nouns (FT{cfg.input_ft}): {cfg.input_nouns}")
		log.info(f"Task working directory: {cfg.working_dir}")
		log.info(f"Output nouns will be: {cfg.output_nouns}")

		task = CustomizeObjectNounsTask(cfg=cfg)
		if task.run():
			generate_output_nouns(cfg=cfg)

# Generate the output nouns file based on the task output
def generate_output_nouns(cfg: gba_utils.Config):
	pass  # TODO: cfg.output_nouns / If task TOTALLY completes then output_nouns JSON is generated based on task output JSONL (open it in read-only mode using output file class)

# Run main function
if __name__ == "__main__":
	log = gba_utils.configure_logging()
	main()
# EOF
