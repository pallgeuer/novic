# Customize object nouns for a particular scenario

# Imports
import os
import re
import json
import argparse
import functools
import unidecode
import dataclasses
from typing import Optional
import pydantic
import openai.types.chat as openai_chat
from gpt_batch_api import task_manager, gpt_requester, utils as gba_utils

#
# Customize object nouns task
#

# Object noun information class (structured outputs schema)
class ObjectNounInfo(pydantic.BaseModel):
	model_config = pydantic.ConfigDict(strict=True)
	object_noun: str = pydantic.Field(title='Object noun', description="The object noun in question, written exactly as given by the user.")
	formatted_noun: str = pydantic.Field(title='Formatted noun', description="The object noun in question, dictionary formatted with MINIMUM possible changes (no substantive changes) to have correct spacing (e.g. for compound nouns), correct punctuation (e.g. hyphenation, apostrophe, other), and correct dictionary capitalization (i.e. in case of proper nouns, otherwise lowercase).")
	dictionary_gloss: str = pydantic.Field(title='Dictionary gloss', description="A dictionary gloss for the object noun (covering its meaning in terms of it being an object or physical entity that can potentially be seen in a camera image).")
	is_valid_object_noun: bool = pydantic.Field(title='Is valid object noun', description="Whether the formatted object noun is valid, i.e. an object, thing, or physical entity that may occur in a camera image.")
	can_conceivably_occur_discussion: str = pydantic.Field(title='Can conceivably occur (discussion)', description="In the specific described scenario, discuss in two sentences how conceivable it is that an instance of the object noun in question could ever be seen in a captured camera image.")
	can_conceivably_occur_rating: int = pydantic.Field(title='Can conceivably occur (rating 1-10)', description="In the specific described scenario, rate from 1 to 10 how conceivable it is that an instance of the object noun in question could ever be seen in a captured camera image.")
	will_likely_occur_discussion: str = pydantic.Field(title='Will likely occur (discussion)', description="In the specific described scenario, discuss in two sentences how likely and regular it is in practice that an instance of the object noun in question will at some point be seen in a captured camera image.")
	will_likely_occur_rating: int = pydantic.Field(title='Will likely occur (rating 1-10)', description="In the specific described scenario, rate from 1 to 10 how likely and regular it is in practice that an instance of the object noun in question will at some point be seen in a captured camera image.")

# Object noun data class (task output entry)
@dataclasses.dataclass
class ObjectNounData:
	target_noun: str
	pretty_noun: str
	opinion_num: int
	target_noun_opinion: str
	pretty_noun_opinion: str
	opinion: ObjectNounInfo

# Object nouns file class (task output file)
class ObjectNounsFile(task_manager.DataclassListOutputFile):
	Dataclass = ObjectNounData

# Customize object nouns task class (task manager)
class CustomizeObjectNounsTask(task_manager.TaskManager):

	# Sample key: The target noun
	# Task state:
	#  - committed_samples: dict[str, int] => Maps a sample key to the number of times it has been committed
	#  - failed_samples:    dict[str, int] => Maps a sample key to the number of times it has failed
	#  - succeeded_samples: dict[str, int] => Maps a sample key to the number of times it has succeeded
	# Task output: Single JSONL file of JSON-serialized ObjectNounData dataclass instances

	def __init__(self, cfg: gba_utils.Config):
		super().__init__(
			task_dir=cfg.working_dir,
			name_prefix=f'object_nouns_{cfg.scenario_safe}',
			output_factory=ObjectNounsFile.output_factory(),
			init_meta=dict(
				model=gba_utils.resolve(cfg.model, default='gpt-4o-mini-2024-07-18'),
				max_completion_tokens=gba_utils.resolve(cfg.max_completion_tokens, default=2048),  # TODO: UPDATE
				completion_ratio=gba_utils.resolve(cfg.completion_ratio, default=0.35),  # TODO: UPDATE
				temperature=gba_utils.resolve(cfg.temperature, default=0.8),
				top_p=gba_utils.resolve(cfg.top_p, default=0.8),
				opinions=gba_utils.resolve(cfg.opinions, 3),
			),
			**gba_utils.get_init_kwargs(cls=task_manager.TaskManager, cfg=cfg),
			**gba_utils.get_init_kwargs(cls=gpt_requester.GPTRequester, cfg=cfg, endpoint='/v1/chat/completions', assumed_completion_ratio=None)
		)
		self.cfg = cfg

	def on_task_enter(self):
		self.GR.set_assumed_completion_ratio(self.T.meta['completion_ratio'])

	def wipe_unfinished(self, wipe_failed: bool, rstack: gba_utils.RevertStack) -> bool:
		self.T.committed_samples.clear()
		for sample_key, num_succeeded in self.T.succeeded_samples.items():
			self.T.committed_samples[sample_key] = num_succeeded
		if wipe_failed:
			self.T.failed_samples.clear()
		else:
			for sample_key, num_failed in self.T.failed_samples.items():
				self.T.committed_samples[sample_key] = self.T.committed_samples.get(sample_key, 0) + num_failed
		return False

	def validate_state(self, *, clean: bool):
		super().validate_state(clean=clean)
		if clean:
			if unclean_sample_keys := {sample_key for sample_key, num_committed in self.T.committed_samples.items() if self.T.succeeded_samples.get(sample_key, 0) + self.T.failed_samples.get(sample_key, 0) != num_committed}:
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
								"The exact scenario to consider for all information you provide about the object nouns is described below. "
								"You must consider for this scenario what kinds of objects could conceivably be seen in the captured camera images, as well as how likely and regularly they are expected to be seen in practice.\n\n"
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

		failed_samples = {}
		succeeded_samples = {}
		num_entries = len(self.D.entries)

		@rstack.callback
		def revert_sample_state():
			for skey, value in failed_samples.items():
				if value is None:
					del self.T.failed_samples[skey]
				else:
					self.T.failed_samples[skey] = value
			for skey, value in succeeded_samples.items():
				if value is None:
					del self.T.succeeded_samples[skey]
				else:
					self.T.succeeded_samples[skey] = value
			del self.D.entries[num_entries:]

		err_noun_mismatch, err_bad_rating, err_misc_failed = [gba_utils.LogSummarizer(log_fn=log.error, show_msgs=self.GR.show_errors) for _ in range(3)]
		for info in result.info.values():

			entry = info.req_info.meta['entry']
			target_noun = entry['target_noun']
			pretty_noun = entry['pretty_noun']

			sample_key = target_noun
			num_committed = self.T.committed_samples.get(sample_key, 0)
			num_failed = self.T.failed_samples.get(sample_key, 0)
			num_succeeded = self.T.succeeded_samples.get(sample_key, 0)
			assert num_committed > num_failed + num_succeeded >= 0

			succeeded = False
			if info.err_info is None and info.resp_info is not None and isinstance(info.resp_info.payload, openai_chat.ParsedChatCompletion) and info.resp_info.payload.choices and isinstance((object_noun_info := info.resp_info.payload.choices[0].message.parsed), ObjectNounInfo):  # noqa
				assert not info.retry and info.retry_counts
				object_noun_info: ObjectNounInfo
				if object_noun_info.object_noun != pretty_noun:
					info.err_info = gpt_requester.ErrorInfo(fatal=False, type='TaskSpecific', subtype='ObjectNounMismatch', data=object_noun_info.object_noun, msg=f"Got object noun '{object_noun_info.object_noun}' instead of '{pretty_noun}'")
					err_noun_mismatch.log(f"Batch {result.batch.id} request ID {info.req_id} retry {info.req_info.retry_num} had a noun mismatch: {info.err_info.msg}")
				elif not (1 <= object_noun_info.can_conceivably_occur_rating <= 10 and 1 <= object_noun_info.will_likely_occur_rating <= 10):
					info.err_info = gpt_requester.ErrorInfo(fatal=False, type='TaskSpecific', subtype='BadRating', data=(object_noun_info.can_conceivably_occur_rating, object_noun_info.will_likely_occur_rating), msg=f"Got rating(s) out of 1-10 range: {object_noun_info.can_conceivably_occur_rating}, {object_noun_info.will_likely_occur_rating}")
					err_bad_rating.log(f"Batch {result.batch.id} request ID {info.req_id} retry {info.req_info.retry_num} had bad rating(s): {info.err_info.msg}")
				else:
					if sample_key not in succeeded_samples:
						succeeded_samples[sample_key] = self.T.succeeded_samples.get(sample_key, None)
					num_succeeded += 1
					self.T.succeeded_samples[sample_key] = num_succeeded
					self.D.entries.append(ObjectNounData(
						target_noun=target_noun,
						pretty_noun=pretty_noun,
						opinion_num=num_succeeded,
						target_noun_opinion=get_canon(noun=object_noun_info.formatted_noun, sanitize=True),
						pretty_noun_opinion=object_noun_info.formatted_noun,
						opinion=object_noun_info,
					))
					succeeded = True
				if not succeeded:
					self.GR.update_result_retry(info=info)

			if not succeeded and not info.retry:
				if info.err_info is None and not self.GR.dryrun:
					err_misc_failed.log(f"Batch {result.batch.id} request ID {info.req_id} retry {info.req_info.retry_num} got no error yet FAILED")
				if sample_key not in failed_samples:
					failed_samples[sample_key] = self.T.failed_samples.get(sample_key, None)
				self.T.failed_samples[sample_key] = num_failed + 1

			err_noun_mismatch.finalize(msg_fn=lambda num_omitted, num_total: f"Encountered {num_omitted} further noun mismatches (total {num_total} occurrences)")
			err_bad_rating.finalize(msg_fn=lambda num_omitted, num_total: f"Encountered {num_omitted} further bad ratings (total {num_total} occurrences)")
			err_misc_failed.finalize(msg_fn=lambda num_omitted, num_total: f"Encountered {num_omitted} further requests that got no error yet FAILED (total {num_total} occurrences)")

			return bool(succeeded_samples) or bool(failed_samples)

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

	log.info(f"Scenario: {cfg.scenario}")
	log.info(f"Scenario description:\n{cfg.description}")
	log.info(f"Input nouns (FT{cfg.input_ft}): {cfg.input_nouns}")
	log.info(f"Task working directory: {cfg.working_dir}")
	log.info(f"Output nouns will be: {cfg.output_nouns}")

	with gpt_requester.GPTRequester.wandb_init(config=cfg):
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
