# Customize object nouns for a particular scenario

# Imports
import os
import argparse
import functools
from gpt_batch_api import task_manager, gpt_requester, utils as gba_utils

#
# Customize object nouns task
#

# Customize object nouns task class
class CustomizeObjectNounsTask(task_manager.TaskManager):

	def __init__(self, cfg: gba_utils.Config):
		super().__init__(
			task_dir=cfg.working_dir,
			name_prefix=f'object_nouns_{cfg.scenario_safe}',
			output_factory=ObjectNounsMetaFile.output_factory(),
			init_meta=dict(
				model=gba_utils.resolve(cfg.model, default='gpt-4o-mini-2024-07-18'),
				max_completion_tokens=gba_utils.resolve(cfg.max_completion_tokens, default=200),  # TODO: UPDATE
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
	parser.add_argument('--input_ft', type=int, default=0, metavar='FT', help="Frequency threshold (integer) strictly below which to ignore input nouns (default: %(default)s)")
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
		task = CustomizeObjectNounsTask(cfg=cfg)
		if task.run():
			pass  # TODO: If task TOTALLY completes then output_nouns JSON is generated based on output file

# Run main function
if __name__ == "__main__":
	log = gba_utils.configure_logging()
	main()
# EOF
