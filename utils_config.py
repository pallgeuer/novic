# Configuration utilities

# Imports
from typing import Any
import omegaconf
import wandb
import utils

# Flatten the configuration to a single-level dictionary
def flatten_config(cfg: omegaconf.DictConfig) -> dict[str, Any]:
	return utils.flatten_dict(omegaconf.OmegaConf.to_container(cfg, resolve=True))

# Convert an OmegaConf configuration to a wandb configuration
def wandb_from_omegaconf(cfg, **cfg_kwargs):
	cfg_dict = omegaconf.OmegaConf.to_container(cfg)
	cfg_dict = utils.flatten_dict(cfg_dict)
	cfg_dict = {k: (format(v) if isinstance(v, list) else v) for k, v in cfg_dict.items() if not (k == 'wandb' or k.startswith('wandb_'))}
	cfg_dict.update(cfg_kwargs)
	return cfg_dict

# Print wandb configuration
def print_wandb_config(C=None, newline=True):
	if C is None:
		C = wandb.config
	print("Wandb configuration:")
	# noinspection PyProtectedMember
	for key, value in C._items.items():
		if key == '_wandb':
			if value:
				print("  wandb:")
				for wkey, wvalue in value.items():
					print(f"    {wkey}: {wvalue}")
			else:
				print("  wandb: -")
		else:
			print(f"  {key}: {value}")
	if newline:
		print()
# EOF
