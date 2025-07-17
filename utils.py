# General utilities

# Imports
import re
import gc
import sys
import math
import json
import types
import random
import pickle
import signal
import inspect
import functools
import collections
import dataclasses
import typing
from typing import Union, Callable, Optional, Iterable, Type, IO, Any
import unidecode
import tqdm
import PIL.Image
import numpy as np
import torch
import torch.utils.data
import torch.backends.cudnn
from logger import log

# Constants
IMAGE_PATTERNS = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.webp', '*.tiff')
ALLOWED_CHARS_CANON = set(" 0123456789abcdefghijklmnopqrstuvwxyz")

#
# Modules
#

# Embedding class that can tie its weights transparently to an external linear layer (no state dict and parameter double-counting issues, and will work even if the linear weight is reassigned with a new tensor/parameter)
class Embedding(torch.nn.Embedding):

	def __init__(self, *args, linear=None, **kwargs):
		super().__init__(*args, **kwargs)
		if linear is not None:
			if kwargs.get('_weight', None) is not None:
				raise ValueError("Cannot specify both _weight and linear")
			delattr(self, 'weight')
			super(torch.nn.Module, self).__setattr__('_linear', linear)

	def __getattr__(self, name: str) -> Any:
		if name == 'weight' and '_linear' in self.__dict__:
			return self.__dict__['_linear'].weight
		else:
			return super().__getattr__(name)

	def __setattr__(self, name: str, value: Union[torch.Tensor, torch.nn.Module]):
		if name == 'weight' and '_linear' in self.__dict__:
			raise AttributeError("Cannot set weight attribute if it is tied to a linear module")
		super().__setattr__(name, value)

	def __delattr__(self, name):
		if name == 'weight' and '_linear' in self.__dict__:
			super(torch.nn.Module, self).__delattr__('_linear')
		else:
			super().__delattr__(name)

# Linear class that can also perform embedding with the same weights
class LinearEmbed(torch.nn.Linear):

	def embed(self, x: torch.Tensor) -> torch.Tensor:
		return torch.nn.functional.embedding(input=x, weight=self.weight)

#
# Module utilities
#

# Named version of recursive apply for a module
def module_named_apply(module: torch.nn.Module, fn: Callable[[str, torch.nn.Module], None], *, prefix=None) -> torch.nn.Module:
	if prefix is None:
		prefix = ''
	for name, mod in module.named_children():
		module_named_apply(mod, fn, prefix=f'{prefix}.{name}')
	fn(prefix, module)
	return module

# Create an activation function and get its approximate gain
def get_activation_gain(name: str, functional: bool, unit_std: bool) -> tuple[Callable[[torch.Tensor], torch.Tensor], float]:
	# name = String name of the desired activation function
	# functional = Whether to return the activation function as a functional callable instead of a module callable
	# unit_std = Whether to return a gain assuming unit normal standard deviation input (True), or infinitesimally small inputs (False)
	# Returns the activation callable and corresponding gain
	#
	# For infinitesimally small inputs, the gain is the absolute value of the slope at zero, or more generally, the quadratic mean of the left- and right-slopes
	# For unit normal inputs, the gain is calculated by transforming the probability distribution function (PDF) via the activation function (https://en.wikipedia.org/wiki/Probability_density_function#Scalar_to_scalar) and calculating the standard deviation around zero (!) of the resulting PDF (https://en.wikipedia.org/wiki/Standard_deviation#Continuous_random_variable)
	#
	# Empirically the values can be determined/checked using:
	#   import torch
	#   std0 = lambda Y: Y.square().sum().div(Y.numel()).sqrt()
	#   X = torch.randn((100000000,), dtype=torch.float64)
	#   act = torch.nn.functional.gelu
	#   print(std0(act(X)) / std0(X), std0(act(X / 1000)) / std0(X / 1000))

	if name == 'tanh':
		activation = torch.nn.functional.tanh if functional else torch.nn.Tanh()
		activation_gain = 0.6279 if unit_std else 1.0  # Note: https://www.wolframalpha.com/input?i=sqrt+of+integral+of+x%5E2*exp%28-atanh%28x%29%5E2%2F2%29%2F%28sqrt%282*pi%29*%281-x%5E2%29%29+from+-1+to+1
	elif name == 'relu':
		activation = torch.nn.functional.relu if functional else torch.nn.ReLU()
		activation_gain = 1 / math.sqrt(2)  # Note: Quadratic mean of the left-slope 0 and right-slope 1
	elif name == 'gelu':
		activation = torch.nn.functional.gelu if functional else torch.nn.GELU()
		activation_gain = 0.6521 if unit_std else 0.5  # Note: Computed empirically using code above
	else:
		raise ValueError(f"Unsupported hidden activation function: {name}")

	return activation, activation_gain

#
# Training utilities
#

# Set whether TF32 should be allowed
def allow_tf32(enable: bool):
	allow = bool(enable)
	torch.backends.cuda.matmul.allow_tf32 = allow
	torch.backends.cudnn.allow_tf32 = allow
	log.info(f"TF32 tensor cores are {'enabled' if allow else 'disabled'}")

# Set whether execution should be deterministic (as much as possible)
def set_determinism(deterministic, seed=1, cudnn_benchmark_mode=False):
	# See: https://pytorch.org/docs/stable/notes/randomness.html
	# Env vars: PYTHONHASHSEED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
	# Also need: Zero dataset workers, no AMP
	# If still slightly non-deterministic, try CPU as fallback
	if deterministic:
		log.info("Deterministic mode with cuDNN benchmark mode disabled")
		seed = max(int(seed), 1)
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		torch.use_deterministic_algorithms(True, warn_only=True)
	else:
		log.info(f"Fast non-deterministic mode with cuDNN benchmark mode {'enabled' if cudnn_benchmark_mode else 'disabled'}")
		random.seed()
		np.random.seed()  # noqa
		torch.seed()
		torch.backends.cudnn.deterministic = False
		torch.backends.cudnn.benchmark = cudnn_benchmark_mode
		torch.use_deterministic_algorithms(False, warn_only=True)

# Release non-required CUDA memory
def release_cuda_memory(device: Union[int, str, torch.device] = 'cuda'):
	if not isinstance(device, torch.device):
		device = torch.device(device)
	if torch.cuda.is_initialized() and device.type == 'cuda':
		torch.cuda.synchronize(device)
	gc.collect()
	if torch.cuda.is_initialized():
		torch.cuda.ipc_collect()
		torch.cuda.empty_cache()

# Collate function for the None type
# noinspection PyUnusedLocal
def collate_none_fn(batch, *, collate_fn_map: Optional[dict[Union[type, tuple[type, ...]], Callable]] = None):
	if all(elem is None for elem in batch):
		return None
	else:
		raise TypeError("Cannot None-collate a batch that is not all None")

# Default collate function map that includes None support
default_collate_fn_map = torch.utils.data._utils.collate.default_collate_fn_map.copy()  # noqa
default_collate_fn_map[type(None)] = collate_none_fn

# Default collate function that includes None support
def default_collate(batch):
	return torch.utils.data._utils.collate.collate(batch, collate_fn_map=default_collate_fn_map)  # noqa

# Rescale all dropout in a model (verify there are no functional uses of dropout in the model, even indirectly via inbuilt PyTorch modules)
def rescale_dropout(model: torch.nn.Module, factor: float):

	assert factor >= 0
	if factor == 0:
		log.info("Setting all model dropout probabilities to zero")
	else:
		log.info(f"Rescaling all model dropout probabilities by \u00D7{factor}")

	def rescale(module: torch.nn.Module):
		# noinspection PyProtectedMember
		if isinstance(module, torch.nn.modules.dropout._DropoutNd):
			module.p *= factor
		elif isinstance(module, torch.nn.MultiheadAttention):
			module.dropout *= factor

	model.apply(rescale)

# Capture and save a gradient in a global map (intended for interactive debugging only)
captured_tensors = {}
def capture_gradient(tensor: torch.Tensor, name: str):
	tensor.retain_grad()
	captured_tensors[name] = tensor

# Capture the gradient of an arbitrary intermediate tensor in the forward pass and check for NaNs both now and in the backward pass
def check_tensor_nans(tensor: torch.Tensor, name: str, always: bool = False):
	# Usage: Just add something like the following anywhere in the forward pass: utils.check_tensor_nans(x, 'Logits')
	def grad_hook(grad):  # Note: Debug breakpoints don't necessarily work inside this method (the code executes but the breakpoint doesn't trigger)
		_print_nan_info(tensor=grad.detach(), name=f"{name} grad", prefix='BWDPASS', always=always)
	_print_nan_info(tensor=tensor.detach(), name=f"{name} value", prefix='FWDPASS', always=always)
	if tensor.requires_grad:
		tensor.register_hook(grad_hook)

# Register forward and backward hooks to detect NaNs in the inputs/outputs of a module
def check_module_nans(module: torch.nn.Module):
	# Usage: model.apply(check_module_nans)
	# Note: This can only check inputs/outputs of modules, and thus may not for instance capture NaNs that occur during loss calculation outside of a module
	module.register_forward_hook(functools.partial(_check_nans_hook, typ='value', prefix='FWDPASS'))  # noqa
	module.register_full_backward_hook(functools.partial(_check_nans_hook, typ='grad', prefix='BWDPASS'))  # noqa

# Generic hook for module NaN checking
def _check_nans_hook(mod, inputs, outputs, typ, prefix):
	for name, srcs in (('Inp', inputs), ('Out', outputs)):
		if not isinstance(srcs, (tuple, list)):
			srcs = (srcs,)
		for i, src in enumerate(srcs):
			if isinstance(src, torch.Tensor):
				modstr = format(mod).partition('\n')[0].rstrip('(')
				_print_nan_info(tensor=src.detach(), name=f"{name} {typ} {i} of module {modstr}", prefix=prefix)

# Helper method for displaying NaN-checking
def _print_nan_info(tensor: torch.Tensor, name: str, prefix: str, always: bool = False):
	nan_mask = torch.isnan(tensor)
	if (nan_count := nan_mask.sum().item()) or always:
		shape = '\u00D7'.join(str(dim) for dim in nan_mask.shape) or 'scalar'
		nandims = tuple(torch.amax(nan_mask, dim=tuple(d for d in range(nan_mask.ndim) if d != dim)).sum().item() for dim in range(nan_mask.ndim))
		nanshape = '\u00D7'.join(str(nandim) for nandim in nandims) or 'scalar'
		print(f"[{prefix}] {name}: Shape {shape} has {nan_count} NaNs across {nanshape} dims = {math.prod(nandims) if nandims else nan_count} elements")

#
# Dataset utilities
#

# Dataset class for unannotated collections of images
class ImageDataset(torch.utils.data.Dataset):

	def __init__(self, image_paths: Iterable[str], transform: Callable[[PIL.Image.Image], torch.Tensor] = None):
		self.image_paths = image_paths if isinstance(image_paths, tuple) else tuple(image_paths)
		self.transform = transform

	def __len__(self) -> int:
		return len(self.image_paths)

	def __getitem__(self, index: int) -> tuple[str, Union[PIL.Image.Image, torch.Tensor]]:
		image_path = self.image_paths[index]
		image = PIL.Image.open(image_path)
		if image.mode != 'RGB':
			image = image.convert('RGB')
		if self.transform:
			image = self.transform(image)
		return image_path, image

	def create_loader(self, batch_size: int, num_workers: int, device_is_cpu: bool) -> torch.utils.data.DataLoader:
		dataset_workers = 0 if debugger_attached() else min(batch_size, num_workers)
		loader = torch.utils.data.DataLoader(self, batch_size=batch_size, num_workers=dataset_workers, shuffle=True, pin_memory=not device_is_cpu, drop_last=False)
		log.info(f"Created shuffled image data loader for {len(self)} images with {len(loader)} batches of nominal size {loader.batch_size} and {loader.num_workers} workers and no sample dropping")
		return loader

#
# Debug utilities
#

# Determine whether there is a debugger attached (more precisely, any debugger, profiler, code coverage tool, etc that implements sys.gettrace())
# Examples of False: Run in PyCharm, run script from console, run normal code in Jupyter notebook
# Examples of True: Debug in PyCharm, interactive Python console, explicitly debug code in Jupyter notebook
def debugger_attached():
	return sys.gettrace() is not None

# Show the statistical contents of a tensor
# Note: While debugging anywhere just do 'from utils import show' then for example 'show(x)'
def show(*tensors: torch.Tensor, dim: Optional[int] = None, prefix: Optional[str] = None):
	if len(tensors) == 1:
		tensor = tensors[0]
		if dim is None:
			tensor = tensor.detach()
			shape = '\u00D7'.join(str(dim) for dim in tensor.shape)
			norms = torch.linalg.norm(tensor, dim=tensor.ndim - 1)
			prefix = '' if prefix is None else f'{prefix}: \t'
			print(f"{prefix}{shape} with elem {tensor.mean().item():.4g} + {tensor.std(correction=0).item():.4g}z (= 0 + {tensor.square().sum().div(tensor.numel()).sqrt().item():.4g}z) and norms {norms.mean().item():.4g} + {norms.std(correction=0).item():.4g}z")
		else:
			for i in range(tensor.shape[dim]):
				show(tensor.index_select(dim, torch.tensor(i, device=tensor.device)), dim=None, prefix=f'[{i}]' if prefix is None else f'{prefix}[{i}]')
			show(tensor, dim=None, prefix='[*]' if prefix is None else prefix + '[*]')
	elif tensors:
		for tensor in tensors:
			show(tensor, dim=dim, prefix=prefix)

# Calculate the SxS correlation coefficient matrix (-1 to 1) of a BxSxE tensor across the S sequence locations (mean correlation coefficient across B batches)
def calc_seq_corr(tensor: torch.Tensor):
	std, mean = torch.std_mean(tensor, correction=0, dim=(0, 2), keepdim=True)
	tensor -= mean
	return ((tensor @ tensor.transpose(1, 2)) / (tensor.shape[2] * (std @ std.transpose(1, 2)))).mean(dim=0)

#
# Miscellaneous utilities
#

# Format a float to a certain number of decimal places, but without trailing zeros
def format_semifix(value: float, precision: int):
	return f'{value:.{precision}f}'.rstrip('0').rstrip('.')

# Get the full class spec of a class
def get_class_str(cls):
	if cls.__module__ == 'builtins':
		return cls.__qualname__
	else:
		return f'{cls.__module__}.{cls.__qualname__}'

# Get the full spec of a type
def get_type_str(typ):
	typ_str = format(typ)
	if typ_str.startswith("<class '") and typ_str.endswith("'>"):
		return typ_str[8:-2]
	else:
		return typ_str

# Convert a noun to its canonical form (as per select_nouns.py)
def get_canon(noun: str, sanitize: bool) -> str:
	if sanitize:
		noun = unidecode.unidecode(' '.join(noun.split()))
	canon = noun.lower()
	canon = canon.replace("'", "").replace('.', '')
	canon = ' '.join(part for part in re.split(r'[\s/-]+', canon) if part)
	if set(canon) - ALLOWED_CHARS_CANON:
		log.warning(f"Invalid canon chars: {canon}")
	return canon

# Strictly construct a dataclass by type from a dict (dict keys and dataclass fields must match one-to-one, and values must all have correct type)
def dataclass_from_dict(cls, state: dict[str, Any]):
	fields = dataclasses.fields(cls)
	field_names = set(field.name for field in fields)
	state_names = set(state.keys())
	if field_names != state_names:
		raise ValueError(f"Cannot construct {cls.__qualname__} from dict that does not include exactly all the fields as keys for safety of correctness reasons => Dict is missing {sorted(field_names - state_names)} and has {sorted(state_names - field_names)} extra")
	field_types = typing.get_type_hints(cls)
	for field in fields:
		if not isinstance((value := state[field.name]), field_types[field.name]):  # noqa
			log.warning(f"{cls.__qualname__} field '{field.name}' should be type {get_type_str(field.type)} but got {get_class_str(type(value))}: {value}")
	return cls(**state)

# Shallow-apply to() to the fields of a dataclass
def dataclass_to(data, **kwargs):
	for field in dataclasses.fields(data):
		value = getattr(data, field.name)
		if isinstance(value, torch.nn.Module):
			value.to(**kwargs)
		elif isinstance(value, torch.Tensor):
			setattr(data, field.name, value.to(**kwargs))

# Flatten a nested dict with string keys by joining them with dots (no original string key may include a dot)
def flatten_dict(D, parent_key=None):
	F = {}
	for k, v in D.items():
		assert '.' not in k
		new_key = f"{parent_key}.{k}" if parent_key else k
		if isinstance(v, dict):
			F.update(flatten_dict(v, parent_key=new_key))
		else:
			F[new_key] = v
	return F

# Unflatten a dict that was flattened by flatten_dict()
def unflatten_dict(F):
	D = {}
	for c, v in F.items():
		parts = c.split('.')
		cursor = D
		for part in parts[:-1]:
			if part not in cursor:
				cursor[part] = {}
			cursor = cursor[part]
			if not isinstance(cursor, dict):
				raise ValueError(f"Nesting conflict at '{part}' while inserting '{c}'")
		leaf = parts[-1]
		if leaf in cursor:
			raise ValueError(f"Nesting conflict at '{leaf}' while inserting '{c}'")
		cursor[leaf] = v
	return D

# Attribute dict class
class AttrDict(dict):

	@classmethod
	def from_dict(cls, D: dict[str, Any]):
		return cls({k: cls.from_dict(v) if isinstance(v, dict) else v for k, v in D.items()})

	def __getattr__(self, key: str) -> Any:
		try:
			return self[key]
		except KeyError as e:
			raise AttributeError(key) from e

	def __setattr__(self, key: str, value: Any):
		self[key] = value

	def __delattr__(self, key: str):
		del self[key]

# Dump JSON to string with no indentation of lists
def json_dumps(obj: Any, *, indent: Union[int, str, None] = None, **kwargs) -> str:
	lines = []
	line_parts = []
	open_lists = 0
	for line in json.dumps(obj, indent=indent, **kwargs).splitlines():
		line_content = line.strip()
		if not line_content:
			continue
		if line_content[0] == ']':
			open_lists -= 1
		if line_content[-1] == '[':
			open_lists += 1
		if open_lists > 0:
			part = line_content if line_parts else line
			line_parts.append(part + ' ' if part[-1] == ',' else part)
		elif line_parts:
			line_parts.append(line_content)
			lines.append(''.join(line_parts))
			line_parts.clear()
		else:
			lines.append(line)
	assert open_lists == 0
	return '\n'.join(lines)

# Dump JSON to file with no indentation of lists
def json_dump(obj: Any, fp: IO[str], *, indent: Union[int, str, None] = None, **kwargs):
	fp.write(json_dumps(obj, indent=indent, **kwargs))

# Get the size in memory in bytes of common Python objects (best attempt to be as accurate as possible)
def get_size(obj, seen=None):
	if seen is None:
		seen = set()
	obj_id = id(obj)
	if obj_id in seen:
		return 0
	seen.add(obj_id)
	size = sys.getsizeof(obj)
	if has_dict := hasattr(obj, '__dict__'):
		size += get_size(obj.__dict__, seen)
	if obj is None or isinstance(obj, (int, float, bool, str, bytes, np.ndarray, np.generic, torch.storage.UntypedStorage, torch.dtype, torch.device)):
		pass
	elif isinstance(obj, torch.Tensor):
		size += get_size(obj.untyped_storage())
	elif isinstance(obj, dict):
		size += sum(get_size(key, seen) + get_size(val, seen) for key, val in obj.items())
	elif hasattr(obj, '__iter__'):
		size += sum(get_size(item, seen) for item in obj)
	elif not has_dict:
		raise TypeError(f"Unsure how to count memory size of object of type: {type(obj)}")
	return size

# Get the size in memory in floating point MiB of common Python objects (best attempt to be as accurate as possible)
def get_size_mb(obj):
	return get_size(obj) / 1048576

# Context manager that temporarily delays keyboard interrupts until the context manager exits
class DelayKeyboardInterrupt:

	def __init__(self):
		self.interrupted = False
		self.original_handler = None

	def __enter__(self):
		self.interrupted = False
		self.original_handler = signal.signal(signal.SIGINT, self.sigint_handler)

	# noinspection PyUnusedLocal
	def sigint_handler(self, signum, frame):
		print("Received SIGINT: Waiting for next opportunity to raise KeyboardInterrupt...")
		self.interrupted = True

	def __exit__(self, exc_type, exc_val, exc_tb):
		signal.signal(signal.SIGINT, self.original_handler)
		if self.interrupted:
			self.interrupted = False
			self.original_handler(signal.SIGINT, inspect.currentframe())
		self.original_handler = None

# Class that can be used to determine what classes occur in an object if it were to be pickled
class PickleClasses(pickle.Pickler):

	BUILTIN_TYPES = {
		types.NoneType,
		bool, int, float, complex,
		str, list, tuple, set, dict,
		bytes, bytearray,
		type, object, slice,
		types.BuiltinFunctionType, types.BuiltinMethodType,
	}

	class NullWriter:
		def write(self, data):
			pass

	def __init__(self, **kwargs):
		kwargs['file'] = self.NullWriter()
		super().__init__(**kwargs)
		self.classes = None

	def persistent_id(self, obj):
		self.classes.add(type(obj))
		return None

	def get_classes(self, obj, exclude_builtins=True) -> set[Type]:
		self.classes = set()
		self.dump(obj)
		classes = self.classes
		self.classes = None
		if exclude_builtins:
			classes.difference_update(self.BUILTIN_TYPES)
		return classes

# Class that can be used to determine what classes occur in an object if it were to be torch saved (new zipfile serialization)
class TorchSaveClasses(PickleClasses):

	TORCH_TYPES = {
		torch.Tensor,             # We do not need to explicitly specify torch storage types as the pickling does not descend inside torch.Tensor due to persistent_id()
		torch.dtype,              # PyTorch tensor data types
		collections.OrderedDict,  # Used by torch module state dicts
	}

	def persistent_id(self, obj):
		super().persistent_id(obj)
		if isinstance(obj, torch.Tensor):
			return 'tensor'
		else:
			return None

	def get_classes(self, obj, exclude_builtins=True, exclude_torch=True) -> set[Type]:
		classes = super().get_classes(obj, exclude_builtins=exclude_builtins)
		if exclude_torch:
			classes.difference_update(self.TORCH_TYPES)
		return classes

# Multi-instance progress bar
class ProgressBar:

	def __init__(self, **kwargs):
		self.kwargs = kwargs
		self.pbar = None

	def __enter__(self):
		self.stop()
		return self

	def start(self, **kwargs):
		if self.pbar is not None:
			self.pbar.close()
		self.pbar = tqdm.tqdm(**{**self.kwargs, **kwargs})

	def update(self, n=1, postfix=None):
		if self.pbar is not None:
			if postfix is not None:
				self.pbar.set_postfix_str(postfix, refresh=False)
			self.pbar.update(n=n)

	@classmethod
	def pause_display(cls, file=None):
		return tqdm.tqdm.external_write_mode(file=file)  # Context manager within which you can write to file (sys.stdout by default) without interfering with the progress bar

	def stop(self):
		if self.pbar is not None:
			self.pbar.close()
			self.pbar = None

	def __exit__(self, exc_type, exc_val, exc_tb):
		self.stop()
		return False
# EOF
