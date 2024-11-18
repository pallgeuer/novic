# Classification dataset

# Imports
import os
import re
import json
from typing import Callable, Optional, Sequence, Union
import PIL.Image
import torch.utils.data
import torchvision.datasets
from logger import log
import utils

# Constants
DATASET_NAMES = {'MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100', 'Food101', 'TinyImageNet', 'Imagenette', 'Imagewoof', 'ImageNet1K', 'ImageNet1KVal', 'ImageNet-A', 'ImageNet-R'}
DATASET_CASED_MAP = {name.lower(): name for name in DATASET_NAMES}
DATASET_SPLITS = ('train', 'valid', 'all')

# Classification dataset class
class ClassificationDataset(torch.utils.data.Dataset):

	def __init__(self, dataset: torch.utils.data.Dataset, paths: bool, path_optional: bool, cls_name: str, cls_split: str, cls_classes: Sequence[str]):

		self.dataset = dataset
		self.paths = paths
		self.path_optional = path_optional
		self.cls_name = cls_name
		self.cls_split = cls_split
		self.cls_classes = cls_classes if isinstance(cls_classes, tuple) else tuple(cls_classes)

		if not self.paths:
			self.dataset_type = -1
		elif isinstance(self.dataset, torchvision.datasets.ImageFolder):
			self.dataset_type = 1
		elif isinstance(self.dataset, torchvision.datasets.Food101):
			self.dataset_type = 2
		elif self.path_optional:
			self.dataset_type = 0
		else:
			raise ValueError(f"Unsupported classification dataset type if returning paths: {type(self.dataset)}")

	# noinspection PyUnresolvedReferences, PyProtectedMember, PyTypeChecker
	def __getitem__(self, index) -> Union[tuple[torch.Tensor, int], tuple[torch.Tensor, int, Optional[str]]]:
		if self.dataset_type == -1:
			return self.dataset.__getitem__(index)
		elif self.dataset_type == 0:
			return *self.dataset.__getitem__(index), None
		elif self.dataset_type == 1:
			return *self.dataset.__getitem__(index), self.dataset.samples[index][0]
		elif self.dataset_type == 2:
			return *self.dataset.__getitem__(index), str(self.dataset._image_files[index])
		else:
			raise AssertionError

	def __len__(self) -> int:
		# noinspection PyTypeChecker
		return len(self.dataset)

# Load an image classification dataset
def load_image_dataset(name: str, root_path: str, split: str, variant: Optional[str], clean: bool, image_transform: Callable[[PIL.Image.Image], torch.Tensor], paths: bool, path_optional: bool) -> ClassificationDataset:
	# name = Image classification dataset name (see DATASET_NAMES)
	# root_path = Image classification dataset root directory (set up datasets as described in https://github.com/pallgeuer/ReLish/blob/master/benchmark/commands.txt)
	# split = Which split of the image classification dataset to use (train = Training split, valid = Validation split, all = Training + validation splits)
	# variant = Variant of class names to use (None = Strictly the class names from the Dataset class, otherwise use data/cls_class_names_{variant}.json as the preferred source of class names with the Dataset class as a backup)
	# clean = Whether the class names retrieved from a variant JSON should be cleaned to not contain brackets, slashes, or's and similar
	# image_transform = Transform to use to convert each sample PIL image to a torch tensor
	# paths = Whether the returned dataset should return the corresponding sample image paths (error if not supported for a dataset, e.g. CIFAR10 and similar)
	# path_optional = If using paths, whether None should be returned instead of raising an error if the dataset does not support paths
	# Returns the classification dataset

	name_lower = name.lower()
	if name_lower not in DATASET_CASED_MAP:
		raise ValueError(f"Unsupported image classification dataset {name} (available: {', '.join(sorted(DATASET_NAMES))})")
	name = DATASET_CASED_MAP[name_lower]
	root_path = os.path.expanduser(root_path)
	split = split.lower()

	log.info(f"Loading {split} split of the {name} classification dataset...")

	if split == 'train':
		train_split = True
		valid_split = False
	elif split == 'valid':
		train_split = False
		valid_split = True
	elif split == 'all':
		train_split = True
		valid_split = True
	else:
		raise ValueError(f"Unrecognised image classification dataset split: {split}")

	train_dataset = valid_dataset = None
	if name in ('MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100'):
		folder_path = os.path.join(root_path, 'CIFAR') if name.startswith('CIFAR') else root_path
		dataset_class = getattr(torchvision.datasets, name)
		if train_split:
			train_dataset = dataset_class(root=folder_path, train=True, transform=image_transform)
		if valid_split:
			valid_dataset = dataset_class(root=folder_path, train=False, transform=image_transform)
	elif name == 'Food101':
		folder_path = os.path.join(root_path, name)
		dataset_class = getattr(torchvision.datasets, name)
		if train_split:
			train_dataset = dataset_class(root=folder_path, split='train', transform=image_transform)
		if valid_split:
			valid_dataset = dataset_class(root=folder_path, split='test', transform=image_transform)
	elif name in ('TinyImageNet', 'Imagenette', 'Imagewoof', 'ImageNet1K'):
		folder_map = {'TinyImageNet': 'tiny-imagenet-200', 'Imagenette': 'imagenette2-320', 'Imagewoof': 'imagewoof2-320', 'ImageNet1K': 'ILSVRC-CLS'}
		folder_path = os.path.join(root_path, name, folder_map[name])
		if train_split:
			train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(folder_path, 'train'), transform=image_transform)
		if valid_split:
			valid_dataset = torchvision.datasets.ImageFolder(root=os.path.join(folder_path, 'val'), transform=image_transform)
	elif name == 'ImageNet1KVal':  # Note: The root path should point directly to the directory containing the class subdirectories (which each contain the actual image files of that class)
		if train_split:
			raise ValueError(f"The {name} dataset only supports the validation split 'valid'")
		if valid_split:
			log.info(f"Using {name} dataset: {root_path}")
			root_entries = os.listdir(root_path)
			if not (len(root_entries) == 1000 and all(re.fullmatch(r'^n\d{8}$', entry) and os.path.isdir(os.path.join(root_path, entry)) for entry in root_entries)):
				raise ValueError("Root directory does not contain the required 1000 class directories and nothing else")
			valid_dataset = torchvision.datasets.ImageFolder(root=root_path, transform=image_transform)
			if len(valid_dataset) < 1000:
				raise ValueError(f"Expect at least 1000 samples: {len(valid_dataset)}")
	elif name in ('ImageNet-A', 'ImageNet-R'):
		if train_split:
			raise ValueError(f"The {name} dataset only supports the validation split 'valid'")
		if valid_split:
			valid_dataset = torchvision.datasets.ImageFolder(root=os.path.join(root_path, name, name.lower()), transform=image_transform)
	else:
		raise AssertionError

	if valid_split and train_split:
		dataset = valid_dataset + train_dataset
	elif valid_split:
		dataset = valid_dataset
	elif train_split:
		dataset = train_dataset
	else:
		raise AssertionError

	class_names = None
	if variant is not None:
		data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
		json_path = os.path.join(data_dir, f'cls_class_names_{variant}.json')
		with open(json_path, 'r') as file:
			cls_class_names: dict[str, list[str]] = json.load(file)
		lookup_name = name_lower
		if lookup_name.startswith('imagenet1k'):
			lookup_name = 'imagenet1k'
		if (class_names := cls_class_names.get(lookup_name, None)) is not None:
			if clean:
				clean_class_names = []
				bracket_regex = r'\([^()]*\)'
				for class_name in class_names:
					clean_class_name = class_name
					while re.search(bracket_regex, clean_class_name):
						clean_class_name = re.sub(bracket_regex, '', clean_class_name)
					clean_class_name = clean_class_name.split(sep='/', maxsplit=1)[0]
					if clean_class_name == 'bell or wind chime':
						clean_class_name = 'chime'
					parts = clean_class_name.split(' or ')
					if len(parts) >= 2:
						first_part = parts[0]
						last_part = parts[1]
						if len(first_part_words := first_part.split()) == 1 and len(last_part_words := last_part.split(maxsplit=1)) > 1:
							clean_class_name = f'{first_part_words[0]} {last_part_words[1]}'
						else:
							clean_class_name = first_part
					clean_class_name = ' '.join(clean_class_name.split())
					clean_class_names.append(clean_class_name)
					if clean_class_name != class_name:
						log.info(f"Cleaned class: {class_name} --> {clean_class_name}")
				class_names = clean_class_names
			log.info(f"Using {len(class_names)} {name} {'cleaned' if clean else 'uncleaned'} class names from {variant} JSON file: {json_path}")

	if class_names is None:
		if (class_names := getattr(dataset.datasets[0] if isinstance(dataset, torch.utils.data.ConcatDataset) else dataset, 'classes', None)) is not None and not any(cname[1:].isdigit() for cname in class_names):
			class_names = tuple(cname.replace('_', ' ') for cname in class_names)
			log.info(f"Using {len(class_names)} {name} class names from loaded class")
		else:
			raise ValueError(f"Failed to resolve {name} class names")

	dataset = ClassificationDataset(dataset=dataset, paths=paths, path_optional=path_optional, cls_name=name, cls_split=split, cls_classes=class_names)
	log.info(f"Loaded {dataset.cls_name}{'' if variant is None else f' {variant}'} classification dataset {dataset.cls_split} split with {len(dataset)} samples and {len(dataset.cls_classes)} classes")
	return dataset

# Load the prompts associated with an image classification dataset
def load_image_dataset_prompts(name: str, variant: str) -> tuple[tuple[str, bool], ...]:
	# name = Image classification dataset name (see DATASET_NAMES)
	# variant = Variant of prompts to use (source from data/cls_prompts_{variant}.json)
	# Returns a tuple of pairs (prompt, need_article)

	name_lower = name.lower()
	if name_lower not in DATASET_CASED_MAP:
		raise ValueError(f"Unsupported image classification dataset {name} (available: {', '.join(sorted(DATASET_NAMES))})")
	name = DATASET_CASED_MAP[name_lower]

	data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
	with open(os.path.join(data_dir, f'cls_prompts_{variant}.json'), 'r') as file:
		cls_prompts: dict[str, list[str]] = json.load(file)
	prompt_keymap = {'FashionMNIST': 'CIFAR10', 'TinyImageNet': 'CIFAR100', 'Imagenette': 'ImageNet1K', 'Imagewoof': 'ImageNet1K', 'ImageNet-A': 'ImageNet1K', 'ImageNet-R': 'ImageNet1K'}
	if (prompts := cls_prompts.get(name_lower, None)) is not None:
		log.info(f"Using {len(prompts)} {name} prompts from JSON file")
	elif (cls_dataset_alt := prompt_keymap.get(name, None)) is not None and (prompts := cls_prompts.get(cls_dataset_alt.lower(), None)) is not None:
		log.info(f"Using {len(prompts)} {name} prompts from JSON file after aliasing to {cls_dataset_alt}")
	else:
		raise ValueError(f"Failed to resolve {name} prompts")

	return tuple((prompt.replace('{c}', '{noun}'), False) for prompt in prompts)

# Create a data loader for an image classification dataset
def load_image_dataset_loader(dataset: ClassificationDataset, batch_size: int, num_workers: int, device_is_cpu: bool) -> torch.utils.data.DataLoader:
	# dataset = Image classification dataset
	# batch_size = Batch size to use
	# num_workers = Number of data loader workers to use
	# device_is_cpu = Whether the PyTorch device in use is the CPU
	# Returns the required data loader
	dataset_workers = 0 if utils.debugger_attached() else min(batch_size, num_workers)
	loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=dataset_workers, shuffle=True, collate_fn=utils.default_collate, pin_memory=not device_is_cpu, drop_last=False)
	log.info(f"Created shuffled data loader for {dataset.__class__.__name__}({utils.get_class_str(type(dataset.dataset))}) dataset class with {len(loader)} batches of nominal size {loader.batch_size} and {loader.num_workers} workers and no sample dropping")
	return loader
# EOF
