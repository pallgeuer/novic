#!/usr/bin/env python3
# Perform image object noun annotation using OpenAI GPTs
#
# Note:
#  - The environment variable OPENAI_API_KEY is required for any OpenAI API requests to work
#  - The small images directory (default name: 512px) can conveniently be created with:
#      mkdir -p 512px && for img in $(ls *.{jpg,jpeg,JPEG,webp,png} | sort); do if [[ -f "$img" ]]; then if [[ "$img" == *.jpg || "$img" == *.jpeg || "$img" == *.JPEG ]] && identify -format "%[fx:w]x%[fx:h]" "$img" | awk -Fx '{ exit !($1 <= 512 && $2 <= 512) }'; then echo "COPY: $img"; cp "$img" "512px/${img%.*}.jpg"; else echo "CONVERT: $img"; convert "$img" -auto-orient -resize 512x512\> -quality 85 -strip "512px/${img%.*}.jpg"; fi; fi; done
#  - The current remote files can be manually managed at: https://platform.openai.com/storage
#  - The current remote batches can be manually managed at: https://platform.openai.com/batches
#  - The cost estimates are underestimates for the batch API in that only requests that actually led to annotation progress count towards the token/cost totals
#  - The cost estimates when using batching are overestimates in that the 50% price reduction has not been factored in to keep things simple (for true usage refer to: https://platform.openai.com/usage)
#  - Very approximately, 500 requests executed via batch API costs around 1 USD

# Imports
import os
import re
import math
import time
import json
import base64
import random
import logging
import fnmatch
import argparse
import datetime
import itertools
import contextlib
import dataclasses
from typing import Sequence, Optional, Type, Any
import requests
import PIL.Image
import PIL.ImageFilter
import tqdm
import numpy as np
import openai
import utils

# Constants
CHARS_PER_TOKEN = 3.8  # Conservative estimate
TOKENS_PER_IMAGE = 85  # Low detail
TOKEN_COST_IN = 5  # USD per 1M tokens
TOKEN_COST_OUT = 15  # USD per 1M tokens
SAVER_INTERVAL = 30  # 30s
MAX_BACKOFF = 12 * 60 * 60  # 12h
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', None)
CHAT_COMPLETIONS_ENDPOINT = '/v1/chat/completions'
CHAT_COMPLETIONS_URL = 'https://api.openai.com' + CHAT_COMPLETIONS_ENDPOINT
REQUEST_HEADERS = {'Content-Type': 'application/json', 'Authorization': f'Bearer {OPENAI_API_KEY}'}
MIB_SIZE = 1 << 20

# Logging configuration
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

#
# Main
#

# Main function
def main():

	parser = argparse.ArgumentParser(description="Perform image object noun annotation using OpenAI GPTs")

	parser.add_argument('--action', type=str, help='Which action to take (new_state, annotate, annotate_batch, save_classes)', required=True)
	parser.add_argument('--state', type=str, help='Path to the state file to manage', required=True)

	parser.add_argument('--images_dir', type=str, help='New state: Directory of images to annotate')
	parser.add_argument('--images_small_dir', type=str, default='512px', help='New state: Directory of resized small images to use (resolved relative to images_dir if relative path, extensions MUST be exactly .jpg)')
	parser.add_argument('--batch_dir', type=str, default='batches', help='New state: Directory to store JSONL annotation batches (resolved relative to images_dir if relative path)')
	parser.add_argument('--model', type=str, default='gpt-4o-2024-05-13', help='New state: Which multimodal OpenAI GPT model to use')
	parser.add_argument('--opinions', type=int, nargs=2, default=[3, 5], help='New state: Minimum number of opinions sought for classification, and maximum number of opinions sought to resolve unclear classifications')
	parser.add_argument('--confidence', type=float, default=0.78, help='New state: Classification confidence required in the range (0.5, 1)')

	parser.add_argument('--from_state', type=str, help='Transfer: Path to the state file to transfer to the current one')

	parser.add_argument('--predictions', type=str, nargs='+', help='Annotate: Path to the predictions JSONs and/or directory of JSONs to consider')
	parser.add_argument('--topk', type=int, default=1, help='Annotate: Maximum top-k predictions to consider from the predictions JSONs')
	parser.add_argument('--min_nouns', type=int, default=8, help='Annotate: Minimum number of nouns to ask per request (nominal range is N to 2N-1)')
	parser.add_argument('--request_queue_mib', type=int, default=1024, help='Annotate: Approximate allowed request queue size in MiB (RAM memory, must be at least 128MiB)')

	parser.add_argument('--max_batch_requests', type=int, default=49000, help='Annotate batch: Approximate maximum number of requests allowed per batch (must be at least 20, may be exceeded by a few requests)')
	parser.add_argument('--max_batch_mib', type=int, default=90, help='Annotate batch: Approximate allowed JSONL batch size in MiB')
	parser.add_argument('--max_files_mib', type=int, default=5120, help='Annotate batch: Approximate maximum allowed total uploaded JSONL batches size in MiB')
	parser.add_argument('--max_pending_batches', type=int, default=50, help='Annotate batch: Maximum number of concurrent pending batches')
	parser.add_argument('--max_pending_ktokens', type=int, default=10000, help='Annotate batch: Approximate maximum number of thousand pending tokens (conservatively estimated, must be at least 5)')
	parser.add_argument('--unbatch_thres', type=int, default=50, help='Annotate batch: Last remaining batches under this number of requests are delegated to single annotation instead')

	parser.add_argument('--classes', type=str, help='Save classes: Class annotations file to save to')

	args = parser.parse_args()

	if args.action == 'new_state':
		action_new_state(args)
	elif args.action == 'transfer':
		action_transfer(args)
	elif args.action == 'annotate':
		action_annotate(args)
	elif args.action == 'annotate_batch':
		if action_annotate_batch(args):
			print('-' * 160)
			print("Continuing with single annotation...")
			action_annotate(args)
	elif args.action == 'save_classes':
		action_save_classes(args)
	elif args.action is None:
		raise ValueError("Please specify what action to take using --action")
	else:
		raise ValueError(f"Unrecognised action: {args.action}")

#
# Actions
#

# Action: Create new state
def action_new_state(args: argparse.Namespace):

	state_path = os.path.abspath(args.state)
	print(f"State file: {state_path}")
	if not state_path.endswith('.json'):
		raise ValueError(f"State file must have JSON extension: {state_path}")
	if os.path.exists(state_path):
		raise ValueError(f"State file already exists: {state_path}")
	state_dir = os.path.dirname(state_path)

	opinions_min, opinions_max = args.opinions
	assert isinstance(opinions_min, int) and isinstance(opinions_max, int)
	if opinions_min < 1 or opinions_max < 1 or opinions_max < opinions_min:
		raise ValueError(f"Invalid opinions specification: Min {opinions_min} Max {opinions_max}")

	confidence = args.confidence
	assert isinstance(confidence, float)
	if confidence <= 0.5 or confidence >= 1.0:
		raise ValueError(f"Invalid confidence specification: {confidence}")

	if args.images_dir is None:
		raise ValueError("Please provide an images directory using --images_dir")
	images_dir = os.path.abspath(args.images_dir)
	images_dir_rel = os.path.relpath(images_dir, start=state_dir)
	print(f"Collecting all image files in directory: {images_dir}")
	filenames = os.listdir(images_dir)
	images = sorted(set().union(*([filename for filename in filenames if fnmatch.fnmatch(name=filename.lower(), pat=pattern)] for pattern in utils.IMAGE_PATTERNS)))
	print(f"Found {len(images)} images and {len(filenames) - len(images)} non-images in the directory of {len(filenames)} entries")

	images_small_dir = os.path.join(images_dir, args.images_small_dir)
	images_small_dir_rel = os.path.relpath(images_small_dir, start=state_dir)
	images_small = [f'{os.path.splitext(image)[0]}.jpg' for image in images]
	assert len(images_small) == len(set(images_small)) == len(images)

	print(f"Verifying that all small images exist and are valid images at most 512x512 in size...")
	for image in images_small:
		image_path = os.path.join(images_small_dir, image)
		img = PIL.Image.open(image_path)
		img.load()
		if not (28 <= img.width <= 512 and 28 <= img.height <= 512):
			raise ValueError(f"Invalid encountered dimensions {img.size}: {image_path}")

	batch_dir = os.path.join(images_dir, args.batch_dir)
	batch_dir_rel = os.path.relpath(batch_dir, start=state_dir)

	state = dict(
		_meta=dict(
			_version=1,
			count=len(images),
			images_dir=images_dir_rel,
			images=images,
			images_small_dir=images_small_dir_rel,
			images_small=images_small,
			batch_dir=batch_dir_rel,
			model=args.model,
			opinions_min=opinions_min,
			opinions_max=opinions_max,
			confidence=confidence,
		),
		_metrics=dict(usage={}),
		annotations={image: {} for image in images},
		descriptions={image: {} for image in images},
		pending=dict(annotate_batch=[]),
	)

	with open(state_path, 'w') as file:
		utils.json_dump(state, file, indent=2, sort_keys=True)
	print(f"Created state file: {state_path}")

# Action: Transfer state
def action_transfer(args: argparse.Namespace):

	print("TARGET:")
	S, tgt_state_path, tgt_state_dir = load_state(args.state)
	if sum(len(pending_list) for pending_list in S['pending'].values()) > 0:
		raise ValueError(f"Cannot transfer as target annotations are pending: {sorted(S['pending'])}")
	saver = StateSaver(S=S, state_path=tgt_state_path, save_interval=SAVER_INTERVAL)

	print("SOURCE:")
	F, src_state_path, src_state_dir = load_state(args.from_state)
	if sum(len(pending_list) for pending_list in F['pending'].values()) > 0:
		raise ValueError(f"Cannot transfer as source annotations are pending: {sorted(F['pending'])}")

	Smeta = S['_meta']
	Fmeta = F['_meta']
	for key in sorted(set().union(Smeta, Fmeta)):
		if key not in Smeta:
			print(f"WARNING: Meta field in source state file but NOT in target state file: {key}")
		elif key not in Fmeta:
			print(f"WARNING: Meta field in target state file but NOT in source state file: {key}")
		elif key not in ('images', 'images_small'):
			Svalue = Smeta[key]
			Fvalue = Fmeta[key]
			if Svalue != Fvalue:
				print(f"WARNING: Meta field '{key}' differs from source to target: {Fvalue} vs {Svalue}")
	time.sleep(3)  # Note: Give a chance to Ctrl+C if unexpected warnings appear

	tgt_images_dir = os.path.abspath(os.path.join(tgt_state_dir, Smeta['images_dir']))
	src_images_dir = os.path.abspath(os.path.join(src_state_dir, Fmeta['images_dir']))
	tgt_images_small_dir = os.path.abspath(os.path.join(tgt_state_dir, Smeta['images_small_dir']))
	src_images_small_dir = os.path.abspath(os.path.join(src_state_dir, Fmeta['images_small_dir']))

	Sann = S['annotations']
	Fann = F['annotations']
	with tqdm.tqdm(tuple(zip(Smeta['images'], Smeta['images_small'], Sann.items(), strict=True)), desc='Transferring annotations', unit='img', unit_scale=False, dynamic_ncols=True) as progress_bar:
		for tgt_image, tgt_image_small, (tgt_image_, tgt_image_opinions) in progress_bar:
			assert tgt_image == tgt_image_
			changed = False
			if tgt_image in Fann:
				src_image = src_image_small = src_image_opinions = None
				for src_image, src_image_small, (src_image_, src_image_opinions) in zip(Fmeta['images'], Fmeta['images_small'], Fann.items(), strict=True):
					assert src_image == src_image_
					if src_image == tgt_image:
						break
				assert src_image == tgt_image and src_image_small == tgt_image_small
				tgt_image_pil = PIL.Image.open(os.path.join(tgt_images_dir, tgt_image))
				src_image_pil = PIL.Image.open(os.path.join(src_images_dir, src_image))
				tgt_image_small_pil = PIL.Image.open(os.path.join(tgt_images_small_dir, tgt_image_small))
				src_image_small_pil = PIL.Image.open(os.path.join(src_images_small_dir, src_image_small))
				verify_same_image(src_image_pil, src_image_small_pil, name=f'source image vs small image for {src_image}')
				verify_same_image(tgt_image_pil, tgt_image_small_pil, name=f'target image vs small image for {tgt_image}')
				verify_same_image(src_image_pil, tgt_image_pil, name=f'source vs target image for {tgt_image}')
				verify_same_image(src_image_small_pil, tgt_image_small_pil, name=f'source vs target small image for {tgt_image}')
				for src_noun, src_opinions in src_image_opinions.items():
					if src_noun not in tgt_image_opinions:
						tgt_image_opinions[src_noun] = src_opinions
						changed = True
			saver.update(changed=changed)

	saver.save()
	print("Finished state transfer")

# Action: Annotate images using single API requests
def action_annotate(args: argparse.Namespace):

	assert OPENAI_API_KEY is not None

	S, state_path, state_dir = load_state(args.state)
	if sum(len(pending_list) for pending_list in S['pending'].values()) > 0:
		raise ValueError(f"Cannot annotate as other annotations are pending: {sorted(S['pending'])}")

	if not args.predictions:
		raise ValueError("Please provide predictions JSONs using --predictions")
	pred_jsons = load_predictions(pred_json_paths=args.predictions)
	preds = load_prediction_sets(S=S, pred_jsons=pred_jsons, topk=args.topk)

	start_index = 0
	usage = S['_metrics']['usage']
	saver = StateSaver(S=S, state_path=state_path, save_interval=SAVER_INTERVAL)
	waiter = BackoffWaiter(max_wait=MAX_BACKOFF)

	while True:

		print('-' * 160)

		chat_requests, start_index = collect_chat_requests(S=S, state_dir=state_dir, preds=preds, min_nouns=args.min_nouns, request_queue_mib=args.request_queue_mib, start_index=start_index)
		if not chat_requests:
			break

		with tqdm.tqdm(chat_requests, desc='Single API requests', unit='req', unit_scale=False, dynamic_ncols=True) as progress_bar:
			for chat_request in progress_bar:
				try:
					response = requests.post(url=CHAT_COMPLETIONS_URL, headers=REQUEST_HEADERS, json=chat_request.payload)
					response.raise_for_status()
					response_json = response.json()
					with tqdm.tqdm.external_write_mode():
						process_response_json(S=S, image=chat_request.image, request_list=chat_request.request_list, response_json=response_json)
					tokens_in, tokens_out, tokens_total = usage.get('prompt_tokens', 0), usage.get('completion_tokens', 0), usage.get('total_tokens', 0)
					progress_bar.set_postfix_str(f"reqs={tqdm.tqdm.format_sizeof(usage.get('requests', 0))}, in={tqdm.tqdm.format_sizeof(tokens_in)}, out={tqdm.tqdm.format_sizeof(tokens_out)}, toks={tqdm.tqdm.format_sizeof(tokens_total)}, cost=${(tokens_in * TOKEN_COST_IN + tokens_out * TOKEN_COST_OUT) / 1e6:.2f}", refresh=True)
				except (requests.exceptions.RequestException, ValueError, KeyError, IndexError) as e:
					with tqdm.tqdm.external_write_mode():
						print(f"ERROR: {utils.get_class_str(type(e))}: {e}")
						saver.update(changed=False)
						waiter.update(success=False)
				else:
					with tqdm.tqdm.external_write_mode():
						saver.update(changed=True)
						waiter.update(success=True)

	print('-' * 160)
	saver.save()
	print("Finished single annotation")

# Action: Annotate images using batch API requests
def action_annotate_batch(args: argparse.Namespace) -> bool:

	assert OPENAI_API_KEY is not None
	client = openai.OpenAI()

	S, state_path, state_dir = load_state(args.state)
	meta = S['_meta']
	count = meta['count']
	pending = S['pending']
	if sum(len(pending_list) for action, pending_list in pending.items() if action != 'annotate_batch') > 0:
		raise ValueError(f"Cannot annotate as other annotations are pending: {sorted(pending)}")

	if not args.predictions:
		raise ValueError("Please provide predictions JSONs using --predictions")
	pred_jsons = load_predictions(pred_json_paths=args.predictions)
	preds = load_prediction_sets(S=S, pred_jsons=pred_jsons, topk=args.topk)
	del pred_jsons

	batch_dir = os.path.join(state_dir, meta['batch_dir'])
	with contextlib.suppress(OSError):
		os.mkdir(batch_dir)
		print(f"Created batch directory: {batch_dir}")

	max_batch_requests = args.max_batch_requests
	assert max_batch_requests >= 20
	assert args.max_batch_mib >= 1
	max_batch_size = args.max_batch_mib * MIB_SIZE
	max_files_size = args.max_files_mib * MIB_SIZE
	assert max_files_size >= 3 * max_batch_size
	max_pending_batches = args.max_pending_batches
	assert max_pending_batches >= 1
	assert args.max_pending_ktokens >= 5
	max_pending_tokens = args.max_pending_ktokens * 1000

	saver = StateSaver(S=S, state_path=state_path, save_interval=SAVER_INTERVAL)
	batch = ChatRequestBatch()
	BCS = BatchCollectState()

	pending_batches = pending['annotate_batch']
	if pending_batches:
		for pending_batch in pending_batches:
			print(f"Pending batch: {pending_batch['images'][0]} --> {pending_batch['images'][-1]} ({pending_batch['start_index']}-{pending_batch['last_index']})")
		BCS.start_index = pending_batches[-1]['stop_index']
		BCS.stop_index = pending_batches[0]['start_index']
		assert sum(circdist(pending_batch['start_index'], pending_batch['stop_index'], count) for pending_batch in pending_batches) + sum((pending_batch_2['start_index'] - pending_batch_1['stop_index']) % count for pending_batch_1, pending_batch_2 in itertools.pairwise(pending_batches)) == circdist(BCS.stop_index, BCS.start_index, count)  # Note: Ensures the pending batches do not make more than one lap of the images and are in 'order' without overlaps
		BCS.chat_requests_full = (BCS.start_index == BCS.stop_index)

	print('-' * 160)

	annotate_after = False
	while True:

		if BCS.chat_requests_full:
			assert BCS.start_index == BCS.stop_index
			chat_requests: list[ChatRequest] = []
		else:
			chat_requests, BCS.start_index = collect_chat_requests(S=S, state_dir=state_dir, preds=preds, min_nouns=args.min_nouns, request_queue_mib=args.request_queue_mib, start_index=BCS.start_index, stop_index=BCS.stop_index)
			if BCS.start_index == BCS.stop_index:
				BCS.chat_requests_full = True

		if chat_requests:

			for chat_request in chat_requests:
				if not batch.chat_requests:
					assert chat_request.first_for_image
					batch.start_index = chat_request.image_index
				batch.chat_requests.append(chat_request)
				batch.jsonl.append(chat_request_jsonl := json.dumps(dict(custom_id=f'{len(batch.chat_requests)}:{chat_request.image}', method='POST', url=CHAT_COMPLETIONS_ENDPOINT, body=chat_request.payload)) + '\n')
				batch.jsonl_size += len(chat_request_jsonl.encode('utf-8'))
				batch.num_tokens += chat_request.num_tokens
				if (len(batch.chat_requests) >= max_batch_requests or batch.jsonl_size >= max_batch_size or batch.num_tokens >= max_pending_tokens - chat_request.num_tokens * 10) and chat_request.last_for_image:
					batch.stop_index = (chat_request.image_index + 1) % count
					send_batch(client=client, S=S, saver=saver, BCS=BCS, batch=batch, batch_dir=batch_dir, max_pending_batches=max_pending_batches, max_pending_tokens=max_pending_tokens, max_files_size=max_files_size)
					batch = ChatRequestBatch()

		elif BCS.chat_requests_full:

			if batch.chat_requests:
				if not pending_batches and len(batch.chat_requests) <= args.unbatch_thres:
					print(f"Stopping batch annotation and leaving remaining {len(batch.chat_requests)}+ requests to single annotation")
					annotate_after = True
					break
				last_chat_request = batch.chat_requests[-1]
				assert last_chat_request.last_for_image
				batch.stop_index = (last_chat_request.image_index + 1) % count
				send_batch(client=client, S=S, saver=saver, BCS=BCS, batch=batch, batch_dir=batch_dir, max_pending_batches=max_pending_batches, max_pending_tokens=max_pending_tokens, max_files_size=max_files_size)
				batch = ChatRequestBatch()
			elif not pending_batches:
				print("Stopping batch annotation as nothing left to do")
				break

			if not wait_for_batch(client=client, S=S, saver=saver, BCS=BCS, batch_dir=batch_dir):
				time.sleep(1)

	saver.save()
	print("Finished batch annotation")
	return annotate_after

# Action: Save class annotations file
def action_save_classes(args: argparse.Namespace):

	S, state_path, state_dir = load_state(args.state)
	if sum(len(pending_list) for pending_list in S['pending'].values()) > 0:
		print(f"WARNING: Some annotations are still pending: {sorted(S['pending'])}")

	meta = S['_meta']
	opinions_min = meta['opinions_min']
	opinions_max = meta['opinions_max']
	assert 1 <= opinions_min <= opinions_max
	confidence = meta['confidence']
	assert 0.5 < confidence < 1
	ann = S['annotations']

	cls_ann = {}
	warn_general_num = []
	warn_close_num = []
	for image, img_ann in ann.items():
		correct = []
		close = []
		incorrect = []
		for noun, opinions in img_ann.items():
			num_opinions = len(opinions)
			if not opinions_min <= num_opinions <= opinions_max:
				warn_general_num.append((image, noun))
			num_correct = sum(opinions)
			num_incorrect = num_opinions - num_correct
			if num_correct / num_opinions >= confidence:
				correct.append(noun)
			elif num_incorrect / num_opinions >= confidence:
				incorrect.append(noun)
			else:
				close.append(noun)
				if num_opinions != opinions_max:
					warn_close_num.append((image, noun))
		cls_ann[image] = dict(correct_primary=correct, correct_secondary=[], close_primary=close, close_secondary=[], incorrect=incorrect)

	if warn_general_num:
		print(f"WARNING: The following image-noun pairs have an unexpected number of opinions (expect {opinions_min}-{opinions_max}): {', '.join(f'{image}/{noun}' for image, noun in warn_general_num)}")
	if warn_close_num:
		print(f"WARNING: The following CLOSE image-noun pairs have an unexpected number of opinions (expect {opinions_max}): {', '.join(f'{image}/{noun}' for image, noun in warn_close_num)}")

	cls_ann_path = os.path.abspath(args.classes)
	with open(cls_ann_path, 'w') as file:
		utils.json_dump(cls_ann, file, indent=2)
	print(f"Saved class annotations file: {cls_ann_path}")

#
# Helper classes
#

# Chat request class
@dataclasses.dataclass(frozen=True)
class ChatRequest:
	image_index: int
	first_for_image: bool
	last_for_image: bool
	image: str
	request_list: list[str]
	payload: dict[str, Any]
	num_tokens: int

# Chat request batch class
@dataclasses.dataclass
class ChatRequestBatch:
	chat_requests: list[ChatRequest] = dataclasses.field(default_factory=list)
	jsonl: list[str] = dataclasses.field(default_factory=list)
	jsonl_size: int = 0
	num_tokens: int = 0
	start_index: Optional[int] = None
	stop_index: Optional[int] = None

# Batch collection state class
@dataclasses.dataclass
class BatchCollectState:
	start_index: int = 0              # The next image index to generate chat requests for in the next call
	stop_index: int = 0               # One past the last image index to generate chat requests for in the next call
	chat_requests_full: bool = False  # If the start and stop indices are equal, whether all (False) or none (True) of the images need chat requests generated for them
	batch_errors: int = 0             # Number of batches that have errored

# Backoff waiter class
class BackoffWaiter:

	def __init__(self, allowed_failures: int = 3, base_wait: float = 10.0, max_wait: float = 3600.0, factor: float = 2.0):
		self.allowed_failures = allowed_failures
		self.base_wait = base_wait
		self.max_wait = max_wait
		self.factor = factor
		assert self.allowed_failures >= 0 and self.max_wait >= self.base_wait > 0 and self.factor >= 1
		self.max_exponent = math.log(self.max_wait / self.base_wait) / math.log(self.factor)
		assert self.max_exponent >= 0
		self.num_failures = 0

	def update(self, success: bool):
		if success:
			self.num_failures = 0
		else:
			self.num_failures += 1
			if self.num_failures >= self.allowed_failures + 1:
				wait_time = self.base_wait * (self.factor ** min(float(self.num_failures - self.allowed_failures - 1), self.max_exponent))
				print(f"Waiting {wait_time:.0f}s...")
				time.sleep(wait_time)

# State saver class
class StateSaver:

	def __init__(self, S: dict[str, Any], state_path: str, save_interval: float = 600.0):
		self.S = S
		self.state_path = state_path
		assert os.path.isfile(self.state_path)
		self.save_interval = save_interval
		self.last_save = time.perf_counter()
		self.changed = False

	def save(self):
		with open(self.state_path, 'w') as file:
			utils.json_dump(self.S, file, indent=2, sort_keys=True)
		self.last_save = time.perf_counter()
		self.changed = False

	def update(self, changed: bool):
		self.changed |= changed
		if self.changed and time.perf_counter() >= self.last_save + self.save_interval:
			self.save()

#
# Helper functions
#

# Load state file
def load_state(state: str) -> tuple[dict[str, Any], str, str]:
	state_path = os.path.abspath(state)
	state_dir = os.path.dirname(state_path)
	print(f"Loading state: {state_path}")
	with open(state_path, 'r') as file:
		S = json.load(file)
	meta = S['_meta']
	ann = S['annotations']
	assert meta['count'] == len(meta['images']) == len(meta['images_small']) == len(ann) and list(ann.keys()) == meta['images']
	print(f"Loaded a total of {sum(len(opinions) for img_ann in ann.values() for opinions in img_ann.values())} opinions for {sum(len(img_ann) for img_ann in ann.values())} noun annotations for {sum(bool(img_ann) for img_ann in ann.values())}/{len(ann)} images")
	return S, state_path, state_dir

# Load prediction JSON files
def load_predictions(pred_json_paths: Sequence[str]) -> dict[str, dict[str, Any]]:

	pred_json_files = set()
	for path in pred_json_paths:
		if path:
			path = os.path.abspath(path)
			if os.path.isdir(path):
				pred_json_files.update(os.path.join(path, json_file) for json_file in fnmatch.filter(os.listdir(path), '*.json'))
			else:
				if not path.endswith('.json'):
					raise ValueError(f"Predictions must be JSON files: {path}")
				pred_json_files.add(path)
	pred_json_files = sorted(pred_json_files)

	pred_jsons = {}
	for path in pred_json_files:
		print(f"Loading predictions JSON: {path}")
		with open(path, 'r') as file:
			pred_jsons[path] = json.load(file)
	print(f"Loaded {len(pred_jsons)} predictions JSONs")

	return pred_jsons

# Load prediction sets from prediction JSON files
def load_prediction_sets(S: dict[str, Any], pred_jsons: dict[str, dict[str, Any]], topk: int) -> dict[str, set[str]]:
	assert topk >= 1
	preds = {image: set() for image in S['_meta']['images']}
	for path, pred_json in pred_jsons.items():
		pred_samples = pred_json['samples']
		assert len(set(pred_samples)) == len(pred_samples)
		if any(image not in preds for image in pred_samples):
			raise ValueError(f"Samples in predictions JSON are not a subset of the state images: {path}")
		for gencfg, gencfg_preds in pred_json['predictions'].items():
			for image, sample_preds in zip(pred_samples, gencfg_preds['pred'], strict=True):
				preds[image].update(sample_preds[:topk])
	print(f"Loaded a total of {sum(len(pred_set) for pred_set in preds.values())} noun predictions for {sum(bool(pred_set) for pred_set in preds.values())} images")
	return preds

# Collect chat requests from prediction sets
def collect_chat_requests(S: dict[str, Any], state_dir: str, preds: dict[str, set[str]], min_nouns: int, request_queue_mib: int, start_index: int = 0, stop_index: int = 0) -> tuple[list[ChatRequest], int]:

	assert min_nouns >= 1
	assert request_queue_mib >= 128
	request_queue_size = request_queue_mib * MIB_SIZE

	meta = S['_meta']
	count = meta['count']
	model = meta['model']
	opinions_min = meta['opinions_min']
	opinions_max = meta['opinions_max']
	assert 1 <= opinions_min <= opinions_max
	confidence = meta['confidence']
	assert 0.5 < confidence < 1
	ann = S['annotations']
	images_small_dir = os.path.join(state_dir, meta['images_small_dir'])

	seen_objects = set()
	chat_requests = []
	chat_requests_size = 0

	image_index = -1
	for image_index, (image, image_small, (image_, image_opinions), (image__, image_pred_set)) in enumerate(zip(meta['images'], meta['images_small'], ann.items(), preds.items(), strict=True)):

		assert image == image_ == image__
		if image_index < start_index:
			continue
		if image_index >= stop_index > start_index:
			image_index -= 1
			break

		opinions_reqd = {}
		opinions_done = set()
		for noun, opinions in image_opinions.items():
			num_opinions = len(opinions)
			num_correct = sum(opinions)
			num_incorrect = num_opinions - num_correct
			num_reqd = max(math.ceil(min((confidence * num_opinions - num_correct) / (1 - confidence), (confidence * num_opinions - num_incorrect) / (1 - confidence), opinions_max - num_opinions)), opinions_min - num_opinions, 0)
			if num_reqd > 0:
				opinions_reqd[noun] = num_reqd
			else:
				opinions_done.add(noun)
		for noun in image_pred_set:
			if noun not in opinions_done:
				opinions_reqd.setdefault(noun, opinions_min)
		opinions_done = list(opinions_done)
		random.shuffle(opinions_done)

		if not opinions_reqd:
			continue
		total_opinions_reqd = sum(opinions_reqd.values())
		assert total_opinions_reqd >= 1

		num_requests = max(max(opinions_reqd.values()), math.ceil(total_opinions_reqd / (2 * min_nouns - 1)))
		request_lists = [[] for _ in range(num_requests)]
		index = 0
		for noun, num_reqd in sorted(opinions_reqd.items(), key=lambda item: (-item[1], item[0])):
			for _ in range(num_reqd):
				request_lists[index].append(noun)
				index = (index + 1) % num_requests
		assert all(request_lists)

		num_pad_nouns = min(num_requests * (min_nouns - len(request_lists[index])) - index, num_requests * len(opinions_done))
		if num_pad_nouns > 0:
			pad_count, pad_extra = divmod(num_pad_nouns, len(opinions_done))
			assert (pad_count > 0 or pad_extra > 0) and (pad_count < num_requests or pad_extra == 0)
			for n, noun in enumerate(opinions_done):
				for _ in range(pad_count + (n < pad_extra)):
					request_lists[index].append(noun)
					index = (index + 1) % num_requests

		assert all(len(request_list) == len(set(request_list)) for request_list in request_lists)

		for request_list in request_lists:
			random.shuffle(request_list)
		random.shuffle(request_lists)

		assert image_small.lower().endswith(('.jpg', '.jpeg'))
		with open(os.path.join(images_small_dir, image_small), 'rb') as file:
			img_data = base64.b64encode(file.read()).decode('utf-8')

		for request_index, request_list in enumerate(request_lists, 1):
			assert request_list
			payload = dict(model=model, max_tokens=512, temperature=0.2, top_p=0.6, messages=[
				dict(role='system', content=(
						"You are an AI assistant that has one and only one narrow task that you should strictly adhere to at all times. Given an image and an enumerated list of nouns by the user, "
						"you should first describe everything you see in the image in complete detail, and then provide an exactly matching enumerated list that for each noun provided by the user "
						"(explicitly repeat the noun) strictly classifies the noun into one of exactly two categories using a single word - Correct or Incorrect. Correct means that at least one "
						"instance of the noun is visible in the image. Incorrect means nothing very visually similar to the noun is visible in the image."
					)),
				dict(role='user', content=[
						{'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{img_data}', 'detail': 'low'}},
						{'type': 'text', 'text': '\n'.join(f'{n}) {noun}' for n, noun in enumerate(request_list, 1))},
					]),
			])
			num_tokens = round(sum((sum((TOKENS_PER_IMAGE if part['type'] == 'image_url' else len(part['text']) / CHARS_PER_TOKEN) for part in content) if isinstance((content := message['content']), (list, tuple)) else len(content) / CHARS_PER_TOKEN) for message in payload['messages']))  # noqa
			chat_requests.append(ChatRequest(image_index=image_index, first_for_image=(request_index == 1), last_for_image=(request_index == len(request_lists)), image=image, request_list=request_list, payload=payload, num_tokens=num_tokens))
			chat_requests_size += utils.get_size(payload, seen=seen_objects)

		if chat_requests_size >= request_queue_size:
			break

	print(f"Collected {len(chat_requests)} requests totalling to {chat_requests_size / MIB_SIZE:.0f}MiB for {image_index - start_index + 1} images: {meta['images'][start_index]} --> {meta['images'][image_index]} ({start_index}-{image_index})")
	return chat_requests, (image_index + 1) % count

# Process the response JSON of a request and update the state
def process_response_json(S: dict[str, Any], image: str, request_list: list[str], response_json: dict[str, Any]):

	assert request_list
	request_list_num = len(request_list)

	response_usage = response_json['usage']
	if not all(isinstance(usage_type, str) and isinstance(count, int) and count >= 0 for usage_type, count in response_usage.items()) or 'requests' in response_usage:
		raise ValueError(f"{image}: Unexpected usage format: {response_usage}")
	response_usage['requests'] = 1

	response_generation = response_json['choices'][0]
	if response_generation['finish_reason'] != 'stop':
		raise ValueError(f"{image}: Response generation did not finish normally: {response_generation['finish_reason']}")
	response_text = response_generation['message']['content']
	if not response_text:
		raise ValueError(f"{image}: Empty response generation")

	num = 0
	last_end = 0
	description = None
	annotations = {}

	dot_matches = tuple(re.finditer(r'^([1-9][0-9]*)\.[^\S\n]*(.*?)[^\S\n]*$', response_text, flags=re.MULTILINE | re.IGNORECASE))
	bracket_matches = tuple(re.finditer(r'^([1-9][0-9]*)\)[^\S\n]*(.*?)[^\S\n]*$', response_text, flags=re.MULTILINE | re.IGNORECASE))
	both_dot_bracket = bool(dot_matches) and bool(bracket_matches)
	if len(dot_matches) > request_list_num or both_dot_bracket:
		dot_matches = tuple(re.finditer(r'^([1-9][0-9]*)\.[^\S\n]*(.*?(?:correct|incorrect))[^\S\n]*$', response_text, flags=re.MULTILINE | re.IGNORECASE))
	if len(bracket_matches) > request_list_num or both_dot_bracket:
		bracket_matches = tuple(re.finditer(r'^([1-9][0-9]*)\)[^\S\n]*(.*?(?:correct|incorrect))[^\S\n]*$', response_text, flags=re.MULTILINE | re.IGNORECASE))

	if dot_matches and bracket_matches:
		print(f"WARNING: {image}: Both enumerated dot matches and bracket matches were found: {response_text!r}")
	if len(dot_matches) == request_list_num:
		matches = dot_matches
	elif len(bracket_matches) == request_list_num:
		matches = bracket_matches
	else:
		raise ValueError(f"{image} requested nouns {request_list} but failed to parse matching number of enumerated outputs: {response_text!r}")

	for num, (requested_noun, match) in enumerate(zip(request_list, matches, strict=True), 1):

		content_since_last = response_text[last_end:match.start()].strip()
		if last_end == 0:
			description = ' '.join(content_since_last.split())
		elif content_since_last:
			print(f"WARNING: {image}: Ignoring unexpected content in response: {content_since_last!r} ==> Complete response was: {response_text!r}")
		last_end = match.end()

		parsed_num = int(match.group(1))
		if parsed_num != num:
			raise ValueError(f"{image}: Got enumerated line index {parsed_num} but expected {num}: {response_text!r}")

		if not requested_noun:
			annotations[requested_noun] = False
		else:
			requested_noun_canon = utils.get_canon(requested_noun, sanitize=True)
			line_content = match.group(2)
			if not (line_match := re.fullmatch(r"(.*?)['.\W\s]+(correct|incorrect)", ' '.join(line_content.split()), flags=re.IGNORECASE)):
				raise ValueError(f"{image}: Failed to parse enumerated line: {line_content!r}")
			raw_parsed_noun = line_match.group(1)
			parsed_noun = raw_parsed_noun.strip('*_')  # Note: To strip possible bold/emphasis markdown
			if utils.get_canon(parsed_noun, sanitize=True) != requested_noun_canon:
				raise ValueError(f"{image}: Parsed noun '{raw_parsed_noun}' could not be matched to requested noun '{requested_noun}' in line: {line_content!r}")
			annotations[requested_noun] = (len(line_match.group(2)) == 7)  # Cheap way of distinguishing correct/incorrect (only possible options due to regex) without having to worry about casing

	if content_after_last := response_text[last_end:].strip():
		print(f"WARNING: {image}: Ignoring content after response: {content_after_last!r} ==> Complete response was: {response_text!r}")
	if not (len(annotations) == request_list_num == num and set(annotations) == set(request_list)):
		raise ValueError(f"{image}: Parsed annotations have unexpected length or content => Does request list have duplicates? => {request_list}")

	if not annotations:
		return

	meta = S['_meta']
	opinions_min = meta['opinions_min']
	opinions_max = meta['opinions_max']
	assert 1 <= opinions_min <= opinions_max
	confidence = meta['confidence']
	assert 0.5 < confidence < 1
	usage = S['_metrics']['usage']
	image_descriptions = S['descriptions'][image]
	image_opinions = S['annotations'][image]
	assert isinstance(usage, dict) and isinstance(image_descriptions, dict) and isinstance(image_opinions, dict)

	for usage_type, count in response_usage.items():
		if usage_type in usage:
			usage[usage_type] += count
		else:
			usage[usage_type] = count

	if description in image_descriptions:
		image_descriptions[description] += 1
	else:
		image_descriptions[description] = 1

	for noun, opinion in annotations.items():
		if (opinions := image_opinions.get(noun, None)) is None:
			image_opinions[noun] = (opinions := [])
		num_opinions = len(opinions)
		num_correct = sum(opinions)
		num_incorrect = num_opinions - num_correct
		num_reqd = max(math.ceil(min((confidence * num_opinions - num_correct) / (1 - confidence), (confidence * num_opinions - num_incorrect) / (1 - confidence), opinions_max - num_opinions)), opinions_min - num_opinions, 0)
		if num_reqd > 0:
			opinions.append(opinion)

# Send a batch for generation
def send_batch(client: openai.OpenAI, S: dict[str, Any], saver: StateSaver, BCS: BatchCollectState, batch: ChatRequestBatch, batch_dir: str, max_pending_batches: int, max_pending_tokens: int, max_files_size: int):

	print('-' * 160)

	meta = S['_meta']
	count = meta['count']
	count_width = len(format(count - 1, 'd'))
	images = meta['images']
	pending_batches = S['pending']['annotate_batch']

	last_index = (batch.stop_index - 1) % count
	print(f"Sending off new batch for generation: {images[batch.start_index]} --> {images[last_index]} ({batch.start_index}-{last_index})")
	print(f"Batch has {len(batch.chat_requests)} requests, {batch.jsonl_size / MIB_SIZE:.0f}MiB JSONL, at most approx {batch.num_tokens / 1000:.3g} input ktokens")
	assert len(batch.chat_requests) == len(batch.jsonl) >= 1
	if pending_batches:
		assert circdist(pending_batches[0]['start_index'], pending_batches[-1]['stop_index'], count) + ((batch.start_index - pending_batches[-1]['stop_index']) % count) + circdist(batch.start_index, batch.stop_index, count) == circdist(pending_batches[0]['start_index'], batch.stop_index, count)  # Note: We make sure the new batch does not overlap existing batches or violate the ordering requirement

	pending_batch = dict(
		start_index=batch.start_index,
		last_index=last_index,
		stop_index=batch.stop_index,
		image_indices=[chat_request.image_index for chat_request in batch.chat_requests],
		images=[chat_request.image for chat_request in batch.chat_requests],
		request_lists=[chat_request.request_list for chat_request in batch.chat_requests],
		json_file=f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{batch.start_index:0{count_width}d}_{last_index:0{count_width}d}_{images[batch.start_index].replace('.', '_')}_{images[last_index].replace('.', '_')}.jsonl",
		json_file_size=batch.jsonl_size,
		num_tokens=batch.num_tokens,
	)
	assert not any(pending['json_file'] == pending_batch['json_file'] for pending in pending_batches)

	assert not pending_batch['num_tokens'] > max_pending_tokens > 0
	while len(pending_batches) >= max_pending_batches > 0 or sum(pending['json_file_size'] for pending in pending_batches) + pending_batch['json_file_size'] > max_files_size > 0 or sum(pending['num_tokens'] for pending in pending_batches) + pending_batch['num_tokens'] > max_pending_tokens > 0:
		if not wait_for_batch(client=client, S=S, saver=saver, BCS=BCS, batch_dir=batch_dir):
			time.sleep(1)

	with utils.DelayKeyboardInterrupt(), contextlib.ExitStack() as stack:

		local_json_file = os.path.join(batch_dir, pending_batch['json_file'])
		stack.callback(delete_local_json, local_json_file=local_json_file)
		with open(local_json_file, 'w', encoding='utf-8') as file:  # Note: Explicitly specify UTF-8 encoding (which also has no BOM bytes)
			file.writelines(batch.jsonl)
			print(f"Created local batch JSONL: {local_json_file}")
			assert os.fstat(file.fileno()).st_size == pending_batch['json_file_size']

		print(f"Uploading local batch JSONL: {local_json_file}")
		remote_json_file = client.files.create(file=open(local_json_file, 'rb'), purpose='batch')
		stack.callback(delete_remote_json, client=client, remote_json_file_id=remote_json_file.id, name='batch JSONL')
		pending_batch['remote_json_file_id'] = remote_json_file.id
		print(f"Uploaded local batch JSONL as remote batch JSONL: {remote_json_file.id}")

		print(f"Launching remote batch from remote batch JSONL: {remote_json_file.id}")
		remote_batch = client.batches.create(completion_window='24h', endpoint=CHAT_COMPLETIONS_ENDPOINT, input_file_id=remote_json_file.id, metadata=dict(host=os.uname().nodename, script=__file__, action='annotate_batch', batch_local=local_json_file, batch_remote=remote_json_file.id))  # noqa
		pending_batch['remote_batch'] = remote_batch.id
		print(f"Launched remote batch: {remote_batch.id}")

		stack.pop_all()
		try:
			pending_batches.append(pending_batch)
			saver.save()
		except (OSError, TypeError, RecursionError, ValueError):
			print(f"Encountered error saving state file after successfully sending off the following batch (so manually add it):\n{utils.json_dumps(pending_batch, indent=2, sort_keys=True)}")
			print(f"State file: {saver.state_path}")
			raise

# Wait for a batch to completely finish
def wait_for_batch(client: openai.OpenAI, S: dict[str, Any], saver: StateSaver, BCS: BatchCollectState, batch_dir: str) -> bool:

	count = S['_meta']['count']
	usage = S['_metrics']['usage']
	pending_batches = S['pending']['annotate_batch']

	done_batches = []
	while pending_batches:

		print(f"\x1b[2K\rWaiting for next of {len(pending_batches)} pending batches to complete... ", end='', flush=True)

		for batch_index, pending_batch in enumerate(pending_batches):
			remote_batch = remote_batch_content = None
			try:
				remote_batch = client.batches.retrieve(batch_id=pending_batch['remote_batch'])
				if remote_batch.status in ('failed', 'completed', 'expired', 'cancelled'):
					if remote_batch.status != 'completed':
						print(f"ERROR: Remote batch {remote_batch.id} has status '{remote_batch.status}' with errors: {remote_batch.errors}")
					if remote_batch.output_file_id:
						remote_batch_content = tuple(json.loads(line) for line in client.files.content(file_id=remote_batch.output_file_id).text.splitlines() if line)
					done_batches.append((batch_index, remote_batch, remote_batch_content))
			except (openai.OpenAIError, RuntimeError, AttributeError, ValueError, json.JSONDecodeError) as e:
				print(f"ERROR: {utils.get_class_str(type(e))}: {e}")
				done_batches.append((batch_index, remote_batch, remote_batch_content))

		if done_batches:
			break

		time.sleep(60)

	print(f"\x1b[2K\r", end='', flush=True)

	if not done_batches:
		return False

	print('-' * 160)
	print(f"Found {len(done_batches)} pending batches that are done (completed or failed)")
	with utils.DelayKeyboardInterrupt():

		batch_errors = len(done_batches)
		for batch_index, remote_batch, remote_batch_content in done_batches:

			pending_batch = pending_batches[batch_index]
			images = pending_batch['images']

			if remote_batch_content:

				assert remote_batch is not None
				print(f"Processing results of remote batch: {remote_batch.id}")
				if len(remote_batch_content) != len(images):
					print(f"ERROR: Remote batch {remote_batch.id} has an unexpected number of responses {len(remote_batch_content)} for the given {len(images)} chat requests")

				response_count = 0
				for response in remote_batch_content:
					try:
						request_index, request_image = response['custom_id'].split(':', maxsplit=1)
						request_index = int(request_index) - 1
						if request_index < 0 or request_index >= len(images):
							raise ValueError(f"Invalid request index: {request_index}")
						if images[request_index] != request_image:
							raise ValueError(f"Request image mismatch: {request_image} vs {images[request_index]}")
						process_response_json(S=S, image=request_image, request_list=pending_batch['request_lists'][request_index], response_json=response['response']['body'])
					except (AttributeError, TypeError, ValueError, KeyError, IndexError) as e:
						print(f"RESPONSE ERROR: {utils.get_class_str(type(e))}: {e}")
					else:
						response_count += 1
				if response_count != len(remote_batch_content):
					print(f"WARNING: Remote batch {remote_batch.id} has only {response_count} valid responses out of {len(remote_batch_content)}")

				if response_count >= 0.8 * len(images):
					batch_errors -= 1

			if remote_batch is not None:
				if remote_batch.output_file_id:
					with print_errors(openai.OpenAIError, RuntimeError):
						delete_remote_json(client=client, remote_json_file_id=remote_batch.output_file_id, name='output JSONL')
				if remote_batch.error_file_id:
					with print_errors(openai.OpenAIError, RuntimeError):
						delete_remote_json(client=client, remote_json_file_id=remote_batch.error_file_id, name='errors JSONL')

			with print_errors(openai.OpenAIError, RuntimeError):
				delete_remote_json(client=client, remote_json_file_id=pending_batch['remote_json_file_id'], name='batch JSONL')
			delete_local_json(local_json_file=os.path.join(batch_dir, pending_batch['json_file']))

		for batch_index in sorted((index for index, _, _ in done_batches), reverse=True):
			remote_batch_id = pending_batches[batch_index]['remote_batch']
			del pending_batches[batch_index]
			print(f"Deleted remote batch from state file: {remote_batch_id}")
		saver.save()

	tokens_in, tokens_out, tokens_total = usage.get('prompt_tokens', 0), usage.get('completion_tokens', 0), usage.get('total_tokens', 0)
	print(f"Total accumulated usage: reqs={tqdm.tqdm.format_sizeof(usage.get('requests', 0))}, in={tqdm.tqdm.format_sizeof(tokens_in)}, out={tqdm.tqdm.format_sizeof(tokens_out)}, toks={tqdm.tqdm.format_sizeof(tokens_total)}, cost=${(tokens_in * TOKEN_COST_IN + tokens_out * TOKEN_COST_OUT) / 1e6:.2f}")

	BCS.batch_errors += batch_errors
	if BCS.batch_errors >= 5:
		raise RuntimeError("Many batches have experienced errors => Aborting for safety")

	if pending_batches:
		BCS.stop_index = pending_batches[0]['start_index']
		assert sum(circdist(pending_batch['start_index'], pending_batch['stop_index'], count) for pending_batch in pending_batches) + sum((pending_batch_2['start_index'] - pending_batch_1['stop_index']) % count for pending_batch_1, pending_batch_2 in itertools.pairwise(pending_batches)) == circdist(BCS.stop_index, pending_batches[-1]['stop_index'], count)  # Note: Ensures the pending batches do not make more than one lap of the images and are in 'order' without overlaps
		BCS.chat_requests_full = (BCS.start_index == BCS.stop_index)
	else:
		BCS.stop_index = BCS.start_index
		BCS.chat_requests_full = False

	return True

# Delete a local batch JSONL file by absolute path
def delete_local_json(local_json_file: str):
	with contextlib.suppress(OSError):
		os.remove(local_json_file)
		print(f"Deleted local batch JSONL: {local_json_file}")

# Delete a remote JSONL file by file ID
def delete_remote_json(client: openai.OpenAI, remote_json_file_id: str, name: str):
	response = client.files.delete(file_id=remote_json_file_id)
	assert response.id == remote_json_file_id
	if response.deleted:
		print(f"Deleted remote {name}: {remote_json_file_id}")
	else:
		raise RuntimeError(f"FAILED to delete remote {name}: {remote_json_file_id}")

# Verify that two PIL images are essentially the same beyond easily visible differences
def verify_same_image(image_a: PIL.Image.Image, image_b: PIL.Image.Image, name: str):

	area_a = image_a.width * image_a.height
	area_b = image_b.width * image_b.height
	if area_a >= area_b:
		image_l = image_a
		image_s = image_b
	else:
		image_l = image_b
		image_s = image_a

	if image_l.width >= image_l.height:
		aspect_error_px = abs(image_s.height - image_s.width * (image_l.height / image_l.width))
	else:
		aspect_error_px = abs(image_s.width - image_s.height * (image_l.width / image_l.height))
	if aspect_error_px > 1.5:
		raise ValueError(f"Images do not have same aspect ratio: {name}")

	if image_l.size == image_s.size:
		image_ls = image_l
	else:
		image_ls = image_l.resize(image_s.size, resample=PIL.Image.LANCZOS)

	image_ls = image_ls.convert('RGB').filter(PIL.ImageFilter.GaussianBlur(radius=4))
	image_s = image_s.convert('RGB').filter(PIL.ImageFilter.GaussianBlur(radius=4))
	rmse = np.sqrt(np.mean(np.square(np.array(image_ls) - np.array(image_s))))
	if rmse > 2.0:
		raise ValueError(f"Images are too dissimilar with RGB RMSE {rmse:.1f} > 2: {name}")

# Context manager that prints errors instead of raising them
@contextlib.contextmanager
def print_errors(*errors: Type[BaseException]):
	try:
		yield
	except errors as e:
		print(f"ERROR: {utils.get_class_str(type(e))}: {e}")

# Calculate the distance between two indices in a circular buffer (assuming that if the indices are equal the full buffer is being spanned)
def circdist(src, dst, size):
	return ((dst - src - 1) % size) + 1

#
# Run
#

# Run main function
if __name__ == "__main__":
	main()
# EOF
