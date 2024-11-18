import re
import json
import unidecode

THRESHOLD = 1

with open(f'extras/data/captions_vt{THRESHOLD}.json', 'r') as file:
	data = json.load(file)

for item in data:

	if sum(item['singulars_freq']) + sum(item['plurals_freq']) < THRESHOLD:
		continue

	if len(item['singular_captions']) < item['num_singular_captions']:
		print(f"[{item['target_noun']}] Want {item['num_singular_captions']} singular captions but have {len(item['singular_captions'])}")
	if len(item['plural_captions']) < item['num_plural_captions']:
		print(f"[{item['target_noun']}] Want {item['num_plural_captions']} plural captions but have {len(item['plural_captions'])}")

	if len(item['singular_captions']) != len(set(item['singular_captions'])):
		print(f"[{item['target_noun']}] Singulars have duplicates")
	if len(item['plural_captions']) != len(set(item['plural_captions'])):
		print(f"[{item['target_noun']}] Plurals have duplicates")
	if len(set(item['singular_captions']).union(set(item['plural_captions']))) != len(item['singular_captions']) + len(item['plural_captions']):
		print(f"[{item['target_noun']}] Combined set has duplicates")

	singular_placeholder = {'{singular}'}
	plural_placeholder = {'{plural}'}
	if any('{singular}' not in caption for caption in item['singular_captions']):
		print(f"[{item['target_noun']}] Singulars don't have placeholder")
	if any(set(re.findall(r'\{[^{}]*}', caption)) != singular_placeholder for caption in item['singular_captions']):
		print()
		print(f"[ERROR][{item['target_noun']}] Singulars contain unexpected placeholders")
		print()
	if any('{plural}' not in caption for caption in item['plural_captions']):
		print(f"[{item['target_noun']}] Plurals don't have placeholder")
	if any(set(re.findall(r'\{[^{}]*}', caption)) != plural_placeholder for caption in item['plural_captions']):
		print()
		print(f"[ERROR][{item['target_noun']}] Plurals contain unexpected placeholders")
		print()

	if any(caption.endswith(punc) for caption in item['singular_captions'] for punc in ',.?!:'):
		print(f"[{item['target_noun']}] Some singular ends with punctuation")
	if any(caption.endswith(punc) for caption in item['plural_captions'] for punc in ',.?!:'):
		print(f"[{item['target_noun']}] Some plural ends with punctuation")

	if any(unidecode.unidecode(caption) != caption for caption in item['singular_captions']):
		print(f"[{item['target_noun']}] Some singular has unicode")
	if any(unidecode.unidecode(caption) != caption for caption in item['plural_captions']):
		print(f"[{item['target_noun']}] Some plural has unicode")
# EOF
