#!/usr/bin/env python3
#
# Examples:
#   extras/specificity/specificity_scores.py --specificity_file extras/specificity/_specificity_annotations.json --pred_dir outputs/ovod_XXX --ann_file extras/infer_images/_class_annotations.json
#   extras/specificity/specificity_scores.py --specificity_file extras/specificity/_specificity_annotations.json --pred_dir outputs/ovod_YYY --ann_file extras/sampled_images/wiki_images_c/_class_annotations.json

import os
import json
import pprint
import argparse

def parse_arguments():
	parser = argparse.ArgumentParser(description="Process predictions and calculate scores.")
	parser.add_argument('--specificity_file', required=True, help="Path to the specificity annotations file")
	parser.add_argument('--pred_dir', required=True, help="Directory containing prediction JSONs")
	parser.add_argument('--ann_file', required=True, help="Path to the class annotations file")
	return parser.parse_args()

def load_predictions(pred_dir):
	pred_data = {}
	for entry in os.listdir(pred_dir):
		with open(os.path.join(pred_dir, entry), 'r') as file:
			json_data = json.load(file)
			assert len(json_data['predictions']) == 1
			preds = tuple(item[0] for item in next(iter(json_data['predictions'].values()))['pred'])
			assert len(json_data['samples']) == len(preds)
			pred_data[entry] = {sample: pred for sample, pred in zip(json_data['samples'], preds, strict=True)}
	return pred_data

def load_annotations(ann_file, category_scores):
	with open(ann_file, 'r') as file:
		ann = json.load(file)
		return {image: {noun: category_scores[cat] for cat, noun_list in anns.items() for noun in noun_list} for image, anns in ann.items()}

def load_specificity(specificity_file):
	with open(specificity_file, 'r') as file:
		return json.load(file)

def calculate_scores(pred_data, ann, specificity):
	pred_scores = {entry: format(sum(ann[sample][pred] for sample, pred in preds.items()) / len(preds), '.2%') for entry, preds in pred_data.items()}
	overall_scores = {entry: format(sum(ann[sample][pred] * specificity[pred] for sample, pred in preds.items()) / len(preds), '.2%') for entry, preds in pred_data.items()}
	return pred_scores, overall_scores

def main():

	args = parse_arguments()

	category_scores = {
		"correct_primary": 1.0,
		"correct_secondary": 0.8,
		"close_primary": 0.5,
		"close_secondary": 0.4,
		"incorrect": 0.0,
	}

	pred_data = load_predictions(args.pred_dir)
	ann = load_annotations(args.ann_file, category_scores)
	specificity = load_specificity(args.specificity_file)

	pred_scores, overall_scores = calculate_scores(pred_data, ann, specificity)

	print("Prediction scores:")
	pprint.pprint(pred_scores, width=120)
	print()

	print("Overall scores:")
	pprint.pprint(overall_scores, width=120)
	print()

if __name__ == "__main__":
	main()
# EOF
