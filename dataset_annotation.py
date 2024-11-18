import cv2
import os
import argparse
from typing import IO
import json
from pathlib import Path
from urllib.parse import quote_plus
import numpy as np
import webbrowser
from typing import Any, Dict, List, Union, Optional, Set

# Constants for keyboard controls
KEYS = {
    'correct_primary': 'q',
    'correct_secondary': 'w',
    'close_primary': 'o',
    'close_secondary': 'p',
    'incorrect': 'i',
    'save': 's',
    'undo': 'u',
    'print_annotations': 'z',
    'google_noun_search': 'g',
    'google_image_search': 'h',
    'wordnet_search': 'j',
    'next_image': 'enter',
    'exit': 'esc'
}

# Maximum image dimensions for display
MAX_DIMENSIONS = (750, 1400)
MIN_DIMENSIONS = (640, 640)

# Text colors in BGR format
COLORS = {
    'yellow': (0, 255, 255),
    'cyan': (255, 255, 0),
    'white': (255, 255, 255)
}

# History stack for undo functionality
action_history = []


def configure_arg_parser():
    """Configure and return the argument parser."""
    parser = argparse.ArgumentParser(description="Dataset Annotation Tool")

    # Common options that are not mutually exclusive
    parser.add_argument(
        "--image_dir",
        type=str,
        help="Path to the directory containing images",
        default="datasets/test/images"
    )
    parser.add_argument(
        "--annotations_file",
        type=str,
        help="Path to the existing annotation JSON file",
        default="datasets/test/annotations/annotation.json"
    )
    parser.add_argument(
        "--k",
        type=int,
        help="Show top-k predictions per noun inference method",
        default=2
    )
    parser.add_argument(
        "--gen_cfg",
        nargs='+',
        type=str,
        help="Generation configurations (or, decoding strategies) for nouns; if not specified, use all",
        required=False
    )

    # Provide either predictions directory or predictions files as list
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--predictions_dirs",
        nargs='+',
        type=str,
        help="List of paths to directories that contain prediction files; "
             "either this or --predictions_files must be specified."
    )
    group.add_argument(
        "--predictions_files",
        nargs='+',
        type=str,
        help="List of paths to individual prediction JSON files; either this or --predictions_dir must be specified."
    )

    return parser


def json_dumps(obj: Any, *, indent: Union[int, str, None] = None, **kwargs):
    """Dump JSON to string with no indentation of lists"""
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


def json_dump(obj: Any, fp: IO[str], *, indent: Union[int, str, None] = None, **kwargs):
    """Dump JSON to file with no indentation of lists."""
    fp.write(json_dumps(obj, indent=indent, **kwargs))


def save_annotations(file_name: str, annotations: Dict):
    """ Save annotations to a file in JSON format. """
    with open(file_name, 'w') as f:
        f.write(json_dumps(annotations, indent=2))


def load_annotations(file_path: str):
    """ Load existing annotations from a JSON file or create a new one if not existing. """
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        if isinstance(e, FileNotFoundError):
            print(f"No existing annotation file found at {file_path}. Starting fresh.")
        else:
            print("Error reading the annotation file. Starting fresh.")
        return {}


def filter_predictions(predictions: Dict, annotations: Dict) -> Dict:
    """Filter predictions to exclude already annotated images."""
    filtered_predictions = {}
    for filename, preds in predictions.items():
        annotated_preds = set().union(*annotations.get(filename, {}).values())
        unannotated_preds = [pred for pred in preds if pred not in annotated_preds]
        if unannotated_preds:
            filtered_predictions[filename] = unannotated_preds
    return filtered_predictions


def get_predictions(prediction_files: List[Path], k: int, gen_cfg: Optional[List[str]] = None) -> Dict[str, Set]:
    """Load prediction data from JSON files."""
    all_predictions = {}
    for file_path in prediction_files:
        with open(file_path, 'r') as file:
            data = json.load(file)
            predictions = data['predictions']
            if gen_cfg:
                predictions = {key: value for key, value in predictions.items() if key in gen_cfg}
            for sample, preds in zip(data['samples'], zip(*[d['pred'] for d in predictions.values()])):
                preds_ = [pred[:k] for pred in preds]
                all_predictions.setdefault(sample, set()).update(*preds_)
    return all_predictions


def resize_image(img: np.array) -> np.array:
    """Resize image maintaining aspect ratio based on maximum and minimum dimensions."""
    h, w = img.shape[:2]
    max_height, max_width = MAX_DIMENSIONS
    min_height, min_width = MIN_DIMENSIONS

    scale_down = min(max_width / w, max_height / h) if h > max_height or w > max_width else 1
    scale_up = max(min_width / w, min_height / h) if h < min_height or w < min_width else 1
    scaling_factor = scale_up if (scale_up - 1) > (1 - scale_down) else scale_down

    if scaling_factor != 1:
        new_width = int(w * scaling_factor)
        new_height = int(h * scaling_factor)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return img


def handle_key_press(image_name, pred, annotations, args):
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (13, 10):  # Enter key to go to next image
            return 'next'
        elif key == 27:  # Esc key to stop
            return 'stop'
        elif chr(key) == KEYS['correct_primary']:
            annotations[image_name]['correct_primary'].append(pred)
            action_history.append((image_name, 'correct_primary', pred))
            print(f"Annotated {image_name} with '{pred}' labeled as correct primary.")
            return 'annotated'
        elif chr(key) == KEYS['correct_secondary']:
            annotations[image_name]['correct_secondary'].append(pred)
            action_history.append((image_name, 'correct_secondary', pred))
            print(f"Annotated {image_name} with '{pred}' labeled as correct secondary.")
            return 'annotated'
        elif chr(key) == KEYS['close_primary']:
            annotations[image_name]['close_primary'].append(pred)
            action_history.append((image_name, 'close_primary', pred))
            print(f"Annotated {image_name} with '{pred}' labeled as close primary.")
            return 'annotated'
        elif chr(key) == KEYS['close_secondary']:
            annotations[image_name]['close_secondary'].append(pred)
            action_history.append((image_name, 'close_secondary', pred))
            print(f"Annotated {image_name} with '{pred}' labeled as close secondary.")
            return 'annotated'
        elif chr(key) == KEYS['incorrect']:
            annotations[image_name]['incorrect'].append(pred)
            action_history.append((image_name, 'incorrect', pred))
            print(f"Annotated {image_name} with '{pred}' labeled as incorrect.")
            return 'annotated'
        elif chr(key) == KEYS['undo']:
            if action_history:
                last_image, last_category, last_pred = action_history.pop()
                annotations[last_image][last_category].remove(last_pred)
                print(f"Undo last action: removed '{last_pred}' from {last_category} of {last_image}.")
            else:
                print(f"Key press ignored: Nothing to undo.")
        elif chr(key) == KEYS['print_annotations']:
            k_width = max(len(k) for k in annotations[image_name]) + 1
            print(f"Current annotations for {image_name}: ")
            for k, v in annotations[image_name].items():
                print(f"{k:{k_width}}{v}")
        elif chr(key) == KEYS['save']:
            save_annotations(args.annotations_file, annotations)
            print(f"Annotations saved to '{args.annotations_file}'")
        elif chr(key) == KEYS['google_noun_search']:
            webbrowser.open(f"https://www.google.com/search?q=define:+{quote_plus(pred)}")
        elif chr(key) == KEYS['google_image_search']:
            webbrowser.open(f"https://www.google.com/search?tbm=isch&q={quote_plus(pred)}")
        elif chr(key) == KEYS['wordnet_search']:
            webbrowser.open(f"http://wordnetweb.princeton.edu/perl/webwn?s={quote_plus(pred)}")
        else:
            print(f"Invalid key: '{chr(key)}'")


def main():
    parser = configure_arg_parser()
    args = parser.parse_args()

    annotations = load_annotations(args.annotations_file)
    prediction_files = []
    if args.predictions_dirs is not None:
        prediction_files.extend(file for pred_dir in args.predictions_dirs for file in Path(pred_dir).glob('*.json'))
    if args.predictions_files is not None:
        prediction_files.extend(args.predictions_files)
    all_predictions = get_predictions(prediction_files, args.k, args.gen_cfg)
    predictions = filter_predictions(all_predictions, annotations)

    # Load each image contained in the predictions file
    for i, (image, preds) in enumerate(predictions.items()):
        img = cv2.imread(os.path.join(args.image_dir, image))
        if img is None:
            continue

        img = resize_image(img)

        # Add image name to annotations file
        if image not in annotations.keys():
            annotations[image] = {'close_primary': [],
                                  'close_secondary': [],
                                  'correct_primary': [],
                                  'correct_secondary': [],
                                  'incorrect': []}

        cv2.namedWindow("Dataset Annotation Tool", cv2.WINDOW_AUTOSIZE)

        result = 'stop'
        for j, pred in enumerate(preds):

            # Define the text to be displayed with the image
            text_left = [
                "",
                "Annotation:",
                f"'{KEYS['correct_primary'].upper()}': correct primary",
                f"'{KEYS['correct_secondary'].upper()}': correct secondary",
                f"'{KEYS['close_primary'].upper()}': close primary",
                f"'{KEYS['close_secondary'].upper()}': close secondary",
                f"'{KEYS['incorrect'].upper()}': incorrect",
                f"",
                f"Get help:",
                f"'{KEYS['google_noun_search'].upper()}': show noun definition",
                f"'{KEYS['google_image_search'].upper()}': show image",
                f"'{KEYS['wordnet_search'].upper()}': show WordNet definition",
                f"",
                f"Other actions:",
                f"'{KEYS['undo'].upper()}': undo",
                f"'{KEYS['print_annotations'].upper()}': display annotations",
                f"'{KEYS['save'].upper()}': save",
                f"'{KEYS['next_image'].upper()}': next image",
                f"'{KEYS['exit'].upper()}': exit"
            ]

            text_right = [
                "",
                f"Image {i + 1}/{len(predictions.keys())}",
                "",
                "Upcoming predictions:",
                *predictions[image][j + 1:j + 11]
            ]

            # Text settings
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = img.shape[1] * 0.001
            thickness = 2
            text_color = (255, 255, 255)  # White text
            padding = 10  # Padding between lines

            # Header settings
            header_font_scale = font_scale * 1.5
            header_thickness = thickness + 1
            header_text = pred

            def get_text_dimensions(text_list, font_, font_scale_, thickness_, padding_):
                max_width = 0
                total_height = 0
                for line in text_list:
                    size = cv2.getTextSize(line, font_, font_scale_, thickness_)[0]
                    max_width = max(max_width, size[0])
                    total_height += size[1] + padding_
                return max_width, total_height

            def place_text(image_, text_list, start_x, initial_y, font_, font_scale_, color, thickness_, padding_):
                current_y = initial_y
                for line in text_list:
                    text_size = cv2.getTextSize(line, font_, font_scale_, thickness_)[0]
                    cv2.putText(image_, line, (start_x, current_y + text_size[1]), font_, font_scale_, color, thickness_)
                    current_y += text_size[1] + padding_

            # Calculate space for header
            header_size = cv2.getTextSize(header_text, font, header_font_scale, header_thickness)[0]
            header_start_x = (img.shape[1] - header_size[0]) // 2

            # Calculate dimensions for text columns
            max_width1, total_height1 = get_text_dimensions(text_left, font, font_scale, thickness, padding)
            max_width2, total_height2 = get_text_dimensions(text_right, font, font_scale, thickness, padding)

            # Calculate new image height
            max_total_height = max(total_height1, total_height2)
            extended_height = img.shape[0] + max_total_height + header_size[1] + 3 * padding

            # Extend the image
            labeled_image = np.concatenate(
                (img, np.zeros((extended_height - img.shape[0], img.shape[1], 3), dtype=np.uint8)), axis=0)

            # Place header
            cv2.putText(labeled_image, header_text, (header_start_x, img.shape[0] + header_size[1] + padding), font,
                        header_font_scale, text_color, header_thickness)

            # Set starting Y position for other texts
            start_y_position = img.shape[0] + header_size[1] + 2 * padding

            # Place texts
            place_text(labeled_image, text_left, 2 * padding, start_y_position, font, font_scale, COLORS['yellow'], thickness,
                       padding)
            place_text(labeled_image, text_right, img.shape[1] - max_width2 - 14 * padding, start_y_position, font,
                       font_scale, COLORS['cyan'], thickness, padding)

            # Display the image
            cv2.imshow("Dataset Annotation Tool", labeled_image)

            result = handle_key_press(image, pred, annotations, args)
            if result in ['next', 'stop']:
                break

        # Check if all annotations are complete for an image
        existing_labels = set(lbl for category in annotations[image].values() for lbl in category)
        if set(predictions[image]).issubset(existing_labels):
            print(f"Annotations complete for '{image}'.")

        if result == 'stop':
            break

    cv2.destroyAllWindows()

    save_annotations(args.annotations_file, annotations)

    print(f"Annotations saved to '{args.annotations_file}'")


if __name__ == "__main__":
    main()
# EOF
