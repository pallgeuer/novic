import json
import os
import sys
import re
import time
from itertools import chain

import hydra.core.hydra_config
import omegaconf
import openai
from openai import OpenAI
from tqdm import tqdm
from unidecode import unidecode

client = OpenAI(api_key=os.environ.get("OPENAI_KEY"))


def get_num_captions(freq, num_captions_per_freq):
    try:
        freq = int(freq)
    except ValueError:
        raise ValueError("Input must be a valid integer")

    return min(freq, 10) * num_captions_per_freq


def get_prompt(noun, mode, num_captions=10):
    return (
        f"Generate a list of {num_captions} image captions each containing the target object \"{noun}\" ({mode}). "
        f"Use the style of the image captions for large language model pre-training. "
        f"Don't use poetic language. "
        f"Use a different sentence structure for each caption. "
        f"Output the list in the following form: <nr>. <caption>"
    )


def generate_captions(cfg, noun):
    """
    Generate a list of captions  using ChatGPT API.
    Requires setting the 'OPENAI_KEY' environment variable.
    :param cfg: Configuration file
    :param noun: Metadata dictionary of noun that should be central part of the caption
    :return:
    """

    count = 0
    num_api_calls = 0
    num_failed_requests = 0
    num_duplicates = 0
    while len(noun['singular_captions']) < noun['num_singular_captions'] \
            or len(noun['plural_captions']) < noun['num_plural_captions']:

        if len(noun['singular_captions']) < noun['num_singular_captions']:
            prompt = get_prompt(noun['pretty_noun'], 'singular', cfg.num_captions_per_call)
        else:
            prompt = get_prompt(noun['plurals'][noun['plurals_freq'].index(max(noun['plurals_freq']))], 'plural',
                                cfg.num_captions_per_call)

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1000,
                temperature=0.8,
                n=1
            )
            num_api_calls += 1
            num_failed_requests = 0

            lines = response.choices[0].message.content.strip().split('\n')
            for line in lines:
                duplicate = False

                if not re.match(r'^\s*\d+\s*\.', line):
                    if line:
                        with tqdm.external_write_mode():
                            print(f"[{noun['pretty_noun']}] Bad line: {line}")
                    continue

                caption = unidecode(line.split('.', maxsplit=1)[1].strip())

                # Rarely problems with formatting in the response can cause empty or single character strings
                if len(caption) < 10:
                    with tqdm.external_write_mode():
                        print(f"[{noun['pretty_noun']}] Too short caption: {caption}")
                    continue

                # Check if singular or plural target noun is contained in caption
                sing_prompt = None
                if noun['singulars']:
                    for n in chain((noun['pretty_noun'],), noun['singulars']):
                        regex = r'(^|\s)(' + re.escape(n) + r')(\.(?!$))?(\'s?|[,.?!:])?(\s|$)'
                        if re.search(regex, caption, flags=re.IGNORECASE):
                            prompt_template = re.sub(regex, r'\1{singular}\4\5', caption, flags=re.IGNORECASE).rstrip(
                                ',.?!:')
                            if prompt_template in noun['singular_captions']:
                                duplicate = True
                            else:
                                sing_prompt = prompt_template
                                break

                plural_prompt = None
                if noun['plurals']:
                    for n in noun['plurals']:
                        regex = r'(^|\s)(' + re.escape(n) + r')(\.(?!$))?(\'s?|[,.?!:])?(\s|$)'
                        if re.search(regex, caption, flags=re.IGNORECASE):
                            prompt_template = re.sub(regex, r'\1{plural}\4\5', caption, flags=re.IGNORECASE).rstrip(
                                ',.?!:')
                            if prompt_template in noun['plural_captions']:
                                duplicate = True
                            else:
                                plural_prompt = prompt_template
                                break

                if duplicate:
                    num_duplicates += 1

                if sing_prompt and plural_prompt:
                    missing_sing_prompts = noun['num_singular_captions'] - len(noun['singular_captions'])
                    missing_plural_prompts = noun['num_plural_captions'] - len(noun['plural_captions'])
                    if missing_sing_prompts <= missing_plural_prompts:
                        sing_prompt = None
                    else:
                        plural_prompt = None

                if sing_prompt:
                    noun['singular_captions'].append(sing_prompt)
                    count = 0

                if plural_prompt:
                    noun['plural_captions'].append(plural_prompt)
                    count = 0

                if not (sing_prompt or plural_prompt) and not duplicate:
                    with tqdm.external_write_mode():
                        print(f"[{noun['pretty_noun']}] Unmatched caption: {line}")

            if count > 3:
                with tqdm.external_write_mode():
                    print(f"[{noun['pretty_noun']}] Caught in infinite loop")
                break
            else:
                count += 1

        except openai.OpenAIError as e:
            with tqdm.external_write_mode():
                print(f"[{noun['pretty_noun']}] Error: {e}")
            num_failed_requests += 1

            if num_failed_requests > 5:
                with tqdm.external_write_mode():
                    print("Too many failed requests: Waiting 30min for next request.")
                time.sleep(1800)
            else:
                with tqdm.external_write_mode():
                    print("Too many failed requests: Waiting 10sec for next request.")
                time.sleep(10)

    return noun, num_api_calls, num_duplicates


@hydra.main(config_path="config", config_name="caption_generation", version_base=None)
def main(cfg: omegaconf.DictConfig):

    try:
        with open(cfg.vocab_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    # Check if caption file exists, otherwise create it
    try:
        with open(cfg.caption_path, 'r') as f:
            caption_data = json.load(f)

        # Copy missing words from vocab database into caption file
        ids = [noun["id"] for noun in caption_data]
        for noun in data:
            if noun["id"] not in ids:
                noun['singular_captions'] = []
                noun['plural_captions'] = []
                caption_data.append(noun)
                print(f"Added noun \"{noun['pretty_noun']}\" from vocab database to caption database.")

        data = caption_data

    except FileNotFoundError:
        # Initialize empty caption lists
        for i in range(len(data)):
            data[i]['singular_captions'] = []
            data[i]['plural_captions'] = []

    # Update expected number of captions
    for i in range(len(data)):
        if data[i]['singulars_freq']:
            data[i]['num_singular_captions'] = get_num_captions(sum(data[i]['singulars_freq']), cfg.num_captions_per_freq)
        else:
            data[i]['num_singular_captions'] = 0
        if data[i]['plurals_freq']:
            data[i]['num_plural_captions'] = get_num_captions(sum(data[i]['plurals_freq']), cfg.num_captions_per_freq)
        else:
            data[i]['num_plural_captions'] = 0

    with open(cfg.caption_path, "w") as f:
        json.dump(data, f, indent=2)

    total_api_calls = 0
    total_duplicates = 0
    last_saved = -1

    with tqdm(range(len(data)), desc="Target nouns", ncols=140, smoothing=0.02) as pbar:
        for i in pbar:

            pbar.set_postfix_str(
                f"{data[i]['pretty_noun']}, API={total_api_calls}, Dup={total_duplicates}, Saved={last_saved}")

            if sum(data[i].get("singulars_freq", [0]) + data[i].get("plurals_freq", [0])) >= cfg.freq_threshold:
                data[i], num_api_calls, num_duplicates = generate_captions(cfg, data[i])
                total_api_calls += num_api_calls
                total_duplicates += num_duplicates

            if (i + 1) % cfg.saving_freq == 0:
                with open(cfg.caption_path, "w") as f:
                    json.dump(data, f, indent=2)
                last_saved = i + 1

        pbar.set_postfix_str(
            f"{data[-1]['pretty_noun']}, API={total_api_calls}, Dup={total_duplicates}, Saved={last_saved}")

    with open(cfg.caption_path, "w") as f:
        json.dump(data, f, indent=2)
    print("Finished")


#
# Run
#

# Run main function
if __name__ == "__main__":
    main()
# EOF
