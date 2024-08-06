#!/usr/bin/env python3

import argparse
import os
import re
import sys
import copy

def _parse_sample_args():
    argp = argparse.ArgumentParser(prog="aria sample")
    argp.add_argument("-m", help="name of model config file")
    argp.add_argument("-c", help="path to model checkpoint")
    argp.add_argument( # paths to all the midi files we will use 
        "-p",
        help="paths to all midi files",
        type=str,
        nargs="+",
        )
    argp.add_argument(
        "-form",
        help="form of the generation",
        type=str,
    )
    argp.add_argument(
        "-pt", help="sample using the pretrained model", action="store_true"
    )
    argp.add_argument(
        "-temp",
        help="change temp value",
        type=float,
        required=False,
        default=0.95,
    )
    argp.add_argument(
        "-top_p",
        help="change top_p value",
        type=float,
        required=False,
        default=0.95,
    )
    argp.add_argument(
        "-metadata",
        nargs=2,
        metavar=("KEY", "VALUE"),
        action="append",
        help="manually add metadata key-value pair when sampling",
    )
    argp.add_argument(
        "-var",
        help="number of variations",
        type=int,
        default=1,
    )
    argp.add_argument(
        "-trunc",
        help="length (in seconds) of the prompt",
        type=int,
        default=20,
    )
    argp.add_argument("-e", action="store_true", help="enable force end")
    argp.add_argument("-l", type=int, help="generation length", default=1024)
    argp.add_argument("-noise", action="store_true", help="add noise to prompt")
    argp.add_argument("-compile", action="store_true", help="compile cudagraph")

    return argp.parse_args(sys.argv[2:])


def _get_model_name(name: str | None, state: dict):
    if name is not None:
        return name

    print("Model name is not provided. Trying to infer from checkpoint...")
    _defaults = {
        16: "small",
        32: "medium",
        64: "large",
    }
    try:
        pattern = re.compile(r"encode_layers\.(\d+)\.")
        layer_keys = [pattern.search(k) for k in state.keys()]
        layer_keys = set(p.group(1) for p in layer_keys if p is not None)
        for i in range(len(layer_keys)):
            assert str(i) in layer_keys

        if len(layer_keys) in _defaults:
            print(f"Selecting model name: {_defaults[len(layer_keys)]}")
            return _defaults[len(layer_keys)]
        assert False
    except:
        raise ValueError("Model name is not provided and cannot be inferred.")


# TODO: Add support for sampling from the pretrained model
def sample(args):
    """Entrypoint for sampling"""

    from torch.cuda import is_available as cuda_is_available
    from aria.inference import TransformerLM
    from aria.model import ModelConfig
    from aria.config import load_model_config, load_config
    from aria.tokenizer import AbsTokenizer, SeparatedAbsTokenizer
    from aria.sample import greedy_sample, decode_tokens, get_pt_prompt, get_inst_prompt
    from aria.data.midi import MidiDict
    from aria.data.datasets import _noise_midi_dict
    from aria.utils import midi_to_audio, _load_weight

    if not cuda_is_available():
        raise Exception("CUDA device is not available.")

    model_state = _load_weight(args.c, "cuda")
    model_state = {
        k: v for k, v in model_state.items() if "rotary_emb" not in k
    }

    manual_metadata = {k: v for k, v in args.metadata} if args.metadata else {}
    valid_metadata = load_config()["data"]["metadata"]["manual"]
    for k, v in manual_metadata.copy().items():
        assert k in valid_metadata.keys(), f"{manual_metadata} is invalid"
        if v not in valid_metadata[k]:
            print(f"Ignoring invalid manual metadata: {k}")
            print(f"Please choose from {valid_metadata[k]}")
            del manual_metadata[k]

    num_variations = args.var
    truncate_len = args.trunc
    force_end = args.e
    model_name = args.m

    if args.pt == True:
        tokenizer = AbsTokenizer(return_tensors=True)
    else:
        tokenizer = SeparatedAbsTokenizer(return_tensors=True)

    model_config = ModelConfig(**load_model_config(model_name))
    model_config.set_vocab_size(tokenizer.vocab_size)
    model_config.grad_checkpoint = False
    model = TransformerLM(model_config).cuda()

    try:
        model.load_state_dict(model_state)
    except Exception as e:
        print(
            "Failed to load model_state. This is likely due to an incompatibility "
            "between the checkpoint file (-c) and model name/config (-m)."
        )
        if args.pt:
            print(
                "When using the -pt flag make sure you provide a checkpoint for "
                "the pretrained model."
            )
        else:
            print(
                "When not using the -pt flag make sure you provide a checkpoint "
                " for the instuct-finetuned (inst) model."
            )

        raise e

    assert args.l > 0, "Generation length must be positive."
    max_new_tokens = args.l

    # Load and format prompts and metadata
    paths_to_midis = args.p # [tests/test_data/1, tests/test_data/2, tests/test_data/3, ...]
    form = args.form # could be 'ABA', 'ABC', BAC'

    all_midi_dicts = {} # will have all of the midi dicts
    paths_to_midis_index = 0
    for idx in range(len(form)):
        if paths_to_midis_index == len(paths_to_midis):
            break

        midi_dict = MidiDict.from_midi(mid_path=paths_to_midis[paths_to_midis_index])

        for k, v in manual_metadata.items():
            midi_dict.metadata[k] = v

        print(f"Extracted metadata: {midi_dict.metadata}")
        print(
            f"Instruments: {set([MidiDict.get_program_to_instrument()[msg['data']] for msg in midi_dict.instrument_msgs])}"
        )

        # we want all_midi_dicts['A'] = midi_dict_from_1
        if form[idx] not in all_midi_dicts.keys():
            all_midi_dicts[form[idx]] = midi_dict
            paths_to_midis_index += 1

    print(f"Number of MIDIs: {len(paths_to_midis)}")

    # structure
    print(f"Form to follow: {form}")

    results = [] # where to put all of the generations
    token_labels = [[] for _ in range(num_variations)]
    generated = {} # holds each generation's tensor
    for idx_section, section in enumerate(form):
        if args.pt:
            if args.noise:
                print("Noising not supported with pretrained model")

            prompt_seq = get_pt_prompt(
                tokenizer=tokenizer,
                midi_dict=all_midi_dicts[section],
                truncate_len=truncate_len,
            )
        else:
            prompt_seq = get_inst_prompt(
                tokenizer=tokenizer,
                midi_dict=all_midi_dicts[section],
                truncate_len=truncate_len,
                noise=args.noise,
            )

        prompts = [prompt_seq for _ in range(num_variations)]
        if len(prompt_seq) + args.l > model_config.max_seq_len:
            print(
                "WARNING: Required context exceeds max_seq_len supported by model"
            )

        print(prompt_seq)

        # pick which generation to use
        if section in generated: # already generated for
            raw_results = copy.deepcopy(generated[section])
        else:
            raw_results = greedy_sample(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=max_new_tokens,
                force_end=True,
                # cfg_gamma=args.cfg,
                temperature=args.temp,
                top_p=args.top_p,
                compile=args.compile,
            )
            generated[section] = raw_results
        decoded_results = decode_tokens(tokenizer, raw_results)

        # removal of unwanted tokens
        print(f"Before manipulation: {decoded_results}")
        for idx_decoded, decoded in enumerate(decoded_results): # for each decoded variant
            print(f"Section {section} before (var {idx_decoded}): {decoded}")
            print(f"Section Index: {idx_section}")
            print(f"Length of form: {len(form)}")
            if idx_section == 0: # first section - remove eos
                decoded_results[idx_decoded] = [note for note in decoded if note != tokenizer.eos_tok]
            elif idx_section > 0 and idx_section < len(form) - 1: # middle section - remove metadata and certain tags and eos
                decoded = [note for note in decoded if note[0] != 'prefix']
                decoded_results[idx_decoded] = [note for note in decoded if note != '<S>' and note != '<INST>' and note != '</INST>' and note != '<E>']
            elif idx_section == len(form) - 1: # end section - remove metadata and certain tags
                print(f"Last section")
                decoded = [note for note in decoded if note[0] != 'prefix']
                print(f"Removed metadata: {decoded}")
                decoded_results[idx_decoded] = [note for note in decoded if note != '<S>' and note != '<INST>' and note != '</INST>']
                # print(f"Removed other tags: {decoded_results[idx_decoded]}")

            if decoded_results[idx_decoded][-1][0] == "piano": # ended with piano token
                decoded_results[idx_decoded] = decoded_results[idx_decoded][:-1]
            elif decoded_results[idx_decoded][-1][0] == "onset": # ended with onset token
                decoded_results[idx_decoded] = decoded_results[idx_decoded][:-2]

            print(f"Section {section} after (var {idx_decoded}): {decoded_results[idx_decoded]}")

        # add to labels the current section for each token
        for idx_decoded, decoded in enumerate(decoded_results):
            token_labels[idx_decoded] += [section for _ in range(len(decoded))]

        # onset changes
        if idx_section != 0: # at least one generation in results
            last_end_time = [0 for _ in range(len(results))] # last ending time for a note in each variant
            for idx_variation, variation in enumerate(results):
                for idx_token in range(len(variation) - 1):
                    if type(variation[idx_token]) is tuple and variation[idx_token][0] == "onset":
                        cur_end_time = variation[idx_token][1] + variation[idx_token + 1][1] # onset + dur
                        last_end_time[idx_variation] = max(last_end_time[idx_variation], cur_end_time)
            print(f"Last end times: {last_end_time}")

            for idx_decoded, decoded in enumerate(decoded_results):
                print(f"Decoded before onset: {decoded}")
                for idx_token in range(len(decoded)):
                    if type(decoded[idx_token]) is tuple and decoded[idx_token][0] == "onset":
                        new_onset = last_end_time[idx_decoded] + decoded[idx_token][1]
                        decoded_results[idx_decoded][idx_token] = ("onset", new_onset) # replace with new onset
                print(f"Decoded after onset: {decoded_results[idx_decoded]}")
        
        # concatenation
        if not results: # empty list
            for idx_variation, variation in enumerate(decoded_results): # decoded (next generation)
                results.append(variation)
        else: 
            for idx_variation, variation in enumerate(results): # final decoded (previous generation)
                results[idx_variation] += decoded_results[idx_variation]

    print(f"Before detokenizing: {results}")

    samples_dir = os.path.join(os.path.dirname(__file__), "..", "samples")
    if os.path.isdir(samples_dir) is False:
        os.mkdir(samples_dir)

    for idx, tokenized_seq_labels in enumerate(token_labels):
        output_file = os.path.join(samples_dir, f"label_{idx + 1}.txt")
        with open(output_file, 'w') as file:
            for idx, token in enumerate(tokenized_seq_labels):
                if not isinstance(token, str):
                    token = str(token)
                file.write(f"Token {idx + 1}: {token}\n")
    
    print("Labels saved to samples/")

    for idx, tokenized_seq in enumerate(results):
        res_midi_dict = tokenizer.detokenize(tokenized_seq)
        res_midi = res_midi_dict.to_midi()
        res_midi.save(f"samples/res_{idx + 1}.mid")

    print("Results saved to samples/")


def _parse_midi_dataset_args():
    argp = argparse.ArgumentParser(prog="aria midi-dataset")
    argp.add_argument("dir", help="directory containing midi files")
    argp.add_argument("save_path", help="path to save dataset")
    argp.add_argument("-r", action="store_true", help="recursively search dirs")
    argp.add_argument(
        "-s", action="store_true", help="shuffle dataset", default=False
    )
    argp.add_argument(
        "-metadata",
        nargs=2,
        metavar=("KEY", "VALUE"),
        action="append",
        help="manually add metadata key-value pair when building dataset",
    )
    argp.add_argument(
        "-split", type=float, help="create train/val split", required=False
    )

    return argp.parse_args(sys.argv[2:])


def build_midi_dataset(args):
    """Entrypoint for building MidiDatasets from a directory"""
    from aria.data.datasets import MidiDataset

    assert args.dir, "build directory must be provided"
    manual_metadata = {k: v for k, v in args.metadata} if args.metadata else {}
    MidiDataset.build_to_file(
        dir=args.dir,
        save_path=args.save_path,
        recur=args.r,
        overwrite=True,
        manual_metadata=manual_metadata,
        shuffle=args.s,
    )

    if args.split:
        assert 0.0 < args.split < 1.0, "Invalid range given for -split"
        MidiDataset.split_from_file(
            load_path=args.save_path,
            train_val_ratio=args.split,
            repeatable=True,
        )


def _parse_pretrain_dataset_args():
    argp = argparse.ArgumentParser(prog="aria pretrain-dataset")
    argp.add_argument("-load_path", help="path midi_dict dataset")
    argp.add_argument("-save_dir", help="path to save dataset")
    argp.add_argument(
        "-tokenizer_name", help="tokenizer name", choices=["abs", "rel"]
    )
    argp.add_argument("-l", help="max sequence length", type=int, default=4096)
    argp.add_argument("-e", help="num epochs", type=int, default=1)

    return argp.parse_args(sys.argv[2:])


def build_pretraining_dataset(args):
    from aria.tokenizer import AbsTokenizer, RelTokenizer
    from aria.data.datasets import PretrainingDataset

    if args.tokenizer_name == "abs":
        tokenizer = AbsTokenizer()
    elif args.tokenizer_name == "rel":
        tokenizer = RelTokenizer()

    dataset = PretrainingDataset.build(
        tokenizer=tokenizer,
        save_dir=args.save_dir,
        max_seq_len=args.l,
        num_epochs=args.e,
        midi_dataset_path=args.load_path,
    )


def _parse_finetune_dataset_args():
    argp = argparse.ArgumentParser(prog="aria finetune-dataset")
    argp.add_argument(
        "-clean_load_path",
        help="path to the clean midi_dict dataset",
    )
    argp.add_argument(
        "-noisy_load_paths",
        nargs="+",
        help="one or more paths to noisy midi_dict datasets",
    )
    argp.add_argument("-save_dir", help="path to save dataset")
    argp.add_argument("-l", help="max sequence length", type=int, default=4096)
    argp.add_argument("-e", help="num epochs", type=int, default=1)

    return argp.parse_args(sys.argv[2:])


def build_finetune_dataset(args):
    from aria.tokenizer import SeparatedAbsTokenizer
    from aria.data.datasets import FinetuningDataset

    tokenizer = SeparatedAbsTokenizer()
    dataset = FinetuningDataset.build(
        tokenizer=tokenizer,
        save_dir=args.save_dir,
        max_seq_len=args.l,
        num_epochs=args.e,
        clean_dataset_path=args.clean_load_path,
        noisy_dataset_paths=args.noisy_load_paths,
    )


def main():
    # Nested argparse inspired by - https://shorturl.at/kuKW0
    parser = argparse.ArgumentParser(usage="aria <command> [<args>]")
    parser.add_argument(
        "command",
        help="command to run",
        choices=(
            "sample",
            "midi-dataset",
            "pretrain-dataset",
            "finetune-dataset",
        ),
    )

    # parse_args defaults to [1:] for args, but you need to
    # exclude the rest of the args too, or validation will fail
    args = parser.parse_args(sys.argv[1:2])

    if not hasattr(args, "command"):
        parser.print_help()
        print("Unrecognized command")
        exit(1)
    elif args.command == "sample":
        sample(args=_parse_sample_args())
    elif args.command == "midi-dataset":
        build_midi_dataset(args=_parse_midi_dataset_args())
    elif args.command == "pretrain-dataset":
        build_pretraining_dataset(args=_parse_pretrain_dataset_args())
    elif args.command == "finetune-dataset":
        build_finetune_dataset(args=_parse_finetune_dataset_args())
    else:
        print("Unrecognized command")
        parser.print_help()
        exit(1)


if __name__ == "__main__":
    main()
