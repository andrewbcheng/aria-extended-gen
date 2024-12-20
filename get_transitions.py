from aria.tokenizer import AbsTokenizer
from aria.data.midi import MidiDict
import os

form_folder = input("enter form folder: ")
sample_num = input("enter sample num: ")

with open(f"/project/jonmay_1426/spangher/old-dir/music-form-structure-modeling/aria-extended-gen/synth_data/{form_folder}_samples/samples_{sample_num}/1_style.txt", 'r') as file:
    content = file.read()

save_dir = os.path.join(os.path.dirname(__file__), f"synth_data/transition_samples/{form_folder}/sample_{sample_num}")
os.makedirs(save_dir, exist_ok=True)

tokenizer = AbsTokenizer()
midi_dict = MidiDict.from_midi(f"/project/jonmay_1426/spangher/old-dir/music-form-structure-modeling/aria-extended-gen/synth_data/{form_folder}_samples/samples_{sample_num}/1_midi.mid")
seq = tokenizer.tokenize(midi_dict)

print("seq: ", seq)

#get metadata, <S> ... <S>
metadata = []
in_metadata = False
for i in range(0, len(seq)):
    if not in_metadata and seq[i] == "<S>":
        in_metadata = True
    elif in_metadata and seq[i] == "<S>":
        in_metadata = False
    if in_metadata:
        metadata.append(seq[i])

print("metadata: ", metadata)

#get transition points
transition_pts = []
for i in range(1, len(content)):
    if content[i] != content[i-1]:
        transition_pts.append(i)

print("transitions at tokens ", transition_pts)

for i in transition_pts:
    start_idx = max(i - 50, 0)
    end_idx = min(i + 50, len(seq))
    #make sure seq starts with piano tok and ends with dur tok
    while seq[start_idx][0] != 'piano':
        start_idx += 1
    while seq[end_idx][0] != 'dur':
        end_idx -= 1
    transition_area_dict = tokenizer.detokenize(metadata + seq[start_idx : end_idx + 1])
    print("seq for ", i, ":", metadata + seq[start_idx : end_idx + 1])
    transition_area_dict.to_midi().save(f"{save_dir}/tok{i}.mid")

print("results saved to ", save_dir)