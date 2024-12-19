from aria.tokenizer import AbsTokenizer
from aria.data.midi import MidiDict
import os

form_folder = input("enter form folder: ")
sample_num = input("enter sample num: ")

tokenizer = AbsTokenizer()
midi_dict = MidiDict.from_midi(f"/project/jonmay_1426/spangher/old-dir/music-form-structure-modeling/aria-extended-gen/synth_data/{form_folder}_samples/{sample_num}_midi.mid")
seq = tokenizer.tokenize(midi_dict)

with open(f"/project/jonmay_1426/spangher/old-dir/music-form-structure-modeling/aria-extended-gen/synth_data/{form_folder}_samples/{sample_num}_style.txt", 'r') as file:
    content = file.read()

transition_pts = []
for i in range(1, len(content)):
    if content[i] != content[i-1]:
        transition_pts.append(i)

print("transitions at tokens ", transition_pts)

save_dir = os.path.join(os.path.dirname(__file__), f"synth_data/transition_samples/{form_folder}/sample_{sample_num}")
os.makedirs(save_dir, exist_ok=True)

for i in transition_pts:
    start_idx = max(i - 50, 0)
    end_idx = min(i + 50, len(seq))
    transition_area = seq[start_idx:end_idx].to_midi()
    transition_area.save(f"{save_dir}/tok{i}.mid")

print("results saved to ", save_dir)