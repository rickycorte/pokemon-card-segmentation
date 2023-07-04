
import os

# path settings
data_root = "data"

dataset_folder = "data/pkm"
raw_dataset_folder = "data/raw"

output_root = "output"

# global settings
card_size = (512, 352)

# data preprocessing settings
min_mask_occupaion_percent = 0.20
min_contour_area = 1000

# pokemon portrait location (top left corner) and size
pkm_loc_tl = [60, 35]
pkm_loc_sz = [185, card_size[1] - 2 * pkm_loc_tl[1]]

noise_tolerance_pixels = 100
noise_loc_tl = [300, 35]
noise_loc_sz = [180, card_size[1] - 2 * pkm_loc_tl[1]]


# training settings
batch_size = 16
max_epochs = 150
input_channels = 3
use_noisy_labels = False
learn_rate =  1e-4  
baseline_model_scale = 2
timmunet_decoder_scale = 1

# prediction settings
baseline_checkpoint_path = "checkpoints/baseline_150.ckpt"
timmunet_checkpoint_path = "checkpoints/timmunet_eff3_0819.ckpt"

################################################################################

os.makedirs(output_root, exist_ok=True)