#
# Unfortunatelly, at the moment no segmentation dataset is available for pokemon
# trading cards.
# 
# This is a preprocessing script used to generate better lables that can used to train
# a segmentation model from foil effects scrapped from internet. Most of the cards are
# bad and unsable but the 
#

import os

import numpy as np
import cv2
from tqdm import tqdm

import settings


raw_card_path = os.path.join(settings.raw_dataset_folder, "images")
raw_mask_path = os.path.join(settings.raw_dataset_folder, "masks")


clean_card_path = os.path.join(settings.dataset_folder, "images")
clean_mask_path = os.path.join(settings.dataset_folder, "masks")

noisy_card_path = os.path.join(settings.dataset_folder, "noisy/masks")
noisy_mask_path = os.path.join(settings.dataset_folder, "noisy/masks")

nolabel_path = os.path.join(settings.dataset_folder, "nolabel")

################################################################################
# Process data

# create folders if missing
os.makedirs(clean_card_path, exist_ok=True)
os.makedirs(clean_mask_path, exist_ok=True)

os.makedirs(noisy_card_path, exist_ok=True)
os.makedirs(noisy_mask_path, exist_ok=True)

os.makedirs(nolabel_path, exist_ok=True)

# get raw files
card_files = sorted([os.path.join(raw_card_path, f) for f in os.listdir(raw_card_path)])
mask_files = sorted([os.path.join(raw_mask_path, f) for f in os.listdir(raw_mask_path)])


# process
sample_id = 0

cnt_clean = 0; cnt_noisy = 0; cnt_bad = 0;

for imp, mp in tqdm(zip(card_files, mask_files), unit="img", total=len(card_files)):
    card = cv2.cvtColor(cv2.imread(imp), cv2.COLOR_BGR2RGB)
    raw_mask =  cv2.cvtColor(cv2.imread(mp), cv2.COLOR_BGR2RGB)

    # resize
    card = cv2.resize(card, settings.card_size[::-1])
    raw_mask = cv2.resize(raw_mask, settings.card_size[::-1])

    # dilate non zero pixels to try to create a mask without too many holes
    kernel = np.ones((8, 8), np.uint8)
    raw_mask = cv2.dilate(raw_mask, kernel, iterations=1)
    raw_mask = cv2.erode(raw_mask, kernel)

    raw_mask[raw_mask.sum(-1) == 255 * 3] = (0,0,0) # convert white to black

    # check if the label is decent enough to be used
    roi = raw_mask[
        settings.pkm_loc_tl[0]: settings.pkm_loc_tl[0] + settings.pkm_loc_sz[0],
        settings.pkm_loc_tl[1]: settings.pkm_loc_tl[1] + settings.pkm_loc_sz[1],
        :
        ].sum(-1)

    invalid_occupation = (roi != 0).sum() / (settings.pkm_loc_sz[0] * settings.pkm_loc_sz[1])

    refined_mask = np.zeros((*raw_mask.shape[:2], 3))

    if invalid_occupation > 1 - settings.min_mask_occupaion_percent:
        cv2.imwrite(os.path.join(nolabel_path, f"{sample_id}.jpg"), cv2.cvtColor(card, cv2.COLOR_RGB2BGR))
        cnt_bad += 1
    else:
        # check of noisy labels
        noisy = raw_mask[
            settings.noise_loc_tl[0]: settings.noise_loc_tl[0] + settings.noise_loc_sz[0],
            settings.noise_loc_tl[1]: settings.noise_loc_tl[1] + settings.noise_loc_sz[1],
            :
        ].sum() > settings.noise_tolerance_pixels

        # find external contour
        thresh = (roi == 0).astype(np.uint8)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros((*roi.shape[:2],3))
        for cnt in contours:
            if cv2.contourArea(cnt) < settings.min_contour_area:
                continue

            cnt = cv2.approxPolyDP(cnt, 0.1, True)

            cv2.drawContours(filled,[cnt],0,255,-1)

        refined_mask[
            settings.pkm_loc_tl[0]: settings.pkm_loc_tl[0] + settings.pkm_loc_sz[0],
            settings.pkm_loc_tl[1]: settings.pkm_loc_tl[1] + settings.pkm_loc_sz[1],
            :
            ][filled.sum(-1)!= 0] = (255, 255, 255)

        if noisy:
            i_path = noisy_card_path
            m_path = noisy_mask_path 
            cnt_noisy += 1
        else:
            i_path = clean_card_path
            m_path = clean_mask_path
            cnt_clean += 1

        cv2.imwrite(os.path.join(i_path, f"{sample_id}.jpg"), cv2.cvtColor(card, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(m_path, f"{sample_id}.jpg"), refined_mask)


    sample_id += 1


print(f"Done! Total: {len(card_files)} - {cnt_clean} clean - {cnt_clean} noisy - {cnt_bad} unusable")