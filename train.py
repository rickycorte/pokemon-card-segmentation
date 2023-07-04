#
# Train script for baseline model
#
import pytorch_lightning as pl
from torchvision import transforms

import settings
from datasets import PkmCardSegmentationDataModule

from models import SimoidSegmentationModule, UnetTimm


# data agumentation
data_aug = transforms.Compose([
    transforms.RandomRotation((-30, 30)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
])


datamodule = PkmCardSegmentationDataModule(
    settings.dataset_folder,
    batch_size=settings.batch_size,
    transform=data_aug,
    use_noisy=settings.use_noisy_labels
)

# replace this line with the model you want to train
torch_model = UnetTimm(
    out_depth=1,
    backbone_name="efficientnet_b3",
    pretrained=True,
    decoder_scale=settings.timmunet_decoder_scale
)


model = SimoidSegmentationModule(torch_model, lr=settings.learn_rate)


trainer = pl.Trainer(max_epochs=settings.max_epochs)
trainer.fit(model, datamodule)