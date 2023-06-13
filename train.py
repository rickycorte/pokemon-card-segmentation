#
# Train script for baseline model
#
import pytorch_lightning as pl
from torchvision import transforms

import settings
from datasets import PkmCardSegmentationDataModule

from models import SimoidSegmentationModule, UNetBaseline


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

torch_model = UNetBaseline(in_depth=settings.input_channels, out_depth=1, depth_scale=settings.baseline_model_scale)


model = SimoidSegmentationModule(torch_model, lr=settings.learn_rate)


trainer = pl.Trainer(max_epochs=settings.max_epochs)
trainer.fit(model, datamodule)