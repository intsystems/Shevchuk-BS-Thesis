import lightning as L
from src.eeg_encoder import EEGAugmentor
from src.fmri_encoder import FmriAugmentor
from train.config import TrainConfig

class ContrastiveModel(L.LightningModule):
    def __init__(self, eeg_encoder, fmri_encoder, config: TrainConfig):
        super().__init__();

        self.eeg_encoder = eeg_encoder
        self.fmri_encoder = fmri_encoder

        self.eeg_augmentor = EEGAugmentor(config)
        self.fmri_encoder = EEGAugmentor(config)

    def forward(self, x):
        
    def on_after_batch_transfer(self, batch, dataloader_idx):
        
    def _calculate_loss(self, x):

    def training_step(self, batch, batch_idx):

        
    def configure_optimizers(self):
        return super().configure_optimizers()