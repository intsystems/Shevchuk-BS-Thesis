from dataclasses import dataclass, asdict, field
import torch

@dataclass
class fmriAug:
    smooth_sigma: float = 1 #std of gaussian for smoothening
    noise_sigma: float = 1 #std for applying gaussian noise
    amplitude: tuple = (0.8, 1.2) #tuple (low, high), how much to scale ampl of signal
    ratio_to_mask: tuple = (0.1, 0.2) #tuple (low, high), low and hign in [0,1]
    tr_jitter: tuple = (-1, 0, 1) 
@dataclass
class eegAug:
    sigma: float = 1 #std for applying gaussian noise and pink noise
    max_time_shift: int = 1  #int number of frames to shift by
    channel_drop_prob: float = 0.2 #float in [0,1]
    ratio_of_freq_to_drop: float = 0.2 #float in [0,1]
    bandwidth_mask_prob: float = 0.2 #float in [0,1]
    amplitude: tuple = (0, 0)#tuple, low, high

@dataclass
class DataConfig:
    data_dir: str = "data/projects/EEG_FMRI/data_indi_preproc/"
    #dataset params
    eeg_sr: int = 200
    tr: float = 2.1
    eeg_win_sec: int = 2
    hrf_shifts_sec: list = field(default_factory=lambda: [4.2])
    stride_tr: int = 1
    n_eeg_channels: int = 60

    #peprocessing settings (dataset specific)
    use_parcellation: bool = False
    parcellation_suffix: str = "space-MNI152Lin_res-3mm_atlas-Schaefer2018_dens-200parcels7networks_desc-sm0_bold"
    no_parcellation_prefix:str = "func_pp_nofilt"
    lower_freq:float = 0.5
    higher_freq:float = 45
    target_eeg_freq:int = 200
    orig_fmri_res: int = 3
    target_fmri_res: int = 2

    #augmentations settings
    eeg_aug: eegAug = field(default_factory=eegAug)
    fmri_aug: fmriAug = field(default_factory=fmriAug)

    #batch sampler settings:


@dataclass
class ModelConfig:
    #models params
    n_eeg_channels: int = 64
    n_eeg_times: int = 500  # eeg_win_sec * eeg_sr
    n_roi: int = 200  # Schaefer 200 parcellation

    projector_hidden_dim: int = 256
    projector_out_dim: int = 128

@dataclass
class TrainingConfig:
    checkpoint_dir: str = "checkpoints/"
    #training params
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 5e-2
    num_epochs:int = 100
    tau: float = 0.1 #for infonce loss

    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    
    seed = 42
    num_workers = 0
    save_every = 5

@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)

if __name__ == "__main__":
    config = TrainConfig()
    print(asdict(config))