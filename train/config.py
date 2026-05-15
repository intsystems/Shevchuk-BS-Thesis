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
    dataset_name: str = "natview"
    eeg_sr: int = 200
    tr: float = 2.1 #sampling rate of fmri, in sec
    eeg_win_sec: int = 15 #window size, for eeg and fmri, in sec
    hrf_shifts_sec: list = field(default_factory=lambda: [4.2])
    stride_tr: int = 1 #shift inside one activity, in trs
    n_eeg_channels: int = 60
    ch_names: list = field(default_factory=lambda: ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz', 'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'TP9', 'TP10', 'POz', 'F1', 'F2', 'C1', 'C2', 'P1', 'P2', 'AF3', 'AF4', 'FC3', 'CP3', 'CP4', 'PO3', 'PO4', 'F5', 'F6', 'C5', 'C6', 'P5', 'P6', 'AF7', 'AF8', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8', 'Fpz', 'CPz'])
    start_sub: int = 1
    end_sub: int = 22
    base_url: str = "https://fcp-indi.s3.amazonaws.com/data/Projects/NATVIEW_EEGFMRI/preproc_data_gz"

    #peprocessing settings (dataset specific)
    use_parcellation: bool = False
    parcellation_suffix: str = "space-MNI152Lin_res-3mm_atlas-Schaefer2018_dens-200parcels7networks_desc-sm0_bold"
    no_parcellation_prefix:str = "func_pp_nofilt_sm0"
    lower_freq:float = 0.5
    higher_freq:float = 45
    target_eeg_freq:int = 200
    target_voxel_size:tuple = (2, 2, 2) #in mm
    excluded_activities: list = field(default_factory=lambda: ["task-checker", "task-rest", "task-peer"])
    fill_zeroback: bool = False
    output_h5: str = "dataset.h5"
    channels_json: str = "eeg_channels.json"

    #augmentations settings
    eeg_aug: eegAug = field(default_factory=eegAug)
    fmri_aug: fmriAug = field(default_factory=fmriAug)

    #batch sampler settings:
    num_timestamps: int = 8 #number of timesteps inside activity used in batch
    num_subjects: int = 16 #number of subject per activity
    margin_tr: int = 15 #minimal number of fmri frames between two neighbouring timesteps (to cancel high time correlation) 

@dataclass
class ModelConfig:
    #models params
    n_eeg_channels: int = 64
    n_eeg_times: int = 500  # eeg_win_sec * eeg_sr
    n_roi: int = 200  # Schaefer 200 parcellation

    #Labram output is [B, 200], [CLS] token embedding
    Labram_out_dim: int = 200
    labram_pretrained: bool = True

    #Neurostorm output is (B, 288, 2, 2, 2, T)
    Neurostorm_ckpt: str = "neurostorm.ckpt"
    Neurostorm_out_dim: int = 288

    projector_hidden_dim: int = 256
    projector_out_dim: int = 128

@dataclass
class TrainingConfig:
    checkpoint_dir: str = "checkpoints/"
    #training params
    batch_size: int = 64
    backbone_lr: float = 1e-4
    proj_lr: float = 1e-5
    weight_decay: float = 5e-2
    num_epochs:int = 100
    tau: float = 0.1 #for infonce loss

    freeze_backbone: bool = True

    #lora params
    lora_rank: int = 4
    lora_alpha: float = 4
    lora_dropout: float = 0.05

    #data
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    seed: int = 42
    num_workers: int = 0
    save_every:int = 5

@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)

if __name__ == "__main__":
    config = TrainConfig()
    print(asdict(config))