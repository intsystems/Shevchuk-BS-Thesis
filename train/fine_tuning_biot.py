import torch
import torch.nn as nn
import torch.optim as optim
from BIOT.model.biot import BIOTEncoder

# ==========================================
# 1. Initialize and Fully Reinitialize Embeddings
# ==========================================
def setup_full_reinit_model():
    # 1. Initialize the model with the original 16 channels to match pre-trained weights
    model = BIOTEncoder(num_channels=16, embed_dim=256, out_dim=512)

    new_channels = 64
    
    # [!] Load your pre-trained weights here
    model.load_state_dict(torch.load("../BIOT/pretrained-models/biot_pretrained_16ch.pth"))
    
    # 2. Completely overwrite the channel embedding layer for 64 channels
    model.channel_tokens = nn.Embedding(new_channels, 256)

    model.index = nn.Parameter(
            torch.LongTensor(range(new_channels)), requires_grad=False
        )
    
    # 3. Full Reinitialization: Apply a normal distribution from scratch
    # We use a small standard deviation so the initial noise doesn't overwhelm the network
    nn.init.normal_(model.channel_tokens.weight, mean=0, std=0.02)
    
    return model

# ==========================================
# 2. Phase Managers (Freezing / Unfreezing)
# ==========================================
def set_phase_1_frozen_backbone(model):
    """
    Freezes the pre-trained Transformer and Temporal extractors.
    Allows ONLY the new 64 channel embeddings and the projection head to learn.
    """
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze ONLY the new channel embeddings
    model.tokenizer.channel_embed.weight.requires_grad = True
    
    # Unfreeze the projection head (since it maps to fMRI space)
    for param in model.projection_head.parameters():
        param.requires_grad = True
        
    print("Phase 1: Backbone FROZEN. Training embeddings only.")
    return optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

def set_phase_2_unfrozen_backbone(model):
    """
    Unfreezes the entire network once the embeddings have stabilized.
    """
    for param in model.parameters():
        param.requires_grad = True
        
    print("Phase 2: Backbone UNFROZEN. Full network fine-tuning.")
    # Drop the learning rate significantly to avoid destroying pre-trained weights
    return optim.Adam(model.parameters(), lr=1e-5)

# ==========================================
# 3. The Two-Phase Training Loop
# ==========================================
def train_full_reinit_pipeline(eeg_encoder, fmri_encoder, dataloader, total_epochs=50, unfreeze_epoch=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eeg_encoder = eeg_encoder.to(device)
    fmri_encoder = fmri_encoder.to(device)
    
    triplet_loss_fn = nn.TripletMarginLoss(margin=1.0)
    
    # Start with Phase 1 (Frozen Backbone)
    optimizer = set_phase_1_frozen_backbone(eeg_encoder)
    
    for epoch in range(total_epochs):
        # ---------------------------------------------------------
        # THE SWITCH: Unfreeze the backbone at the designated epoch
        # ---------------------------------------------------------
        if epoch == unfreeze_epoch:
            optimizer = set_phase_2_unfrozen_backbone(eeg_encoder)
            # You would also add your fMRI encoder parameters to this optimizer
            # if you are co-training it, e.g., optim.Adam(list(eeg.params) + list(fmri.params))
            
        eeg_encoder.train()
        fmri_encoder.train()
        epoch_loss = 0.0
        
        for batch_idx, (anchor_fmri, pos_eeg, neg_eeg) in enumerate(dataloader):
            anchor_fmri, pos_eeg, neg_eeg = anchor_fmri.to(device), pos_eeg.to(device), neg_eeg.to(device)
            
            optimizer.zero_grad()
            
            # Forward passes
            anchor_out = fmri_encoder(anchor_fmri)
            positive_out = eeg_encoder(pos_eeg)
            negative_out = eeg_encoder(neg_eeg)
            
            # Loss and Backprop
            loss = triplet_loss_fn(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{total_epochs}] - Loss: {epoch_loss/len(dataloader):.4f}")

# Execute
# eeg_model = setup_full_reinit_model()
# train_full_reinit_pipeline(eeg_model, fmri_model, my_dataloader, total_epochs=50, unfreeze_epoch=5)