"""EEGNet + physiological MLP, with 16 features from each branch."""

import copy

import numpy as np
import torch
from torch import nn

from .signals import FEATURE_SLICES


class EEGEncoder(nn.Module):
    """Braindecode 1.3 EEGNet has no return_features flag; expose it here.

    Remove only the final classifier and flatten the learned convolution features.
    See https://braindecode.org/1.2/generated/braindecode.models.EEGNet.html
    """

    def __init__(self):
        super().__init__()
        from braindecode.models import EEGNet
        network = EEGNet(n_chans=62, n_outputs=3, n_times=500, sfreq=250,
                         F1=8, D=2, F2=16, drop_prob=.25)
        self.features = nn.Sequential(*[module for name, module in network.named_children()
                                        if name != "final_layer"], nn.Flatten())
        self.features.eval()
        with torch.no_grad():
            size = self.features(torch.zeros(1, 62, 500)).shape[1]
        self.projection = nn.Linear(size, 16)

    def forward(self, x, return_features=True):
        features = self.projection(self.features(x))
        return features


class FusionModel(nn.Module):
    def __init__(self, dropout=.2):
        super().__init__()
        self.eeg = EEGEncoder()
        self.physio = nn.Sequential(nn.Linear(25, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 16))
        self.fusion = nn.Sequential(nn.Linear(16 + 16 + 1 + 4, 32), nn.ReLU(),
                                    nn.Dropout(dropout), nn.Linear(32, 3))
        self.modality_dropout = dropout

    def forward(self, eeg, physio, time, mask):
        mask, physio = mask.clone(), physio.clone()
        if self.training:
            mask *= (torch.rand_like(mask) >= self.modality_dropout).float()
        for index, kind in enumerate(("ecg", "eda", "resp"), 1):
            physio[:, FEATURE_SLICES[kind]] *= mask[:, index:index+1]
        # Mask AFTER each encoder too: linear biases must not leak missing modalities.
        eeg_features = self.eeg(eeg * mask[:, :1, None], return_features=True) * mask[:, :1]
        physio_features = self.physio(physio) * (mask[:, 1:].sum(1, keepdim=True) > 0)
        return self.fusion(torch.cat([eeg_features, physio_features, time, mask], dim=1))


def train_model(train, validation, epochs=20, seed=7, device="cpu"):
    """Equal total loss weight per trial, with inverse trial-class frequencies."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = FusionModel().to(device)
    scaler = (np.mean(train["physio"], axis=0), np.maximum(np.std(train["physio"], axis=0), 1e-4))
    def tensors(data):
        return (torch.from_numpy(data["eeg"]),
                torch.as_tensor((data["physio"] - scaler[0]) / scaler[1], dtype=torch.float32),
                torch.as_tensor(data["time"][:, None] / 10, dtype=torch.float32),
                torch.as_tensor(data["mask"], dtype=torch.float32))
    x, vx = tensors(train), tensors(validation)
    y = torch.as_tensor(train["target"], dtype=torch.long)
    vy = torch.as_tensor(validation["target"], dtype=torch.long, device=device)
    _, first, count = np.unique(train["trial"], return_index=True, return_counts=True)
    counts = np.bincount(train["target"][first], minlength=3)
    if np.any(counts == 0):
        raise ValueError(f"All three classes needed in training; trial counts={counts.tolist()}")
    weight = torch.tensor(counts.sum() / (3 * counts), dtype=torch.float32, device=device)
    trial_sizes = dict(zip(train["trial"][first], count))
    sample_weight = torch.tensor([1 / trial_sizes[t] for t in train["trial"]], device=device)
    sample_weight *= len(sample_weight) / sample_weight.sum()
    criterion = nn.CrossEntropyLoss(weight=weight, reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    best, best_loss, history = None, float("inf"), []
    for epoch in range(epochs):
        model.train()
        for ids in np.array_split(rng.permutation(len(y)), max(1, int(np.ceil(len(y) / 64)))):
            logits = model(*(a[ids].to(device) for a in x))
            loss = (criterion(logits, y[ids].to(device)) * sample_weight[ids]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = torch.cat([model(*(a[i:i+128].to(device) for a in vx))
                                for i in range(0, len(vy), 128)])
            # Validate on trial-averaged probabilities, never count 19 windows as 19 trials.
            probabilities = logits.softmax(-1)
            losses = []
            for trial in np.unique(validation["trial"]):
                ids = np.flatnonzero(validation["trial"] == trial)
                losses.append(-torch.log(probabilities[ids].mean(0)[vy[ids[0]]].clamp_min(1e-8)))
            value = torch.stack(losses).mean().item()
        history.append({"epoch": epoch + 1, "validation_trial_log_loss": value})
        if value < best_loss:
            best_loss, best = value, copy.deepcopy(model.state_dict())
    model.load_state_dict(best)
    model.eval()
    return model, scaler, history


def fit_temperature(model, data, scaler):
    """A single positive temperature, fitted to validation trials only."""
    from scipy.optimize import minimize_scalar
    from scipy.special import softmax
    device = next(model.parameters()).device
    logits = []
    with torch.inference_mode():
        for start in range(0, len(data["target"]), 128):
            s = slice(start, start + 128)
            logits.append(model(torch.as_tensor(data["eeg"][s], device=device),
                                torch.as_tensor((data["physio"][s] - scaler[0]) / scaler[1], dtype=torch.float32, device=device),
                                torch.as_tensor(data["time"][s, None] / 10, dtype=torch.float32, device=device),
                                torch.as_tensor(data["mask"][s], dtype=torch.float32, device=device)).cpu().numpy())
    logits = np.concatenate(logits)
    groups = [np.flatnonzero(data["trial"] == t) for t in np.unique(data["trial"])]
    def objective(log_temperature):
        p = softmax(logits / np.exp(log_temperature), axis=1)
        return np.mean([-np.log(max(p[g].mean(0)[data["target"][g[0]]], 1e-10)) for g in groups])
    result = minimize_scalar(objective, bounds=(np.log(.1), np.log(10.)), method="bounded")
    return float(np.exp(result.x))
