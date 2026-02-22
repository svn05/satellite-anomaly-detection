"""LSTM-Autoencoder for satellite telemetry anomaly detection.

Architecture: Encoder (LSTM) -> Bottleneck -> Decoder (LSTM) -> Reconstruction
Anomalies are detected via reconstruction error thresholding.
"""

import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    """LSTM encoder that compresses input sequences to a latent representation."""

    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_latent = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_dim)
        Returns:
            latent: (batch_size, latent_dim) — compressed representation
            hidden: LSTM hidden state tuple
        """
        lstm_out, hidden = self.lstm(x)
        # Use the last time step's output
        last_output = lstm_out[:, -1, :]
        latent = self.fc_latent(last_output)
        return latent, hidden


class LSTMDecoder(nn.Module):
    """LSTM decoder that reconstructs sequences from latent representation."""

    def __init__(self, latent_dim, hidden_dim, output_dim, seq_len, num_layers=2, dropout=0.2):
        super().__init__()
        self.seq_len = seq_len
        self.fc_expand = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_output = nn.Linear(hidden_dim, output_dim)

    def forward(self, latent):
        """
        Args:
            latent: (batch_size, latent_dim)
        Returns:
            reconstruction: (batch_size, seq_len, output_dim)
        """
        expanded = self.fc_expand(latent)
        # Repeat latent across time steps
        repeated = expanded.unsqueeze(1).repeat(1, self.seq_len, 1)
        lstm_out, _ = self.lstm(repeated)
        reconstruction = self.fc_output(lstm_out)
        return reconstruction


class LSTMAutoencoder(nn.Module):
    """LSTM-Autoencoder for multivariate time series anomaly detection.

    The model learns to reconstruct normal telemetry patterns. Anomalies
    are detected when reconstruction error exceeds a learned threshold.

    Args:
        input_dim: Number of input features (telemetry channels).
        hidden_dim: LSTM hidden state dimension.
        latent_dim: Bottleneck dimension.
        seq_len: Input sequence length (window size).
        num_layers: Number of LSTM layers.
        dropout: Dropout rate.
    """

    def __init__(self, input_dim=25, hidden_dim=128, latent_dim=64, seq_len=100,
                 num_layers=2, dropout=0.2):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim, latent_dim, num_layers, dropout)
        self.decoder = LSTMDecoder(latent_dim, hidden_dim, input_dim, seq_len, num_layers, dropout)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_dim) — input telemetry window
        Returns:
            reconstruction: (batch_size, seq_len, input_dim) — reconstructed window
            latent: (batch_size, latent_dim) — latent representation
        """
        latent, _ = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent

    def get_reconstruction_error(self, x):
        """Compute per-sample reconstruction error (MSE).

        Args:
            x: (batch_size, seq_len, input_dim)
        Returns:
            errors: (batch_size,) — mean reconstruction error per sample
        """
        self.eval()
        with torch.no_grad():
            reconstruction, _ = self.forward(x)
            errors = torch.mean((x - reconstruction) ** 2, dim=(1, 2))
        return errors


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    model = LSTMAutoencoder(input_dim=25, hidden_dim=128, latent_dim=64, seq_len=100)
    print(f"Model parameters: {count_parameters(model):,}")

    x = torch.randn(8, 100, 25)
    recon, latent = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {recon.shape}")
    print(f"Latent shape: {latent.shape}")
    print(f"Reconstruction error: {model.get_reconstruction_error(x).mean():.4f}")
