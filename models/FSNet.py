"""
FSNet Model Wrapper
Wraps the FSNet encoder from layers/ts2vec for use as a forecasting model
"""

import torch
from torch import nn
from layers.ts2vec.fsnet import GlobalLocalMultiscaleTSEncoder


class Model(nn.Module):
    """
    FSNet: Fast and Slow Network for Time Series Forecasting
    Uses GlobalLocalMultiscaleTSEncoder from layers/ts2vec
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in

        # FSNet encoder
        self.encoder = GlobalLocalMultiscaleTSEncoder(
            input_dims=configs.enc_in,
            output_dims=configs.d_model if hasattr(configs, 'd_model') else 64,
            hidden_dims=configs.d_model if hasattr(configs, 'd_model') else 64,
            depth=configs.e_layers if hasattr(configs, 'e_layers') else 10,
        )

        # Projection head for forecasting
        self.projection = nn.Linear(
            configs.d_model if hasattr(configs, 'd_model') else 64,
            configs.pred_len * configs.enc_in
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        Args:
            x_enc: [batch, seq_len, enc_in]
            x_mark_enc: time features (optional)

        Returns:
            forecast: [batch, pred_len, enc_in]
        """
        # Encode the input sequence
        # FSNet encoder expects [batch, seq_len, features]
        enc_out = self.encoder(x_enc)  # [batch, seq_len, d_model]

        # Take the last timestep encoding
        last_enc = enc_out[:, -1, :]  # [batch, d_model]

        # Project to forecast
        forecast = self.projection(last_enc)  # [batch, pred_len * enc_in]

        # Reshape to [batch, pred_len, enc_in]
        forecast = forecast.reshape(-1, self.pred_len, self.enc_in)

        return forecast
