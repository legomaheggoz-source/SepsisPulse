"""
Temporal Fusion Transformer Lite (TFT-Lite) Architecture.

This module implements a lightweight version of the Temporal Fusion Transformer
optimized for deployment on resource-constrained environments (2GB RAM).

Key simplifications from full TFT:
    - Reduced hidden size (32 vs 256)
    - Single LSTM layer (vs 2)
    - 2 attention heads (vs 4)
    - Simplified variable selection network
    - ~500K parameters vs ~10M in original

Reference:
    Lim, B., et al. "Temporal Fusion Transformers for Interpretable
    Multi-horizon Time Series Forecasting." arXiv:1912.09363 (2019).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GatedLinearUnit(nn.Module):
    """
    Gated Linear Unit (GLU) for feature gating.

    Applies element-wise gating: GLU(x) = sigmoid(Wx + b) * (Vx + c)
    """

    def __init__(self, input_size: int, output_size: int, dropout: float = 0.1):
        """
        Initialize the GLU layer.

        Args:
            input_size: Dimension of input features.
            output_size: Dimension of output features.
            dropout: Dropout probability.
        """
        super().__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.fc2 = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GLU.

        Args:
            x: Input tensor of shape (batch, *, input_size).

        Returns:
            Output tensor of shape (batch, *, output_size).
        """
        sig = self.sigmoid(self.fc1(x))
        x = sig * self.fc2(x)
        return self.dropout(x)


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) for non-linear processing with skip connections.

    Architecture: Linear -> ELU -> Linear -> GLU + Residual + LayerNorm
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.1,
        context_size: Optional[int] = None
    ):
        """
        Initialize the GRN layer.

        Args:
            input_size: Dimension of input features.
            hidden_size: Dimension of hidden layer.
            output_size: Dimension of output features.
            dropout: Dropout probability.
            context_size: Optional dimension of context vector.
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        # Main pathway
        self.fc1 = nn.Linear(input_size, hidden_size)

        # Optional context integration
        self.context_size = context_size
        if context_size is not None:
            self.fc_context = nn.Linear(context_size, hidden_size, bias=False)

        self.fc2 = nn.Linear(hidden_size, output_size)
        self.glu = GatedLinearUnit(output_size, output_size, dropout)

        # Skip connection projection if dimensions differ
        if input_size != output_size:
            self.skip_layer = nn.Linear(input_size, output_size)
        else:
            self.skip_layer = None

        self.layer_norm = nn.LayerNorm(output_size)
        self.elu = nn.ELU()

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through GRN.

        Args:
            x: Input tensor of shape (batch, *, input_size).
            context: Optional context tensor of shape (batch, context_size).

        Returns:
            Output tensor of shape (batch, *, output_size).
        """
        # Skip connection
        if self.skip_layer is not None:
            skip = self.skip_layer(x)
        else:
            skip = x

        # Main pathway
        hidden = self.fc1(x)

        # Add context if provided
        if context is not None and self.context_size is not None:
            # Expand context to match x dimensions
            context_proj = self.fc_context(context)
            if x.dim() == 3:
                context_proj = context_proj.unsqueeze(1).expand(-1, x.size(1), -1)
            hidden = hidden + context_proj

        hidden = self.elu(hidden)
        hidden = self.fc2(hidden)
        hidden = self.glu(hidden)

        # Residual connection + layer norm
        return self.layer_norm(hidden + skip)


class VariableSelectionNetwork(nn.Module):
    """
    Simplified Variable Selection Network for feature selection.

    Uses GRN-based gating to weight input features based on their importance.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float = 0.1
    ):
        """
        Initialize the Variable Selection Network.

        Args:
            input_size: Number of input features.
            hidden_size: Hidden dimension size.
            dropout: Dropout probability.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Feature embedding
        self.feature_embedding = nn.Linear(input_size, hidden_size)

        # Variable selection weights via softmax gating
        self.flattened_grn = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=input_size,
            dropout=dropout
        )

        # Transform after selection
        self.transform = GatedResidualNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Variable Selection Network.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size).

        Returns:
            Tuple of:
                - Selected features of shape (batch, seq_len, hidden_size).
                - Variable weights of shape (batch, seq_len, input_size).
        """
        # Embed features
        embedded = self.feature_embedding(x)

        # Compute variable selection weights
        weights = self.flattened_grn(embedded)
        weights = F.softmax(weights, dim=-1)

        # Apply weights to input
        weighted_input = x * weights

        # Transform selected features
        output = self.transform(weighted_input)

        return output, weights


class InterpretableMultiHeadAttention(nn.Module):
    """
    Simplified Multi-Head Attention for interpretability.

    Returns attention weights along with output for interpretability.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        """
        Initialize the attention module.

        Args:
            hidden_size: Dimension of hidden states.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        assert hidden_size % num_heads == 0, \
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"

        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)

        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through attention.

        Args:
            query: Query tensor of shape (batch, seq_len, hidden_size).
            key: Key tensor of shape (batch, seq_len, hidden_size).
            value: Value tensor of shape (batch, seq_len, hidden_size).
            mask: Optional attention mask.

        Returns:
            Tuple of:
                - Output tensor of shape (batch, seq_len, hidden_size).
                - Attention weights of shape (batch, num_heads, seq_len, seq_len).
        """
        batch_size, seq_len, _ = query.size()

        # Project and reshape to (batch, num_heads, seq_len, head_dim)
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        output = self.out_proj(attn_output)

        return output, attn_weights


class TFTLite(nn.Module):
    """
    Temporal Fusion Transformer Lite for sepsis prediction.

    A lightweight implementation of TFT optimized for resource-constrained
    environments. Designed for 2GB RAM with ~500K parameters.

    Architecture:
        1. Variable Selection Network - Feature importance learning
        2. LSTM Encoder - Temporal pattern extraction
        3. Multi-Head Attention - Long-range dependency modeling
        4. Output Projection - Binary classification

    Attributes:
        input_size: Number of input features (default: 41).
        hidden_size: Hidden dimension size (default: 32).
        lstm_layers: Number of LSTM layers (default: 1).
        attention_heads: Number of attention heads (default: 2).
        output_size: Number of output classes (default: 1 for binary).
        dropout: Dropout probability (default: 0.1).
        max_seq_length: Maximum sequence length (default: 24).

    Example:
        >>> model = TFTLite(input_size=41, hidden_size=32)
        >>> x = torch.randn(16, 24, 41)  # (batch, seq_len, features)
        >>> output = model(x)
        >>> print(output.shape)  # torch.Size([16, 1])
    """

    def __init__(
        self,
        input_size: int = 41,
        hidden_size: int = 32,
        lstm_layers: int = 1,
        attention_heads: int = 2,
        output_size: int = 1,
        dropout: float = 0.1,
        max_seq_length: int = 24
    ):
        """
        Initialize the TFT-Lite model.

        Args:
            input_size: Number of input features.
            hidden_size: Hidden dimension size.
            lstm_layers: Number of LSTM layers.
            attention_heads: Number of attention heads.
            output_size: Number of output units.
            dropout: Dropout probability.
            max_seq_length: Maximum sequence length.
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.attention_heads = attention_heads
        self.output_size = output_size
        self.dropout_rate = dropout
        self.max_seq_length = max_seq_length

        # Variable Selection Network
        self.variable_selection = VariableSelectionNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropout
        )

        # LSTM Encoder
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=False
        )

        # Post-LSTM processing
        self.post_lstm_gate = GatedLinearUnit(hidden_size, hidden_size, dropout)
        self.post_lstm_norm = nn.LayerNorm(hidden_size)

        # Multi-Head Attention
        self.attention = InterpretableMultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=attention_heads,
            dropout=dropout
        )

        # Post-attention processing
        self.post_attn_gate = GatedLinearUnit(hidden_size, hidden_size, dropout)
        self.post_attn_norm = nn.LayerNorm(hidden_size)

        # Output network
        self.output_grn = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout
        )
        self.output_projection = nn.Linear(hidden_size, output_size)

        # Store last attention weights for interpretability
        self._last_attention_weights = None
        self._last_variable_weights = None

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights using Xavier initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through TFT-Lite.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size).

        Returns:
            Output tensor of shape (batch, output_size).
        """
        batch_size, seq_len, _ = x.size()

        # Variable Selection
        selected, var_weights = self.variable_selection(x)
        self._last_variable_weights = var_weights.detach()

        # LSTM Encoding
        lstm_out, _ = self.lstm(selected)

        # Post-LSTM gating with residual
        gated_lstm = self.post_lstm_gate(lstm_out)
        lstm_out = self.post_lstm_norm(gated_lstm + selected)

        # Self-Attention
        attn_out, attn_weights = self.attention(
            query=lstm_out,
            key=lstm_out,
            value=lstm_out
        )
        self._last_attention_weights = attn_weights.detach()

        # Post-attention gating with residual
        gated_attn = self.post_attn_gate(attn_out)
        attn_out = self.post_attn_norm(gated_attn + lstm_out)

        # Take the last timestep for prediction
        final_hidden = attn_out[:, -1, :]

        # Output network
        output = self.output_grn(final_hidden)
        output = self.output_projection(output)

        return output

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """
        Get the attention weights from the last forward pass.

        Returns:
            Attention weights of shape (batch, num_heads, seq_len, seq_len),
            or None if no forward pass has been performed.
        """
        return self._last_attention_weights

    def get_variable_weights(self) -> Optional[torch.Tensor]:
        """
        Get the variable selection weights from the last forward pass.

        Returns:
            Variable weights of shape (batch, seq_len, input_size),
            or None if no forward pass has been performed.
        """
        return self._last_variable_weights

    def count_parameters(self) -> int:
        """
        Count the total number of trainable parameters.

        Returns:
            Number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        """Return string representation of the model."""
        param_count = self.count_parameters()
        return (
            f"TFTLite(\n"
            f"  input_size={self.input_size},\n"
            f"  hidden_size={self.hidden_size},\n"
            f"  lstm_layers={self.lstm_layers},\n"
            f"  attention_heads={self.attention_heads},\n"
            f"  output_size={self.output_size},\n"
            f"  dropout={self.dropout_rate},\n"
            f"  max_seq_length={self.max_seq_length},\n"
            f"  trainable_params={param_count:,}\n"
            f")"
        )
