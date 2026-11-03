
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaBlock(nn.Module):
    """
    Simplified Mamba-1 selective state-space block.

    Input:
        x: (batch, sequence_length, d_model)

    Output:
        y: (batch, sequence_length, d_model)

    This module is a sequence mixer. Residual connections and
    normalization are handled by the surrounding decoder layer.
    """

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank=None,
        dt_min=0.001,
        dt_max=0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        # Expanded hidden dimension
        self.d_inner = d_model * expand

        # Rank used to generate input-dependent time steps
        if dt_rank is None:
            dt_rank = math.ceil(d_model / 16)

        self.dt_rank = dt_rank

        # --------------------------------------------------
        # 1. Input projection
        # --------------------------------------------------

        # Split projected input into x and gating branches
        self.in_proj = nn.Linear(
            d_model,
            2 * self.d_inner,
            bias=False,
        )

        # --------------------------------------------------
        # 2. Causal depthwise convolution
        # --------------------------------------------------

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=0,
            bias=True,
        )

        # --------------------------------------------------
        # 3. Input-dependent SSM parameter projection
        # --------------------------------------------------

        # Produces:
        # dt: dt_rank
        # B:  d_state
        # C:  d_state

        self.x_proj = nn.Linear(
            self.d_inner,
            self.dt_rank + 2 * d_state,
            bias=False,
        )

        # Expand dt from dt_rank to d_inner
        self.dt_proj = nn.Linear(
            self.dt_rank,
            self.d_inner,
            bias=True,
        )

        # --------------------------------------------------
        # 4. State-space parameters
        # --------------------------------------------------

        # A = -exp(A_log)
        # Initialize A using log of positive state indices.
        A = torch.arange(
            1,
            d_state + 1,
            dtype=torch.float32,
        )

        A = A.unsqueeze(0).repeat(self.d_inner, 1)

        self.A_log = nn.Parameter(torch.log(A))

        # Skip connection parameter D
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # --------------------------------------------------
        # 5. Output projection
        # --------------------------------------------------

        self.out_proj = nn.Linear(
            self.d_inner,
            d_model,
            bias=False,
        )

        # --------------------------------------------------
        # 6. Initialize dt projection
        # --------------------------------------------------

        dt_init_std = self.dt_rank ** -0.5

        nn.init.uniform_(
            self.dt_proj.weight,
            -dt_init_std,
            dt_init_std,
        )

        # Initialize time steps in [dt_min, dt_max]
        dt = torch.exp(
            torch.rand(self.d_inner)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )

        # Inverse softplus initialization
        inv_dt = dt + torch.log(-torch.expm1(-dt))

        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    # --------------------------------------------------
    # Selective scan
    # --------------------------------------------------

    def selective_scan(self, u, delta, B, C):
        """
        Simple sequential selective state-space recurrence.

        u:     (batch, seq_len, d_inner)
        delta: (batch, seq_len, d_inner)
        B:     (batch, seq_len, d_state)
        C:     (batch, seq_len, d_state)

        Returns:
            output: (batch, seq_len, d_inner)
        """

        batch_size, seq_len, d_inner = u.shape

        # Continuous-time state matrix
        A = -torch.exp(self.A_log.float())

        # Initial recurrent state
        state = torch.zeros(
            batch_size,
            d_inner,
            self.d_state,
            device=u.device,
            dtype=torch.float32,
        )

        outputs = []

        # Keep recurrence calculations in float32
        u_float = u.float()
        delta_float = delta.float()
        B_float = B.float()
        C_float = C.float()

        for t in range(seq_len):

            # Current input and time step
            u_t = u_float[:, t, :]
            delta_t = delta_float[:, t, :]

            B_t = B_float[:, t, :]
            C_t = C_float[:, t, :]

            # Discretize A
            delta_A = torch.exp(
                delta_t.unsqueeze(-1) * A.unsqueeze(0)
            )

            # Input contribution
            delta_B_u = (
                delta_t.unsqueeze(-1)
                * B_t.unsqueeze(1)
                * u_t.unsqueeze(-1)
            )

            # Update hidden state
            state = delta_A * state + delta_B_u

            # Read state using C
            y_t = torch.sum(
                state * C_t.unsqueeze(1),
                dim=-1,
            )

            # Skip connection
            y_t = y_t + self.D.float() * u_t

            outputs.append(y_t)

        output = torch.stack(outputs, dim=1)

        return output.to(dtype=u.dtype)

    # --------------------------------------------------
    # Forward pass
    # --------------------------------------------------

    def forward(self, x):

        batch_size, seq_len, _ = x.shape

        # Input projection
        xz = self.in_proj(x)

        x_branch, z_branch = xz.chunk(2, dim=-1)

        # Causal convolution
        x_conv = x_branch.transpose(1, 2)

        # Left padding ensures no future information is used
        x_conv = F.pad(
            x_conv,
            (self.d_conv - 1, 0),
        )

        x_conv = self.conv1d(x_conv)

        x_conv = x_conv.transpose(1, 2)

        # Nonlinear activation
        x_conv = F.silu(x_conv)

        # Generate input-dependent dt, B and C
        x_ssm = self.x_proj(x_conv)

        dt, B, C = torch.split(
            x_ssm,
            [
                self.dt_rank,
                self.d_state,
                self.d_state,
            ],
            dim=-1,
        )

        # Expand dt to the inner dimension
        delta = F.softplus(
            self.dt_proj(dt)
        )

        # Selective state-space scan
        y = self.selective_scan(
            x_conv,
            delta,
            B,
            C,
        )

        # Gating branch
        y = y * F.silu(z_branch)

        # Project back to model dimension
        output = self.out_proj(y)

        return output
