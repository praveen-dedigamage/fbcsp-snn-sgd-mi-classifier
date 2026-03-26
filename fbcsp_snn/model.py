"""Spiking Neural Network classifier built with snnTorch."""

from typing import List, Tuple

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class SNNClassifier(nn.Module):
    """Two-layer leaky integrate-and-fire SNN with population coding at the output.

    Parameters
    ----------
    input_size:
        Number of input spike channels.
    hidden_size:
        Number of hidden LIF neurons.
    output_size:
        Number of classes.
    population_per_class:
        Number of output neurons assigned to each class.
    beta:
        Membrane potential decay factor shared by both LIF layers.
    dropout_prob:
        Dropout probability applied after each linear projection.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        population_per_class: int = 5,
        beta: float = 0.95,
        dropout_prob: float = 0.5,
    ) -> None:
        super().__init__()
        self.population_per_class = population_per_class
        self.total_outputs = output_size * population_per_class

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        self.fc2 = nn.Linear(hidden_size, self.total_outputs)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass over all time steps.

        Parameters
        ----------
        x : Tensor, shape ``(n_time_steps, batch_size, input_size)``

        Returns
        -------
        spk_out : ``(n_time_steps, batch_size, total_outputs)``
        spk_hidden : ``(n_time_steps, batch_size, hidden_size)``
        mem_out : ``(n_time_steps, batch_size, total_outputs)``
        mem_hidden : ``(n_time_steps, batch_size, hidden_size)``
        """
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk1_rec: List[torch.Tensor] = []
        mem1_rec: List[torch.Tensor] = []
        spk2_rec: List[torch.Tensor] = []
        mem2_rec: List[torch.Tensor] = []

        for t in range(x.size(0)):
            spk1, mem1 = self.lif1(self.dropout1(self.fc1(x[t])), mem1)
            spk2, mem2 = self.lif2(self.dropout2(self.fc2(spk1)), mem2)

            spk1_rec.append(spk1)
            mem1_rec.append(mem1)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return (
            torch.stack(spk2_rec),
            torch.stack(spk1_rec),
            torch.stack(mem2_rec),
            torch.stack(mem1_rec),
        )
