"""
model.py
--------
Defines the EZSD-CP model architecture for zero-shot stance detection.
Components:
  - PromptGenerator  : lightweight linear prompt head
  - gMLP             : gated MLP block
  - EZSD_CP_Model    : full BERT + prompt + gMLP classifier
"""

import torch
import torch.nn as nn
from transformers import BertModel


class PromptGenerator(nn.Module):
    """
    Generates a soft prompt embedding from the CLS token representation.

    Args:
        input_dim  (int): Input feature dimension (768 for BERT-base).
        hidden_dim (int): Output prompt embedding dimension.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super(PromptGenerator, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class gMLP(nn.Module):
    """
    Gated MLP block with GELU activation.

    Args:
        input_dim  (int): Input feature dimension.
        output_dim (int): Output feature dimension.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super(gMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        return self.fc2(x)


class EZSD_CP_Model(nn.Module):
    """
    EZSD-CP: Chain-of-thought Prompted Zero-Shot Stance Detection model.

    Architecture:
        1. BERT encoder → CLS embedding
        2. PromptGenerator → soft prompt vector
        3. Concatenate CLS + prompt → two gMLP branches (v_c, v_e)
        4. Interaction features: [v_c, v_e, |v_c - v_e|, v_c * v_e]
        5. Linear classifier → 3 classes (FAVOR / AGAINST / NEUTRAL)

    Args:
        bert_model_name (str): HuggingFace model name (default: bert-base-uncased).
        num_labels      (int): Number of output classes (default: 3).
    """

    def __init__(
        self,
        bert_model_name: str = "bert-base-uncased",
        num_labels: int = 3,
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.prompt_gen = PromptGenerator(input_dim=768, hidden_dim=128)
        self.gmlp = gMLP(input_dim=768 + 128, output_dim=256)
        self.classifier = nn.Linear(256 * 4, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids      (Tensor): Token IDs  [batch, seq_len]
            attention_mask (Tensor): Mask tensor [batch, seq_len]

        Returns:
            logits (Tensor): Class logits [batch, num_labels]
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]          # [B, 768]

        prompt = self.prompt_gen(cls_embedding)                      # [B, 128]
        combined = torch.cat([cls_embedding, prompt], dim=-1)        # [B, 896]

        v_c = self.gmlp(combined)                                    # [B, 256]
        v_e = self.gmlp(combined)                                    # [B, 256]

        # Pairwise interaction features
        features = torch.cat(
            [v_c, v_e, torch.abs(v_c - v_e), v_c * v_e], dim=-1    # [B, 1024]
        )
        logits = self.classifier(features)                           # [B, 3]
        return logits
