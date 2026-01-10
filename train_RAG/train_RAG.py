import json
import torch
import torch.nn as nn
import torch.nn.functional as f
from transformers import AutoModel, AutoTokenizer

from torch.utils.data import Dataset

class QADataset(Dataset):
    def __init__(self, data_path: str):
        self.data = json.load(open(data_path, 'r', encoding='utf-8'))
        self.train_data = self.data['train']
        self.valid_data = self.data['eval']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.train_data[index]


class BGEEncoder(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    @staticmethod
    def mean_pooling(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform mean pooling over valid tokens.

        Args:
            hidden_states: (B, T, H) token embeddings.
            attention_mask: (B, T) attention mask.

        Returns:
            (B, H) pooled embeddings.
        """
        mask = attention_mask.unsqueeze(-1).float()
        summed = torch.sum(hidden_states * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-6)
        return summed / counts

    def encode(self, text: str, device: torch.device):
        tokens = self.tokenizer(
            text,
            padding = True,
            truncation=True,
            max_length=512,
            return_tensors="pt").to(device)
        
        outputs = self.model(**tokens)
        pooled = self.mean_pooling(
            outputs.last_hidden_state,
            tokens["attention_mask"],
        )
        return f.normalize(pooled, p=2, dim=-1)

class DenseRetrievalLoss(nn.Module):
    """Contrastive loss using in-batch negatives."""

    def forward(
        self,
        query_emb: torch.Tensor,
        passage_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive loss.

        Args:
            query_emb: (B, H) query embeddings.
            passage_emb: (B, H) passage embeddings.

        Returns:
            Scalar loss.
        """
        logits = query_emb @ passage_emb.t()
        labels = torch.arange(
            logits.size(0),
            device=logits.device,
        )
        return f.cross_entropy(logits, labels)

from torch.utils.data import DataLoader
from torch.optim import AdamW


def train_bge_retriever(
    json_path: str,
    model_name: str = "BAAI/bge-base-en-v1.5",
    batch_size: int = 16,
    lr: float = 2e-5,
    epochs: int = 10,
) -> BGEEncoder:
    """
    Train a BGE-based dense retriever using in-batch negatives.

    Args:
        json_path: Path to training JSON file.
        model_name: BGE model name.
        batch_size: Batch size.
        lr: Learning rate.
        epochs: Number of training epochs.

    Returns:
        Trained BGEEncoder model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = QADataset(json_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    model = BGEEncoder(model_name).to(device)
    criterion = DenseRetrievalLoss()
    optimizer = AdamW(model.parameters(), lr=lr)

    model.train()

    for epoch in range(epochs):
        for batch in loader:
            questions = [
                f"Represent this question for retrieving relevant passages: {q}"
                for q in batch["question"]
            ]
            answers = batch["answer"]

            q_emb = model.encode(questions, device)
            a_emb = model.encode(answers, device)

            loss = criterion(q_emb, a_emb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, loss={loss.item():.4f}")

    return model




if __name__ == "__main__":
    train_bge_retriever(json_path = 'data/insurance_carrier_qa_dataset.json')

