import torch
import torch.nn.functional as F


def infonce_similarity_matrix(
    query_emb: torch.Tensor,
    doc_emb: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return (query_emb @ doc_emb.transpose(0, 1)) / temperature


def infonce_loss(
    query_emb: torch.Tensor,
    doc_emb: torch.Tensor,
    temperature: float = 0.05,
    symmetric: bool = True,
) -> torch.Tensor:
    if query_emb.shape[0] != doc_emb.shape[0]:
        raise ValueError("query_emb and doc_emb must have the same batch size")
    if query_emb.shape[1] != doc_emb.shape[1]:
        raise ValueError("query_emb and doc_emb must have the same embedding size")
    logits = infonce_similarity_matrix(query_emb, doc_emb, temperature)
    batch_size = logits.shape[0]
    if batch_size == 0:
        return query_emb.sum() * 0.0
    labels = torch.arange(batch_size, device=logits.device, dtype=torch.long)
    loss_q2d = F.cross_entropy(logits, labels)
    if not symmetric:
        return loss_q2d
    loss_d2q = F.cross_entropy(logits.transpose(0, 1), labels)
    return 0.5 * (loss_q2d + loss_d2q)
