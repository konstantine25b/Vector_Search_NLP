import torch
import torch.nn.functional as F


def contrastive_loss(
    query_emb: torch.Tensor,
    doc_emb: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    if query_emb.shape[0] != doc_emb.shape[0]:
        raise ValueError("query_emb and doc_emb must have the same batch size")
    if query_emb.shape[1] != doc_emb.shape[1]:
        raise ValueError("query_emb and doc_emb must have the same embedding size")
    if margin < 0:
        raise ValueError("margin must be non-negative")
    b = query_emb.shape[0]
    if b == 0:
        return query_emb.sum() * 0.0
    dist = torch.cdist(query_emb, doc_emb, p=2)
    pos = (dist.diag() ** 2).mean()
    if b == 1:
        return pos
    eye = torch.eye(b, dtype=torch.bool, device=dist.device)
    neg_hinge = F.relu(margin - dist) ** 2
    neg_hinge = neg_hinge.masked_fill(eye, 0.0)
    neg = neg_hinge.sum() / (b * (b - 1))
    return pos + neg
