import torch

from ..utils import loss_and_miner_utils as lmu
from .base_miner import BaseTupleMiner


class TripletMarginMiner(BaseTupleMiner):
    """
    Returns triplets that violate the margin
    Args:
        margin
        type_of_triplets: options are "all", "hard", or "semihard".
                "all" means all triplets that violate the margin
                "hard" is a subset of "all", but the negative is closer to the anchor than the positive
                "semihard" is a subset of "all", but the negative is further from the anchor than the positive
            "easy" is all triplets that are not in "all"
    """

    def __init__(self, margin=0.2, type_of_triplets="all", **kwargs):
        super().__init__(**kwargs)
        self.margin = margin
        self.type_of_triplets = type_of_triplets
        self.add_to_recordable_attributes(
            list_of_names=["margin"], is_stat=False)
        self.add_to_recordable_attributes(
            list_of_names=["avg_triplet_margin", "pos_pair_dist",
                           "neg_pair_dist"],
            is_stat=True,)

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        anchor_idx, positive_idx, negative_idx = lmu.get_all_triplets_indices(
            labels, ref_labels
        )
        mat = self.distance(embeddings, ref_emb)
        ap_dist = mat[anchor_idx, positive_idx]
        an_dist = mat[anchor_idx, negative_idx]
        triplet_margin = (
            ap_dist - an_dist
            if self.distance.is_inverted else an_dist - ap_dist)

        if self.type_of_triplets == "easy":
            threshold_condition = triplet_margin > self.margin
        else:
            threshold_condition = triplet_margin <= self.margin
            if self.type_of_triplets == "hard":
                threshold_condition &= triplet_margin <= 0
            elif self.type_of_triplets == "semihard":
                threshold_condition &= triplet_margin > 0

        return (
            anchor_idx[threshold_condition],
            positive_idx[threshold_condition],
            negative_idx[threshold_condition],
        )

    def set_stats(self, ap_dist, an_dist, triplet_margin):
        if self.collect_stats:
            with torch.no_grad():
                self.pos_pair_dist = torch.mean(ap_dist).item()
                self.neg_pair_dist = torch.mean(an_dist).item()
                self.avg_triplet_margin = torch.mean(triplet_margin).item()


def get_my_triplets_indices(labels, ref_labels=None):
    if ref_labels is None:
        ref_labels = labels
    y = labels[:len(labels)//4]
    wrong_y = labels[-len(labels)//4:]
    an = (wrong_y.unsqueeze(0) == wrong_y.unsqueeze(1)).byte()
    ap = torch.eye(len(y), len(y)).byte().to(an.device)
    triplets = ap.unsqueeze(2) * an.unsqueeze(1)
    return torch.where(triplets)


class MyLabelTripletMarginMiner(TripletMarginMiner):
    def __init__(self, margin=0.2, type_of_triplets="all", **kwargs):
        super().__init__(margin=margin, type_of_triplets=type_of_triplets, **kwargs)
        print(">>>> using flymin's version for 1->n triplet miner <<<<")

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        anchor_idx, positive_idx, negative_idx = get_my_triplets_indices(
            labels, ref_labels
        )
        positive_idx = positive_idx + len(labels) // 4
        negative_idx = negative_idx + len(labels) // 4 * 3
        neg_anchor = anchor_idx + len(labels) // 4 * 2
        mat = self.distance(embeddings, ref_emb)
        ap_dist = mat[anchor_idx, positive_idx]
        an_dist = mat[neg_anchor, negative_idx]
        triplet_margin = (
            ap_dist - an_dist
            if self.distance.is_inverted else an_dist - ap_dist)

        if self.type_of_triplets == "easy":
            threshold_condition = triplet_margin > self.margin
        else:
            threshold_condition = triplet_margin <= self.margin
            if self.type_of_triplets == "hard":
                threshold_condition &= triplet_margin <= 0
            elif self.type_of_triplets == "semihard":
                threshold_condition &= triplet_margin > 0

        return (
            anchor_idx[threshold_condition],
            positive_idx[threshold_condition],
            negative_idx[threshold_condition],
        )

def get_myNN_triplets_indices(labels, ref_labels=None):
    N = len(labels) // 20
    y = labels[:N]
    wrong_y = labels[-N*9:]
    ap = torch.eye(len(y), len(y)).byte().to(y.device)
    anchor_idx, positive_idx, anchor2_idx, negative_idx = [], [], [], []
    for i in range(9):
        start = i * N
        an = (
            wrong_y[start:start + N].unsqueeze(1) == wrong_y.unsqueeze(0)
        ).byte()
        triplets = ap.unsqueeze(2) * an.unsqueeze(1)
        indexs = torch.where(triplets)
        anchor_idx.append(indexs[0])
        anchor2_idx.append(indexs[0] + i*N)
        positive_idx.append(indexs[1])
        negative_idx.append(indexs[2])
    anchor_idx = torch.cat(anchor_idx)
    anchor2_idx = torch.cat(anchor2_idx)
    positive_idx = torch.cat(positive_idx)
    negative_idx = torch.cat(negative_idx)
    return anchor_idx, positive_idx, anchor2_idx, negative_idx

class MyLabelNNTripletMarginMiner(TripletMarginMiner):
    def __init__(self, margin=0.2, type_of_triplets="all", **kwargs):
        super().__init__(margin=margin, type_of_triplets=type_of_triplets, **kwargs)
        print(">>>> using flymin's version for n->n triplet miner <<<<")

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        assert len(embeddings) % 20 == 0
        anchor_idx, positive_idx, anchor2_idx, negative_idx = \
            get_myNN_triplets_indices(
                labels, ref_labels
            )
        N = len(embeddings) // 20
        # pos_mat should have size N x N
        pos_mat = self.distance(embeddings[:N], ref_emb[N:2*N])
        # neg_mat should have size 9N x 9N
        neg_mat = self.distance(embeddings[2*N:11*N], ref_emb[11*N:])
        assert neg_mat.size(0) == neg_mat.size(1)
        ap_dist = pos_mat[anchor_idx, positive_idx]
        an_dist = neg_mat[anchor2_idx, negative_idx]
        triplet_margin = (
            ap_dist - an_dist
            if self.distance.is_inverted else an_dist - ap_dist)

        if self.type_of_triplets == "easy":
            threshold_condition = triplet_margin > self.margin
        else:
            threshold_condition = triplet_margin <= self.margin
            if self.type_of_triplets == "hard":
                threshold_condition &= triplet_margin <= 0
            elif self.type_of_triplets == "semihard":
                threshold_condition &= triplet_margin > 0

        return (
            anchor_idx[threshold_condition],
            positive_idx[threshold_condition],
            anchor2_idx[threshold_condition],
            negative_idx[threshold_condition],
        )