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
    y = labels[:len(labels)//3]
    wrong_y = labels[-len(labels)//3:]
    # an and ap here are symmetric
    an = (y.unsqueeze(1) != wrong_y.unsqueeze(0)).byte()
    ap = torch.eye(len(y), len(y)).byte().to(an.device)
    triplets = ap.unsqueeze(2) * an.unsqueeze(1)
    return torch.where(triplets)


class MyTripletMarginMiner(TripletMarginMiner):
    def __init__(self, margin, type_of_triplets, **kwargs):
        super().__init__(margin=margin, type_of_triplets=type_of_triplets, **kwargs)
        print(">>>> using flymin's version for 1->n triplet miner <<<<")

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        anchor_idx, positive_idx, negative_idx = get_my_triplets_indices(
            labels, ref_labels
        )
        positive_idx = positive_idx + len(labels) // 3
        negative_idx = negative_idx + len(labels) // 3 * 2
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

def get_myNN_triplets_indices(labels, ref_labels=None):
    N = len(labels) // 11
    y = labels[:N]
    wrong_y = labels[-N*9:]
    an = (y.unsqueeze(1) != wrong_y.unsqueeze(0)).byte()
    ap = torch.eye(len(y), len(y)).byte().to(an.device)
    triplets = ap.unsqueeze(2) * an.unsqueeze(1)
    return torch.where(triplets)

class MyNNTripletMarginMiner(TripletMarginMiner):
    def __init__(self, margin, type_of_triplets, **kwargs):
        super().__init__(margin=margin, type_of_triplets=type_of_triplets, **kwargs)
        print(">>>> using flymin's version for n->n triplet miner <<<<")

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        """Generate pos pair and neg pair

        Args:
            For every batch size = N inputs
            embeddings (4D Tensor): N + N + 9N for ori + y_gen + wrong_y_gen
            labels (1D Tensor): N + N + 9N for y + y + wrong_y
            ref_emb ([type]): embeddings
            ref_labels ([type]): labels

        Returns:
            tuple of anchor_idx, positive_idx, negative_idx
        """
        anchor_idx, positive_idx, negative_idx = get_myNN_triplets_indices(
            labels, ref_labels
        )
        assert len(embeddings) % 11 == 0
        N = len(embeddings) // 11
        # pos_mat should have size N x N
        pos_mat = self.distance(embeddings[:N], ref_emb[N:2*N])
        # neg_mat should have size N x 9N
        neg_mat = self.distance(embeddings[:N], ref_emb[2*N:])
        ap_dist = pos_mat[anchor_idx, positive_idx]
        an_dist = neg_mat[anchor_idx, negative_idx]
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