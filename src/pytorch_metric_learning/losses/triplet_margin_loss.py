import torch

from ..reducers import AvgNonZeroReducer
from ..utils import loss_and_miner_utils as lmu
from .base_metric_loss_function import BaseMetricLossFunction


class TripletMarginLoss(BaseMetricLossFunction):
    """
    Args:
        margin: The desired difference between the anchor-positive distance and the
                anchor-negative distance.
        swap: Use the positive-negative distance instead of anchor-negative distance,
              if it violates the margin more.
        smooth_loss: Use the log-exp version of the triplet loss
    """

    def __init__(
        self,
        margin=0.05,
        swap=False,
        smooth_loss=False,
        triplets_per_anchor="all",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.margin = margin
        self.swap = swap
        self.smooth_loss = smooth_loss
        self.triplets_per_anchor = triplets_per_anchor
        self.add_to_recordable_attributes(list_of_names=["margin"], is_stat=False)

    def compute_loss(self, embeddings, labels, indices_tuple):
        indices_tuple = lmu.convert_to_triplets(
            indices_tuple, labels, t_per_anchor=self.triplets_per_anchor
        )
        anchor_idx, positive_idx, negative_idx = indices_tuple
        if len(anchor_idx) == 0:
            return self.zero_losses()
        mat = self.distance(embeddings)
        ap_dists = mat[anchor_idx, positive_idx]
        an_dists = mat[anchor_idx, negative_idx]
        if self.swap:
            pn_dists = mat[positive_idx, negative_idx]
            an_dists = self.distance.smallest_dist(an_dists, pn_dists)

        current_margins = self.distance.margin(an_dists, ap_dists)
        if self.smooth_loss:
            loss = torch.log(1 + torch.exp(-current_margins))
        else:
            loss = torch.nn.functional.relu(-current_margins + self.margin)

        return {
            "loss": {
                "losses": loss,
                "indices": indices_tuple,
                "reduction_type": "triplet",
            }
        }

    def get_default_reducer(self):
        return AvgNonZeroReducer()

class MyNNTripletMarginLoss(TripletMarginLoss):
    def __init__(
        self,
        margin=0.05,
        swap=False,
        smooth_loss=False,
        triplets_per_anchor="all",
        **kwargs
    ):
        super().__init__(margin=margin, swap=swap, smooth_loss=smooth_loss, triplets_per_anchor=triplets_per_anchor, **kwargs)
        print(">>>> using flymin's version for n->n triplet loss <<<<")
    
    def compute_loss(self, embeddings, labels, indices_tuple):
        indices_tuple = lmu.convert_to_triplets(
            indices_tuple, labels, t_per_anchor=self.triplets_per_anchor
        )
        anchor_idx, positive_idx, negative_idx = indices_tuple
        if len(anchor_idx) == 0:
            return self.zero_losses()
        assert len(embeddings) % 11 == 0
        N = len(embeddings) // 11
        # pos_mat should have size N x N
        pos_mat = self.distance(embeddings[:N], embeddings[N:2*N])
        # neg_mat should have size N x 9N
        neg_mat = self.distance(embeddings[:N], embeddings[2*N:])
        ap_dists = pos_mat[anchor_idx, positive_idx]
        an_dists = neg_mat[anchor_idx, negative_idx]
        assert self.swap is False

        current_margins = self.distance.margin(an_dists, ap_dists)
        if self.smooth_loss:
            loss = torch.log(1 + torch.exp(-current_margins))
        else:
            loss = torch.nn.functional.relu(-current_margins + self.margin)

        return {
            "loss": {
                "losses": loss,
                "indices": indices_tuple,
                "reduction_type": "triplet",
            }
        }

class MyNNTripletMarginLossV2(TripletMarginLoss):
    def __init__(
        self,
        margin=0.05,
        swap=False,
        smooth_loss=False,
        triplets_per_anchor="all",
        **kwargs
    ):
        super().__init__(margin=margin, swap=swap, smooth_loss=smooth_loss, triplets_per_anchor=triplets_per_anchor, **kwargs)
        print(">>>> using flymin's version for n->n triplet loss V2 <<<<")
    
    def compute_loss(self, embeddings, labels, indices_tuple):
        indices_tuple = lmu.convert_to_triplets(
            indices_tuple, labels, t_per_anchor=self.triplets_per_anchor
        )
        anchor_idx, positive_idx, negative_idx = indices_tuple
        if len(anchor_idx) == 0:
            return self.zero_losses()
        assert len(embeddings) % 11 == 0
        N = len(embeddings) // 11
        # pos_mat should have size N x N
        pos_mat = self.distance(embeddings[:N], embeddings[N:2*N])
        # neg_mat should have size N x 9N
        neg_mat = self.distance(embeddings[:N], embeddings[2*N:])
        ap_dists = pos_mat[anchor_idx[:,0], positive_idx]
        an_dists = neg_mat[anchor_idx[:,1], negative_idx]
        assert self.swap is False

        current_margins = self.distance.margin(an_dists, ap_dists)
        if self.smooth_loss:
            loss = torch.log(1 + torch.exp(-current_margins))
        else:
            loss = torch.nn.functional.relu(-current_margins + self.margin)

        return {
            "loss": {
                "losses": loss,
                "indices": indices_tuple,
                "reduction_type": "triplet",
            }
        }