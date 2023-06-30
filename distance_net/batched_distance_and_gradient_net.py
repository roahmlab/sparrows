import torch
import torch.nn as nn
import numpy as np


class BatchedDistanceGradientNet(nn.Module):
    '''
    This class is a PyTorch module that computes the distance and gradient of a point to a zonotope.
    The zonotope is defined by a set of hyperplanes and vertices.
    The distance is the minimum distance from the point to the zonotope.
    The gradient is the gradient of the distance with respect to the point.

    The distance is computed by projecting the point onto the hyperplanes and vertices.
    The gradient is computed at the same time as the distance, and returned during the forward pass.
    There is no backward pass implemented for this module.
    '''
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, point, hyperplane_A, hyperplane_b, v1, v2, EPS: float = 1e-4):
        '''
        Compute the distance and gradient of a point to a zonotope.

        Args:
            point: torch.Tensor, shape (num_points_batch, 3)
                The point to compute the distance to the zonotope.
            hyperplane_A: torch.Tensor, shape (num_obs, num_hyperplanes, 3)
                The normal vectors of the hyperplanes that define the zonotope.
            hyperplane_b: torch.Tensor, shape (num_obs, num_hyperplanes)
                The offset of the hyperplanes that define the zonotope.
            v1: torch.Tensor, shape (num_obs, num_vertices, 3)
                The first vertex of the edges that define the zonotope.
            v2: torch.Tensor, shape (num_obs, num_vertices, 3)
                The second vertex of the edges that define the zonotope.
            EPS: float, optional
                A small value to avoid division by zero.
        
        Returns:
            distances: torch.Tensor, shape (num_points_batch, num_obs)
                The minimum distance from each point to the zonotope.
            gradients: torch.Tensor, shape (num_points_batch, num_obs, 3)
                The gradient of the distance with respect to the point.
        '''
        # point: (num_points_batch, 3)
        # hyperplane_A: (num_obs, num_hyperplanes, 3)
        # hyperplane_b: (num_obs, num_hyperplanes)
        # v1: (num_obs, num_vertices, 3)
        # v2: (num_obs, num_vertices, 3)

        point_shape = point.shape[:-1]
        # num_points = point_shape.numel()

        num_obs = hyperplane_A.shape[0]
        # num_vertices = v1.shape[1]
        # num_hyperplanes = hyperplane_A.shape[1]

        distances = torch.zeros(point_shape + (num_obs,), device=point.device, dtype=point.dtype)
        gradients = torch.zeros(point_shape + (num_obs, 3), device=point.device, dtype=point.dtype)

        ### Get the perpendicular distance from the point to faces
        # map point c to c1, its projection onto the hyperplane
        # target, (num_points_batch, num_obs, num_hyperplanes, 3)
        points = point[..., None, None, :] # (num_points_batch, num_obs, num_hyperplanes, 3)
        Ap_minus_b = torch.sum(hyperplane_A * points, dim=-1) - hyperplane_b # (num_points_batch, num_obs, num_hyperplanes)
        Ap_minus_b = Ap_minus_b.nan_to_num(nan=-torch.inf, posinf=-torch.inf) # should posinf be positive

        # check sign; if there is negative distance, directly assign distance
        is_negative_distance = torch.all(Ap_minus_b <= 0, dim=-1)
        max_negative_pairs = torch.max(Ap_minus_b, dim=-1)
        distances[is_negative_distance] = max_negative_pairs.values[is_negative_distance]
        gradients[is_negative_distance] = hyperplane_A.expand(point_shape + hyperplane_A.shape)[is_negative_distance, max_negative_pairs.indices[is_negative_distance]]

        # we only need to consider the positive distances in the rest of the discussion
        non_negative_distance = torch.logical_not(is_negative_distance)
        
        hyperplane_projected_points = (points - Ap_minus_b.unsqueeze(-1) * hyperplane_A)
        # (num_points_batch, num_obs, num_hyperplanes, 3)
        Apprime_minus_b = (hyperplane_A.unsqueeze(-3)@hyperplane_projected_points.unsqueeze(-1)).squeeze(-1) - hyperplane_b.unsqueeze(-2)
        # (num_points_batch, num_obs, num_hyperplanes, num_hyperplanes)
        Apprime_minus_b = Apprime_minus_b.nan_to_num(nan=-torch.inf, posinf=-torch.inf) # should posinf be positive
        on_zonotope = torch.all(Apprime_minus_b <= EPS, dim=-1)

        perpendicular_distances = torch.linalg.norm(points - hyperplane_projected_points, dim=-1).clone() # (num_points_batch, num_obs, num_hyperplanes)
        perpendicular_distances[torch.logical_not(on_zonotope)] = torch.inf
        perpendicular_distances = perpendicular_distances.nan_to_num(nan=torch.inf, neginf=torch.inf)
        min_perpendicular_pair = perpendicular_distances.min(dim=-1)
        distances[non_negative_distance] = min_perpendicular_pair.values[non_negative_distance]
        gradients[non_negative_distance] = hyperplane_A.expand(point_shape + hyperplane_A.shape)[non_negative_distance, min_perpendicular_pair.indices[non_negative_distance]]
        ### Get the distance from the point to edges
        # NOTE: should probably pad vertices with fake edges such that every zonotope have the same number of verticess
        # edge_distance_points = point[non_negative_distance].repeat(1, num_vertices).view(num_nonnegative_points, num_vertices, 3)
        v2_minus_v1_square = torch.square(v2 - v1)
        v2_minus_v1_square_sum = torch.sum(v2_minus_v1_square, dim=-1, keepdim=True) # + EPS

        t_hat = torch.sum((points - v1) * (v2 - v1), dim=-1, keepdim=True) / v2_minus_v1_square_sum
        # (num_points_batch, num_obs, num_vertices, 1)
        t_star = t_hat.clamp(0,1)
        v = v1 + t_star * (v2 - v1)
        edge_distances = torch.linalg.norm(points - v, dim=-1, keepdim=True)
        edge_distances = edge_distances.nan_to_num(nan=torch.inf).squeeze(-1)
        # (num_points_batch, num_obs, num_vertices)
        
        edge_distance_pair = edge_distances.min(dim=-1)
        edge_distances = edge_distance_pair.values
        use_edge_distance = torch.logical_and((edge_distances < distances), non_negative_distance)

        distances[use_edge_distance] = edge_distances[use_edge_distance]
        edge_distance_points = points.expand(point_shape + (num_obs,-1,-1))[use_edge_distance].squeeze(-2)
        v = v[use_edge_distance, edge_distance_pair.indices[use_edge_distance]]
        gradients[use_edge_distance] = (edge_distance_points - v) / edge_distances[use_edge_distance].unsqueeze(-1)

        return distances, gradients
