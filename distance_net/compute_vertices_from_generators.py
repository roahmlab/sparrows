import sys
import torch
sys.path.append('../')

EPS = 1e-4
LARGE = 1e6
THERESHOLD = 1e5

def compute_edges_from_generators(centers, generators, A, b):
    '''
    Compute the edges of a zonotope from its generators.

    Args:
        centers: torch.Tensor, shape (num_obs, 3)
            The center of the zonotope.
        generators: torch.Tensor, shape (num_obs, num_generators, 3)
            The generators of the zonotope.
        A: torch.Tensor, shape (num_obs, num_hyperplanes, 3)
            The normal vectors of the hyperplanes that define the zonotope in polytope form.
        b: torch.Tensor, shape (num_obs, num_hyperplanes)
            The offset of the hyperplanes that define the zonotope in polytope form.

    Returns:
        v1: torch.Tensor, shape (num_obs, num_edges, 3)
            The first vertex of the edges that define the zonotope.
        v2: torch.Tensor, shape (num_obs, num_edges, 3)
            The second vertex of the edges that define the zonotope.
    '''
    num_generators = generators.shape[1]
    device = A.device
    batch_size = A.shape[0]
    num_hyperplanes = A.shape[1]
    num_generator_combinations = 2 ** num_generators
    mask = 2 ** torch.arange(num_generators - 1, -1, -1, device=device)
    coeff = torch.arange(num_generator_combinations, device=centers.device).unsqueeze(-1).bitwise_and(mask).ne(0).float().view(1, num_generator_combinations, num_generators, 1).repeat(batch_size, 1, 1, 1)
    coeff[coeff == 0.] = -1.
    generator_combinations = generators.view(batch_size, 1, num_generators, 3).expand(batch_size, num_generator_combinations, num_generators, 3) * coeff
    vertices = generator_combinations.sum(dim=-2) + centers

    A1 = A.unsqueeze(1).expand(batch_size, num_generator_combinations, num_hyperplanes, 3)
    b1 = b.unsqueeze(1).expand(batch_size, num_generator_combinations, num_hyperplanes, 1)
    V = vertices.view(batch_size, num_generator_combinations, 1, 3)    
    AV_minus_b = (torch.sum(A1 * V, dim=-1, keepdim=True) - b1).view(batch_size, num_generator_combinations, num_hyperplanes)
    
    AV_minus_b_max = torch.max(AV_minus_b, dim=-1).values
    non_valid_vertices = torch.logical_or(AV_minus_b_max > EPS, AV_minus_b_max < -EPS)
    vertices[non_valid_vertices] = torch.nan
    
    AV_minus_b = AV_minus_b.permute(0, 2, 1) # shape: batch_size, num_hyperplanes, num_combinations
    hyperplane_mask = torch.logical_or((AV_minus_b > EPS), (AV_minus_b < -EPS))
    edge_candidates = vertices.repeat(1, num_hyperplanes, 1).view(batch_size, num_hyperplanes, num_generator_combinations, 3)
    edge_candidates[hyperplane_mask] = torch.nan
    
    sorted = edge_candidates[:,:,:,0].sort(dim=2, stable=True)
    edge_candidates = edge_candidates.gather(dim=2, index=sorted.indices.unsqueeze(-1).expand(batch_size, num_hyperplanes, num_generator_combinations,3))

    valid_mask = torch.logical_not(torch.any(torch.isnan(edge_candidates), dim=3))
    num_valid = torch.sum(valid_mask, dim=-1).max().item()
    edge_candidates = edge_candidates[:, :, :num_valid, :]
    
    hyperplane_centers = torch.nanmean(edge_candidates, dim=-2, keepdim=True)
    hyperplane_vertices_vectors = edge_candidates - hyperplane_centers
    hyperplane_vertices_base_vectors = edge_candidates[:,:,0:1] - hyperplane_centers
    hyperplane_vertices_vectors = hyperplane_vertices_vectors / torch.linalg.norm(hyperplane_vertices_vectors, dim=-1, keepdim=True)
    hyperplane_vertices_base_vectors = hyperplane_vertices_base_vectors / torch.linalg.norm(hyperplane_vertices_base_vectors, dim=-1, keepdim=True)
    hyperplane_vertices_vectors_dotproduct = torch.sum(hyperplane_vertices_vectors * hyperplane_vertices_base_vectors, dim=-1).clamp(-1, 1) # important to clamp due to numerical error
    hyperplane_vertices_vectors_crossproduct = torch.cross(hyperplane_vertices_vectors, hyperplane_vertices_base_vectors, dim=-1)
    A2 = A.unsqueeze(2).expand(batch_size, num_hyperplanes, num_valid, 3)
    cw_direction = torch.sum(hyperplane_vertices_vectors_crossproduct * A2, dim=-1) < 0
    angles = torch.acos(hyperplane_vertices_vectors_dotproduct)
    angles[cw_direction] *= -1
    
    _, vertices_order = torch.sort(angles, dim=-1)
    vertices_order = vertices_order.unsqueeze(-1).expand(batch_size, num_hyperplanes, num_valid, 3)
    edge_candidates = edge_candidates.gather(2, vertices_order)

    v2_indices = list(range(1, num_valid)) + [0]
    v2 = edge_candidates[:,:,v2_indices,:]
    vertices_notnan_mask = torch.logical_not(torch.any(torch.isnan(edge_candidates), dim=-1))
    v2_vertices_indices = (vertices_notnan_mask.sum(-1) - 1)
    v2.scatter_(dim=2, index=v2_vertices_indices.view(batch_size, num_hyperplanes, 1, 1).expand_as(v2), src=edge_candidates[:,:,0:1,:].expand_as(v2))
    v1 = edge_candidates.view(batch_size, -1, 3)
    v2 = v2.view(batch_size, -1, 3)
    
    return v1, v2
