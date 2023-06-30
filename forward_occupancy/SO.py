# TODO CLEANUP IMPORTS & DOCUMENT

from zonopy import polyZonotope, matPolyZonotope, batchPolyZonotope, batchMatPolyZonotope
from collections import OrderedDict
from zonopyrobots.kinematics import forward_kinematics
from zonopyrobots import ZonoArmRobot
from urchin import URDF
import numpy as np
import functools

from typing import Union, Dict, List, Tuple
from typing import OrderedDict as OrderedDictType

SQRT_3 = np.sqrt(3)

@functools.lru_cache
def __get_fk_so_links(urdf: URDF,
                      links: Tuple[str, ...] = None
                      ) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    fk_links = None # the links we use for FK
    so_links = None # the links we actually generate the SO for
    if links is not None:
        fk_links = set()
        so_links = []
        for link_name in links:
            link = urdf.link_map[link_name]
            next_link_list = list(urdf._G.reverse()[link].keys())
            if len(next_link_list) == 0:
                import warnings
                warnings.warn('Not generating SO for ' + link_name)
                continue
            for nl in next_link_list:
                fk_links.add(nl.name)
            fk_links.add(link_name)
            so_links.append(link_name)
        fk_links = list(fk_links)
    else:
        so_links = urdf.links.copy()
        so_links.remove(urdf.base_link)
        fk_links = [link.name for link in so_links]
        for el in urdf.end_links:
            so_links.remove(el)
        so_links = [link.name for link in so_links]
    return tuple(fk_links), tuple(so_links)


@functools.lru_cache
def __make_link_pairs(robot: ZonoArmRobot,
                      so_links: Tuple[str, ...]
                      ) -> OrderedDictType[str, Tuple[str,str]]:
    # Create the link occupancy pairs
    lso = OrderedDict()
    for link_name in so_links:
        pj = robot.link_parent_joint[link_name].name
        pairs = [(pj, cj.name) for cj in robot.link_child_joints[link_name]]
        lso[link_name] = pairs
    return lso


# Use forward occupancy or forward kinematics to get the joint occupancy
# For the true bohao approach, enable use_outer_bb
# ONLY USE WITH LINKS THAT ARE NOT BASE OR END LINKS
def sphere_occupancy(rotatotopes: Union[Dict[str, Union[matPolyZonotope, batchMatPolyZonotope]],
                                        List[Union[matPolyZonotope, batchMatPolyZonotope]]],
                     robot: ZonoArmRobot,
                     zono_order: int = 20,
                     links: List[str] = None,
                     joint_radius_override: Dict[str, float] = {},
                     cylindrical_joints: bool = True,
                     ) -> Tuple[OrderedDictType[str, Union[polyZonotope, batchPolyZonotope]],
                                OrderedDictType[str, Union[Tuple[polyZonotope, matPolyZonotope],
                                                           Tuple[batchPolyZonotope, batchMatPolyZonotope]]]]:
    
    urdf = robot.urdf
    # if len(urdf.end_links) > 1:
    #     import warnings
    #     warnings.warn('The robot appears to be branched! Branched robots may not function as expected.')
    
    # Get adjacent links as needed.
    # Remove end links
    # fk_links: the links we use for FK
    # so_links: the links we actually generate the SO for
    fk_links, so_links = __get_fk_so_links(urdf, tuple(links) if links is not None else None)

    # Get forward kinematics
    link_fk_dict = forward_kinematics(rotatotopes, robot, zono_order, links=fk_links)
    
    # Create the joint occupancy
    joint_fk_dict = OrderedDict()
    jso = OrderedDict()
    for name, val in link_fk_dict.items():
        joint = robot.link_parent_joint[name]
        joint_name = joint.name
        if joint_name in joint_radius_override:
            joint_radius = joint_radius_override[joint_name]
        else:
            # Make sure create_joint_occupany on the robot is set!
            joint_radius = robot.joint_data[joint].radius
            if cylindrical_joints:
                joint_radius = torch.linalg.vector_norm(joint_radius.topk(2,sorted=False).values)
            else:
                torch.linalg.vector_norm(joint_radius)
        
        joint_center, joint_uncertainty = val[0].split_dep_indep()
        # uncertainty_rad = joint_uncertainty.to_interval().rad().max(-1)[0]*SQRT_3
        uncertainty_rad = torch.linalg.vector_norm(joint_uncertainty.to_interval().rad(), dim=-1)
        radius = uncertainty_rad + joint_radius

        jso[joint_name] = (joint_center, radius)
        joint_fk_dict[joint_name] = val
    
    # Create the link occupancy pairs
    lso = __make_link_pairs(robot, so_links)
    
    return jso, lso, joint_fk_dict

import torch
def make_spheres(center_1: torch.Tensor,
                 center_2: torch.Tensor,
                 radius_1: torch.Tensor,
                 radius_2: torch.Tensor,
                 center_jac_1: torch.Tensor = None,
                 center_jac_2: torch.Tensor = None,
                 n_spheres: int = 5
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Generates centers and radii for spheres that overapproximate the taper between two spheres.
    
    Args:
        center_1: A tensor of shape (batch_dims, 3) representing the center(s) of the first sphere.
        center_2: A tensor of shape (batch_dims, 3) representing the center(s) of the second sphere.
        radius_1: A tensor of shape (batch_dims,) representing the radius/radii of the first sphere.
        radius_2: A tensor of shape (batch_dims,) representing the radius/radii of the second sphere.
        center_jac_1: A tensor of shape (batch_dims, 3, n_q) representing the Jacobian(s) to the n_q parameters that parameterize the center of the first sphere.
        center_jac_2: A tensor of shape (batch_dims, 3, n_q) representing the Jacobian(s) to the n_q parameters that parameterize the center of the second sphere.
        n_spheres: An integer representing the number of spheres to generate.
    
    Returns:
        A tuple of two tensors if either center_jac_1 or center_jac_2 is None, otherwise a tuple of four tensors:
            centers: A tensor of shape (batch_dims, n_spheres, 3) representing the centers of the generated spheres.
            radii: A tensor of shape (batch_dims, n_spheres) representing the radii of the generated spheres.
            center_jac: A tensor of shape (batch_dims, n_spheres, 3, n_q) representing the Jacobians to the n_q parameters for the centers of the generated spheres.
            radii_jac: A tensor of shape (batch_dims, n_spheres, n_q) representing the Jacobians to the n_q parameters for the radii of the generated spheres.
    '''
    ## Wrapper for the torchscript function
    if center_jac_1 is not None or center_jac_2 is not None:
        return _make_spheres_script(center_1, center_2, radius_1, radius_2, center_jac_1, center_jac_2, True, n_spheres)
    return _make_spheres_script(center_1, center_2, radius_1, radius_2, bool_jacs=False, n_spheres=n_spheres)[:2]


# @torch.jit.script
# Adding script for some reason adds excessive overhead
def _make_spheres_script(center_1: torch.Tensor,
        center_2: torch.Tensor,
        radius_1: torch.Tensor,
        radius_2: torch.Tensor,
        center_jac_1: torch.Tensor = torch.empty(0),
        center_jac_2: torch.Tensor = torch.empty(0),
        bool_jacs: bool = False,
        n_spheres: int = 5
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # This can be expensive so start this subtraction early
    djac = (center_jac_2-center_jac_1)

    ## Compute the terms
    delta = center_2 - center_1
    height = torch.linalg.vector_norm(delta, dim=-1, keepdim=True) # ..., 1
    direction = (delta)/height # ..., 3
    sidelength = height/(2*n_spheres) # ..., 1
    m_idx = (2*torch.arange(n_spheres, dtype=center_1.dtype, device=center_1.device) + 1) # n_spheres
    offsets = m_idx.unsqueeze(-1) * sidelength.unsqueeze(-2) # ..., n_spheres, 1
    radial_delta = (radius_1-radius_2)/(2*n_spheres) # ..., 1
    centers = center_1.unsqueeze(-2) + direction.unsqueeze(-2) * offsets # ..., n_spheres, 3
    radii = torch.sqrt(sidelength**2 + (radius_1.unsqueeze(-1) - m_idx * radial_delta.unsqueeze(-1))**2) # ..., n_spheres

    ## Compute the jacobians
    if bool_jacs:
        dir_jac = direction.unsqueeze(-1) # ..., 3, 1
        jach = torch.sum(dir_jac * djac, dim=-2).unsqueeze(-2) # ..., 1, n_q
        jacu = (djac - dir_jac * jach)/height.unsqueeze(-1) # ..., 3, n_q
        jacd = (m_idx/(2*n_spheres)).unsqueeze(-1) * jach # ..., n_spheres, n_q
        jacc = center_jac_1.unsqueeze(-3) + offsets.unsqueeze(-1) * jacu.unsqueeze(-3) + dir_jac.unsqueeze(-3) * jacd.unsqueeze(-2) # ..., n_spheres, 3, n_q
        jacr = jach * (sidelength / (2*n_spheres*radii)).unsqueeze(-1) # ..., n_spheres, n_q
        return centers, radii, jacc, jacr
    return centers, radii, torch.empty(0), torch.empty(0)