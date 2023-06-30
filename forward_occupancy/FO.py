# TODO CLEANUP IMPORTS & DOCUMENT

from zonopy import polyZonotope, matPolyZonotope, batchPolyZonotope, batchMatPolyZonotope
from collections import OrderedDict
from zonopyrobots.kinematics import forward_kinematics
from zonopyrobots import ZonoArmRobot

from typing import Union, Dict, List, Tuple
from typing import OrderedDict as OrderedDictType

# Use forward kinematics to get the forward occupancy
# Note: zono_order=2 is 5 times faster than zono_order=20 on cpu
def forward_occupancy(rotatotopes: Union[Dict[str, Union[matPolyZonotope, batchMatPolyZonotope]],
                                         List[Union[matPolyZonotope, batchMatPolyZonotope]]],
                      robot: ZonoArmRobot,
                      zono_order: int = 20,
                      links: List[str] = None,
                      link_zono_override: Dict[str, polyZonotope] = None,
                      ) -> Tuple[OrderedDictType[str, Union[polyZonotope, batchPolyZonotope]],
                                 OrderedDictType[str, Union[Tuple[polyZonotope, matPolyZonotope],
                                                            Tuple[batchPolyZonotope, batchMatPolyZonotope]]]]:
    
    link_fk_dict = forward_kinematics(rotatotopes, robot, zono_order, links=links)
    urdf = robot.urdf
    link_zonos = {name: robot.link_data[urdf._link_map[name]].bounding_pz for name in link_fk_dict.keys()}
    if link_zono_override is not None:
        link_zonos.update(link_zono_override)
    
    fo = OrderedDict()
    for name, (pos, rot) in link_fk_dict.items():
        link_zono = link_zonos[name]
        fo_link = pos + rot@link_zono
        fo[name] = fo_link.reduce_indep(zono_order)
    
    return fo, link_fk_dict

import torch
def forward_occupancy_old(rotatos,link_zonos,robot_params,zono_order = 2):
    '''
    P: <list>
    '''    
    dtype, device = rotatos[0].dtype, rotatos[0].device
    n_joints = robot_params['n_joints']
    P = robot_params['P']
    R = robot_params['R']
    P_motor = [polyZonotope(torch.zeros(3,dtype=dtype,device=device).unsqueeze(0))]
    R_motor = [matPolyZonotope(torch.eye(3,dtype=dtype,device=device).unsqueeze(0))]
    FO_link = []
    for i in range(n_joints):
        P_motor_temp = R_motor[-1]@P[i] + P_motor[-1]
        P_motor.append(P_motor_temp.reduce_indep(zono_order))
        R_motor_temp = R_motor[-1]@R[i]@rotatos[i]
        R_motor.append(R_motor_temp.reduce_indep(zono_order))
        FO_link_temp = R_motor[-1]@link_zonos[i] + P_motor[-1]
        FO_link.append(FO_link_temp.reduce_indep(zono_order))
    return FO_link, R_motor[1:], P_motor[1:]