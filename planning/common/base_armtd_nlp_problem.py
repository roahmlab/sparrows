import torch
import numpy as np
import time


class BaseArmtdNlpProblem():
    """
    Base for the ArmTD NLP problem. This class is used to define the optimization problem for the ArmTD.
    The class is designed to be used with the `ipopt` solver, and is based on the `cyipopt` interface.

    Constraints are defined as classes like in planning.armtd.armtd_nlp_problem or in planning.sparrows.sphere_nlp_problem.
    The constraints are called in the `__call__` method of the class, and Cons_out and Jac_out are updated in place.
    The M attribute is the number of constraints in a given constraint class
    """
    __slots__ = [
        't_plan',
        't_full',
        'dtype',
        'n_joints',
        'g_ka',
        'pos_lim',
        'vel_lim',
        'continuous_joints',
        'pos_lim_mask',
        'M_limits',
        '_g_ka_masked',
        '_pos_lim_masked',
        '_masked_eye',
        '_grad_qpos_peak',
        '_grad_qvel_peak',
        '_grad_qpos_brake',
        'qpos',
        'qvel',
        'qgoal',
        'qdgoal',
        '_additional_constraints',
        '_cons_val',
        'M',
        '_qpos_masked',
        '_qvel_masked',
        '_Cons',
        '_Jac',
        '_x_prev',
        '_constraint_times',
        'num_constraint_evaluations',
        'num_jacobian_evaluations',
        '_joint_cost_weights',
        'use_t_final'
        ]

    def __init__(
            self,
            n_joints: int,
            g_ka: np.ndarray, #[object],
            pos_lim: np.ndarray, #[float],
            vel_lim: np.ndarray, #[float],
            continuous_joints: np.ndarray, #[int],
            pos_lim_mask: np.ndarray, #[bool],
            dtype: torch.dtype = torch.float,
            t_plan: float = 0.5,
            t_full: float = 1.0,
            weight_joint_costs: bool = False,
            ):

        # Core constants
        self.t_plan = t_plan
        self.t_full = t_full
        self.use_t_final = False
        # convert from torch dtype to np dtype
        self.dtype = torch.empty(0,dtype=dtype).numpy().dtype

        # Optimization parameters and range, which is known a priori for armtd
        self.n_joints = n_joints
        self.g_ka = g_ka

        # Joint limit constraints
        self.pos_lim = pos_lim
        self.vel_lim = vel_lim
        self.continuous_joints = continuous_joints
        self.pos_lim_mask = pos_lim_mask
        self.M_limits = int(2*self.n_joints + 6*self.pos_lim_mask.sum())

        # Extra usefuls precomputed for joint limit
        self._g_ka_masked = self.g_ka[self.pos_lim_mask]
        self._pos_lim_masked = self.pos_lim[:,self.pos_lim_mask]
        self._masked_eye = np.eye(self.n_joints, dtype=self.dtype)[self.pos_lim_mask]

        # Precompute some joint limit gradients
        self._grad_qpos_peak = (0.5 * self._g_ka_masked * self.t_plan**2).reshape(-1,1) * self._masked_eye
        self._grad_qvel_peak = np.diag(self.g_ka * self.t_plan)
        self._grad_qpos_brake = (0.5 * self._g_ka_masked * self.t_plan * self.t_full).reshape(-1,1) * self._masked_eye

        # Basic cost weighting per joint, but maybe calculate this using actual distances
        if weight_joint_costs:
            self._joint_cost_weights = np.linspace(1, 0, num=n_joints, endpoint=False)
        else:
            self._joint_cost_weights = 1

    def reset(self, qpos, qvel, qgoal, additional_constraints = [], cons_val = 0, t_final_thereshold=0., qdgoal = None):
        # Key parameters for optimization
        self.qpos = qpos
        self.qvel = qvel
        self.qgoal = qgoal
        self.qdgoal = qdgoal
        self.use_t_final = np.linalg.norm(self.qpos - self.qgoal) < t_final_thereshold
        # FO constraint holder & functional
        if not isinstance(additional_constraints, list):
            additional_constraints = [additional_constraints]
        self._additional_constraints = additional_constraints
        self._cons_val = cons_val

        self.M = self.M_limits
        for constraints in additional_constraints:
            self.M += int(constraints.M)
        
        # Prepare the rest of the joint limit constraints
        self._qpos_masked = self.qpos[self.pos_lim_mask]
        self._qvel_masked = self.qvel[self.pos_lim_mask]

        # Constraints and Jacobians
        self._Cons = np.zeros(self.M, dtype=self.dtype)
        self._Jac = np.zeros((self.M, self.n_joints), dtype=self.dtype)

        # Internal
        self._x_prev = np.zeros(self.n_joints)*np.nan
        self._constraint_times = []
        
        # IPOPT stats
        self.num_constraint_evaluations = 0
        self.num_jacobian_evaluations = 0

    def objective(self, x):
        qplan = self.qpos + self.qvel*self.t_plan + 0.5*self.g_ka*x*self.t_plan**2
        if self.use_t_final:
            qvel_plan = self.qvel + self.t_plan * self.g_ka * x
            qfinal = qplan + 0.5 * qvel_plan * (self.t_full - self.t_plan)
            return np.sum(self._joint_cost_weights*self._wrap_cont_joints(qfinal-self.qgoal)**2)
        
        q_obj = np.sum(self._joint_cost_weights*self._wrap_cont_joints(qplan-self.qgoal)**2)
        if self.qdgoal is None:
            return q_obj
        qd_obj = np.sum(self._joint_cost_weights*(self.qvel+self.g_ka*x*self.t_plan-self.qdgoal)**2)
        return q_obj + qd_obj

    def gradient(self, x):
        qplan = self.qpos + self.qvel*self.t_plan + 0.5*self.g_ka*x*self.t_plan**2
        qplan_grad = 0.5*self.g_ka*self.t_plan**2
        if self.use_t_final:
            qvel_plan = self.qvel + self.t_plan * self.g_ka * x
            qfinal = qplan + 0.5 * qvel_plan * (self.t_full - self.t_plan)
            qfinal_grad = 0.5*self.g_ka*self.t_plan**2 + 0.5 * (self.t_full - self.t_plan) * self.g_ka * self.t_plan
            return self._joint_cost_weights*2*qfinal_grad*self._wrap_cont_joints(qfinal-self.qgoal)
        
        q_obj_grad = self._joint_cost_weights*2*qplan_grad*self._wrap_cont_joints(qplan-self.qgoal)
        if self.qdgoal is None:
            return q_obj_grad
        qdplan_grad = self.g_ka*self.t_plan
        qd_obj_grad = self._joint_cost_weights*2*qdplan_grad*(self.qvel+self.g_ka*x*self.t_plan - self.qdgoal)
        return q_obj_grad + qd_obj_grad

    def constraints(self, x):
        self.num_constraint_evaluations += 1
        self.compute_constraints(x)
        return self._Cons

    def jacobian(self, x):
        self.num_jacobian_evaluations += 1
        self.compute_constraints(x)
        return self._Jac

    def compute_constraints(self,x):
        if (self._x_prev!=x).any():
            start = time.perf_counter()
            self._x_prev = np.copy(x)

            # zero out the underlying constraints and jacobians
            self._Cons[...] = self._cons_val
            self._Jac[...] = 0

            # Joint limits
            self._constraints_limits(x, Cons_out=self._Cons[:self.M_limits], Jac_out=self._Jac[:self.M_limits])

            # Additional constraints as desired
            M_start = self.M_limits
            for constraints in self._additional_constraints:
                if constraints.M <= 0:
                    continue
                M_end = M_start + constraints.M
                constraints(x, Cons_out=self._Cons[M_start:M_end], Jac_out=self._Jac[M_start:M_end])
                M_start = M_end
            
            # Timing
            self._constraint_times.append(time.perf_counter() - start)

    def _constraints_limits(self, x, Cons_out=None, Jac_out=None):
        ka = x # is this numpy? we will see
        if Cons_out is None:
            Cons_out = np.empty(self.M_limits, dtype=self.dtype)
        if Jac_out is None:
            Jac_out = np.empty((self.M_limits, self.n_joints), dtype=self.dtype)

        ## position and velocity constraints
        # disable warnings for this section
        settings = np.seterr('ignore')
        # scake k and get the part relevant to the constrained positions
        scaled_k = self.g_ka*ka
        scaled_k_masked = scaled_k[self.pos_lim_mask]
        # time to optimum of first half traj.
        t_peak_optimum = -self._qvel_masked/scaled_k_masked
        # if t_peak_optimum is in the time, qpos_peak_optimum has a value
        t_peak_in_range = (t_peak_optimum > 0) * (t_peak_optimum < self.t_plan)
        # Get the position and gradient at the peak
        qpos_peak_optimum = np.nan_to_num(t_peak_in_range * (self._qpos_masked + self._qvel_masked * t_peak_optimum + 0.5 * scaled_k_masked * t_peak_optimum**2))
        grad_qpos_peak_optimum = np.nan_to_num(t_peak_in_range * 0.5*self._qvel_masked**2/(scaled_k_masked**2)).reshape(-1,1) * self._masked_eye
        # restore
        np.seterr(**settings)

        ## Position and velocity at velocity peak of trajectory
        qpos_peak = self._qpos_masked + self._qvel_masked * self.t_plan + 0.5 * scaled_k_masked * self.t_plan**2
        qvel_peak = self.qvel + scaled_k * self.t_plan

        ## Position at braking
        # braking_accel = (0 - qvel_peak)/(T_FULL - T_PLAN)
        # qpos_brake = qpos_peak + qvel_peak*(T_FULL - T_PLAN) + 0.5*braking_accel*(T_FULL-T_PLAN)**2
        # NOTE: swapped to simplified form
        qpos_brake = self._qpos_masked + 0.5 * self._qvel_masked * (self.t_full + self.t_plan) + 0.5 * scaled_k_masked * self.t_plan * self.t_full
        
        ## compute the final constraint values and store them to the desired output arrays
        qpos_possible_max_min = np.vstack((qpos_peak_optimum,qpos_peak,qpos_brake))
        qpos_ub = (qpos_possible_max_min - self._pos_lim_masked[1]).flatten()
        qpos_lb = (self._pos_lim_masked[0] - qpos_possible_max_min).flatten()
        qvel_ub = qvel_peak - self.vel_lim
        qvel_lb = (-self.vel_lim) - qvel_peak
        np.concatenate((qpos_ub, qpos_lb, qvel_ub, qvel_lb), out=Cons_out)

        ## Do the same for the gradients
        grad_qpos_ub = np.vstack((grad_qpos_peak_optimum,self._grad_qpos_peak,self._grad_qpos_brake))
        grad_qpos_lb = -grad_qpos_ub
        grad_qvel_ub = self._grad_qvel_peak
        grad_qvel_lb = -self._grad_qvel_peak
        np.concatenate((grad_qpos_ub, grad_qpos_lb, grad_qvel_ub, grad_qvel_lb), out=Jac_out)

        return Cons_out, Jac_out
    
    def _wrap_cont_joints(self, pos: np.ndarray) -> np.ndarray:
        pos = np.copy(pos)
        pos[..., self.continuous_joints] = (pos[..., self.continuous_joints] + np.pi) % (2 * np.pi) - np.pi
        return pos

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                d_norm, regularization_size, alpha_du, alpha_pr,
                ls_trials):
        pass

    @property
    def constraint_times(self):
        return self._constraint_times