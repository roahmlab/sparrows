import torch
# from tapcap_orig import solve_one_four as solve_one_four_orig
import copy

# highly recommended unless quadratic and imaginary tolerance is lax
# otherwise default is float32, which will significantly speed up on
# nvidia cuda
# torch.set_default_dtype(torch.float32)

# set constants
global IMAGINARY_TOLERANCE, QUADRATIC_TOLERANCE, DISTANCE_TOLERANCE, \
    CONSTRAINT_TOLERANCE, STR_T, STR_U, INF, PRINT_VERBOSE, PRINT_MATLAB
STR_T = "t"
STR_U = "u"
INF = float("Inf")

# abs threshold for a negative quadratic root
IMAGINARY_TOLERANCE = 1e-4
# abs threshold validating a root to be 0
QUADRATIC_TOLERANCE = 1e-4
# abs distance error tol to scipy.optimize
DISTANCE_TOLERANCE = 1e-4
# abs t,u soln error tol to scopy.optimize
CONSTRAINT_TOLERANCE = 1e-4

# device = (
# 	"cuda"
# 	if torch.cuda.is_available()
# 	else "mps"
# 	if torch.backends.mps.is_available()
# 	else "cpu"
# )
# print(f"Using device: {device}")

# All batched tensors will have batchSize as the "first" dim, dim=0.

def self_collision(tc1, tc2, return_grad=False):
    def setup_grad(t: torch.Tensor):
        if not return_grad:
            return t
        out = t.detach().clone().requires_grad_(True)
        return out
    tc1_copy = tuple(map(setup_grad, tc1))
    tc2_copy = tuple(map(setup_grad, tc2))
    sol, _ = solve_one_four(tc1_copy, tc2_copy)
    if not return_grad:
        return sol.detach().clone()
    sol.sum().backward()
    tc1_grad = tuple(tc.grad.clone() for tc in tc1_copy)
    tc2_grad = tuple(tc.grad.clone() for tc in tc2_copy)
    
    return sol.detach().clone(), tc1_grad, tc2_grad

def main():
    # 3D space to generate individual radius centers
    ptRange = [0,15]
    # acceptable base radius, equivalent to radius center #1.
    radRange = [0.5,1.5]
    # enable radius taper to be added onto radius center #2. r2 = r1 + [-baseRadius, segmentLength]
    deltaOn = True
    # batchSize/tensor dim per iteration
    BATCH_SIZE = 1
    # num iterations
    N = 4900

    # segAB, segCD: batchSize*N x 2 x 3, so these are row vectors corresponding to 3D, not col vectors
    # rA, rB, rC, rD: batchSize*N x 1, col vectors of batch radii
    from tqdm import tqdm
    import time
    forward_time = 0
    backward_time = 0
    tests = 10000
    def exp_avg_fn(val, new_val, exp_avg = 0.001):
        return (1-exp_avg)*val + exp_avg*new_val
    for _ in tqdm(range(tests)):
        fullsegAB, fullrA, fullrB = batchRandtpill(ptRange, radRange, deltaOn, BATCH_SIZE*N)
        fullsegCD, fullrC, fullrD = batchRandtpill(ptRange, radRange, deltaOn, BATCH_SIZE*N)

        fullsegAB.requires_grad = True
        fullsegCD.requires_grad = True
        fullrA.requires_grad = True
        fullrB.requires_grad = True
        fullrC.requires_grad = True
        fullrD.requires_grad = True

        tc1 = (fullsegAB[:,0],fullsegAB[:,1],fullrA,fullrB)
        tc2 = (fullsegCD[:,0],fullsegCD[:,1],fullrC,fullrD)

        def clone_tc(seg, ra, rb):
            seg = copy.deepcopy(seg)
            ra = copy.deepcopy(ra)
            rb = copy.deepcopy(rb)
            return (seg[:,0], seg[:,1], ra, rb), seg

        tc1_orig, seg1 = clone_tc(fullsegAB, fullrA, fullrB)
        tc2_orig, seg2 = clone_tc(fullsegCD, fullrC, fullrD)

        torch.cuda.synchronize()
        start = time.perf_counter()
        sol, ind = solve_one_four(tc1, tc2)
        torch.cuda.synchronize()
        mid = time.perf_counter()
        # print(sol)
        sol.sum().backward()
        torch.cuda.synchronize()
        end = time.perf_counter()
        forward_time = exp_avg_fn(forward_time, mid-start)
        backward_time = exp_avg_fn(backward_time, end-mid)

        # Validate
        sol_orig, ind_orig = solve_one_four_orig(tc1_orig, tc2_orig)
        sol_orig.sum().backward()

        def assert_print(a, b, tol=1e-8):
            assert torch.allclose(a, b, atol=tol), f"{a} expected, got {b}, delta {a-b}, max {torch.max(torch.abs(a-b))}. {ind} vs {ind_orig}"
        assert_print(sol, sol_orig)
        assert_print(ind, ind_orig)
        assert_print(fullsegAB.grad, seg1.grad, tol=1e-6)
        assert_print(fullsegCD.grad, seg2.grad, tol=1e-6)
        # assert_print(fullrA.grad, tc1_orig[2].grad, tol=1e-6)
        # assert_print(fullrB.grad, tc1_orig[3].grad, tol=1e-6)
        # assert_print(fullrC.grad, tc2_orig[2].grad, tol=1e-6)
        # assert_print(fullrD.grad, tc2_orig[3].grad, tol=1e-6)
    print("forward_time ms", forward_time * 1000)
    print("backward_time ms", backward_time * 1000)
    # print(fullsegAB.grad,
    # fullsegCD.grad,
    # fullrA.grad,
    # fullrB.grad,
    # fullrC.grad,
    # fullrD.grad)

    # store computed solns here:

# batch tapered pill generation
# Param:
# ptRange: [#, #] cube dims to generate radii centers
# radRange: [#, #] base radius range, equal to radius of center #1
# deltaOn: boolean to generate random taper to add to base radius, rad2 = baseRad + [-baseRad, segmentLength]
# Returns: 
# segments: batchSize x 2 x 3, row tensors corresponding to 3D radius centers
# r1: batchSize x 1, base radius corresponding to center 1
# r2: batchSize x 1: tapered radius computed by adding delta to r1
@torch.no_grad()
def batchRandtpill(ptRange, radRange, deltaOn, batchSize):
    segments = ptRange[0] + (ptRange[1] - ptRange[0])*torch.rand([batchSize,2,3])
    length = torch.norm(segments[:,0,:] - segments[:,1,:], dim =1)
    r1 = radRange[0] + (radRange[1] - radRange[0])*torch.rand([batchSize])
    if deltaOn:
        r2 = (length + r1)*torch.rand([batchSize])
    else:
        r2 = r1
    return [segments, r1, r2]

# TC are the tapered capsule, they are in tuple of c1, c2, r1, r2
# @torch.compile()
def solve_one_four(tc1, tc2):
    d1 = tc1[1] - tc1[0]
    d2 = tc2[1] - tc2[0]
    d12 = tc2[0] - tc1[0]
    rd1 = tc1[3] - tc1[2]
    rd2 = tc2[3] - tc2[2]

    # solve when t=0, u is unknown
    # a = D1
    # b = R12
    # c = A1
    # e = D2
    # f = A2
    # g = D12
    D1 = torch.linalg.vecdot(d1, d1)
    D2 = torch.linalg.vecdot(d2, d2)
    D12 = torch.linalg.vecdot(d12, d12)
    R12 = torch.linalg.vecdot(d1, d2)
    A1 = torch.linalg.vecdot(d1, d12)
    A2 = torch.linalg.vecdot(d2, d12)

    def inner_fcn(D1, D2, D12, R12, A1, A2, rd, out=None):
        # common for case 1 & 2
        inner = (D2 - rd ** 2)
        ap = D2 * inner

        # case 1
        f_1 = A2
        g_1 = D12
        bp_1 = 2 * f_1 * inner
        cp_1 = f_1 ** 2 - rd ** 2 * g_1

        # case 2
        f_2 = A2 - R12
        g_2 = D12 - 2 * A1 + D1
        bp_2 = 2 * f_2 * inner
        cp_2 = f_2 ** 2 - rd ** 2 * g_2

        out_u = torch.empty_like(ap[..., None].expand(ap.shape + (4,))) if out is None else out
        stableQuadraticRoots(ap, bp_1, cp_1, out=out_u[...,:2], invalid_fill=-100)
        stableQuadraticRoots(ap, bp_2, cp_2, out=out_u[...,2:], invalid_fill=-100)
        return out_u
    
    def obj(t, u):
        return torch.norm(t.unsqueeze(-1)*d1.unsqueeze(-2) - u.unsqueeze(-1)*d2.unsqueeze(-2) - d12.unsqueeze(-2), dim=-1) -t*rd1.unsqueeze(-1) - u*rd2.unsqueeze(-1)
    
    # Make outputs and cases 6-9
    u_test = torch.empty_like(tc1[2].unsqueeze(-1).expand(tc1[2].shape + (4 + 4 + 3 + 4,)))
    t_test = torch.empty_like(u_test)
    u_test[...,4:6] = 0
    u_test[...,6:8] = 1
    u_test[...,11:13] = 0
    u_test[...,13:15] = 1
    t_test[...,0:2] = 0
    t_test[...,2:4] = 1
    t_test[...,11:14:2] = 0
    t_test[...,12:15:2] = 1

    # case 1-4
    t = inner_fcn(D2, D1, D12, R12, -A2, -A1, rd1, out=t_test[...,4:8])
    u = inner_fcn(D1, D2, D12, R12, A1, A2, rd2, out=u_test[...,0:4])
    # print(u, t)

    # case 5
    sol = solve_five(D1, D2, D12, R12, A1, A2, rd1, rd2, tout=t_test[...,8:11], uout=u_test[...,8:11])
    # print(sol)
    # u = torch.cat([u, sol[1]], dim=-1)
    # t = torch.cat([t, sol[0]], dim=-1)
    # valid_u = torch.logical_and(u >= 0, u <= 1).detach()
    # valid_t = torch.logical_and(t >= 0, t <= 1).detach()
    # valid_sol_t = torch.logical_and(sol[0] >= 0, sol[0] <= 1)
    # valid_sol_u = torch.logical_and(sol[1] >= 0, sol[1] <= 1)
    # valid_sol = torch.logical_and(valid_sol_t, valid_sol_u)
    # sol[0][~valid_sol] = 0
    # sol[1][~valid_sol] = 0
    
    # u[~valid_u] = torch.nan
    # t[~valid_t] = torch.nan
    # tu_const = torch.zeros_like(u)
    # tu_const[...,2:] = 1
    # tu_const2 = torch.zeros_like(u)
    # tu_const2[...,1:4:2] = 1
    # u_test = torch.cat([u, tu_const, sol[1], tu_const], dim=-1)
    # t_test = torch.cat([tu_const, t, sol[0], tu_const2], dim=-1)
    obj_test = obj(t_test, u_test)

    # valid_mask = torch.cat([valid_u, valid_t, valid_sol, torch.ones_like(valid_t)], dim=-1)
    invalid_mask_u = torch.logical_or(u_test < 0, u_test > 1)
    invalid_mask_t = torch.logical_or(t_test < 0, t_test > 1)
    invalid_mask = torch.logical_or(invalid_mask_t, invalid_mask_u)

    # print(u_test, t_test)

    obj_test[invalid_mask] = torch.inf
    out = torch.min(obj_test, dim=-1)
    ind = out.indices
    # print(out.indices)
    out = out.values
    
    return out - tc1[2] - tc2[2], ind


def solve_five(D1, D2, D12, R12, A1, A2, r1, r2, tout=None, uout=None,
        eps: float = 1e-8,):

    tout = torch.empty_like(r1[...,None].expand(r1.shape + (3,))) if tout is None else tout
    uout = torch.empty_like(r1[...,None].expand(r1.shape + (3,))) if uout is None else uout
    tout.fill_(-100)
    uout.fill_(-100)

    sol = solve_five_prime(D1, D2, D12, R12, A1, A2, r1, r2, tout=tout[...,2], uout=uout[...,2], invalid_fill=-100)
    solve = ~sol[2]

    D1 = D1[solve]
    D2 = D2[solve]
    D12 = D12[solve]
    R12 = R12[solve]
    A1 = A1[solve]
    A2 = A2[solve]
    r1 = r1[solve]
    r2 = r2[solve]
    # tout1 = tout[...,:2][solve]
    # uout1 = uout[...,:2][solve]

    a = r1*R12 + D1*r2
    b = A2*r1 + A1*r2
    c = D2*r1 + R12*r2
    r1g = torch.abs(r1) > torch.abs(r2)
    r2g = ~r1g

    dfunc = lambda DR, r, m: DR ** 2 - D1[m] * r ** 2
    efunc = lambda D, r, m: 2 * R12[m] * (D - r ** 2)
    ffunc = lambda Ac, DR, r, m: 2 * (Ac * DR - A1[m] * r ** 2)
    gfunc = lambda Ac, DR, r, m: 2 * (Ac * DR - A2[m] * r ** 2)
    hfunc = lambda DR, r, m: DR ** 2 - D2[m] * r ** 2
    ifunc = lambda A, r, m: A ** 2 - D12[m] * r ** 2

    d = torch.empty_like(a)
    e = torch.empty_like(a)
    f = torch.empty_like(a)
    g = torch.empty_like(a)
    h = torch.empty_like(a)
    i = torch.empty_like(a)

    d[r1g] = dfunc(D1[r1g], r1[r1g], r1g)
    e[r1g] = efunc(D1[r1g], r1[r1g], r1g)
    f[r1g] = ffunc(A1[r1g], D1[r1g], r1[r1g], r1g)
    g[r1g] = gfunc(A1[r1g], R12[r1g], r1[r1g], r1g)
    h[r1g] = hfunc(R12[r1g], r1[r1g], r1g)
    i[r1g] = ifunc(A1[r1g], r1[r1g], r1g)

    d[r2g] = dfunc(R12[r2g], r2[r2g], r2g)
    e[r2g] = efunc(D2[r2g], r2[r2g], r2g)
    f[r2g] = ffunc(A2[r2g], R12[r2g], r2[r2g], r2g)
    g[r2g] = gfunc(A2[r2g], D2[r2g], r2[r2g], r2g)
    h[r2g] = hfunc(D2[r2g], r2[r2g], r2g)
    i[r2g] = ifunc(A2[r2g], r2[r2g], r2g)

    j = a/(c + eps)
    k = b/(c + eps)
    ap = d-j*e + h*j**2
    bp = k*e -f+j*g - 2*j*k*h
    cp = i + h*k**2 - k*g

    tout[...,:2][solve] = stableQuadraticRoots(ap, bp, cp, invalid_fill=-100)
    uout[...,:2][solve] = ((a[...,None]*tout[...,:2][solve]-b[...,None])/(c[...,None] + eps))
    # print("here", uout, tout)
    # tout2 = torch.empty_like(a[...,None].expand(a.shape + (3,))) if tout is None else tout
    # uout2 = torch.empty_like(a[...,None].expand(a.shape + (3,))) if uout is None else uout
    # tout2.fill_(-100)
    # uout2.fill_(-100)
    # tout2[..., :2][~sol[2]] = tout1[~sol[2]]
    # uout2[..., :2][~sol[2]] = uout1[~sol[2]]
    # tout2[...,2] = sol[0]
    # uout2[...,2] = sol[1]
    return tout, uout


def solve_five_prime(D1, D2, D12, R12, A1, A2, r1, r2, tout=None, uout=None,
        eps: float = 1e-8,
        invalid_fill: float = torch.nan,):
    tout = torch.empty_like(r1) if tout is None else tout
    uout = torch.empty_like(r1) if uout is None else uout
    # tout.fill_(invalid_fill)
    # uout.fill_(invalid_fill)
    zero_taper = torch.logical_and(torch.abs(r1) <= eps, torch.abs(r2) <= eps)
    tout[zero_taper] = (A1[zero_taper]*D2[zero_taper] - A2[zero_taper]*R12[zero_taper])/(D1[zero_taper]*D2[zero_taper] - R12[zero_taper]**2)
    uout[zero_taper] = (tout[zero_taper]*D1[zero_taper] - A1[zero_taper])/R12[zero_taper]
    # print("here", uout, tout)
    return tout, uout, zero_taper


from torch import Tensor
def stableQuadraticRoots(
        a: Tensor,
        b: Tensor,
        c: Tensor,
        # imaginaryTolerance = 1e-8,
        imaginaryTolerance = IMAGINARY_TOLERANCE,
        # imaginaryTolerance = None,
        eps: float = 1e-8,
        out: Tensor = None,
        invalid_fill: float = torch.nan,
    ) -> Tensor:
    out = torch.empty_like(a[...,None].expand(a.shape + (2,))) if out is None else out
    out.fill_(invalid_fill)

    # if discriminant is negative, ignore those and have them return nan
    discriminant = b ** 2 - 4 * a * c
    if imaginaryTolerance is not None:
        # condition = torch.logical_and(discriminant < 0, discriminant > -imaginaryTolerance)
        # discriminant = torch.where(condition, 0, discriminant)
        discriminant = discriminant + imaginaryTolerance
    # discriminant = torch.where(torch.abs(discriminant) < imaginaryTolerance, 0, discriminant)
    non_nan_mask = discriminant >= 0

    down_a = a[non_nan_mask]
    down_b = b[non_nan_mask]
    down_c = c[non_nan_mask]
    discriminant_sign = torch.sign((down_b<0).to(a.dtype)-0.5)
    down_discriminant = discriminant_sign * torch.sqrt(discriminant[non_nan_mask])

    out[...,0][non_nan_mask] = (-down_b + down_discriminant) / (2 * down_a + eps)
    out[...,1][non_nan_mask] = (2 * down_c) / (-down_b + down_discriminant + eps)

    return out


# call main code after initializing all fns
if __name__ == '__main__':
    main()