# Instructions: QP-based Reaction-Minimizing algorithm for UAM

## Objective

Modify `clik_uam_node.cpp` so to control the arm when in redundant mode. In this case I would like to exploit redundancy by asking the arm to track a 2D trajectory with its end-effector (only x-y-z coordinates, without caring about EE orientation) so that at least it has one degree of redundancy. 

The controller shall compute **joint accelerations** `qdd` by solving, at each control step, the following optimization problem:

$$
\argmin_{\ddot q} \| J_{gen}\ddot q - \dot{v}_{des} \|_{W_{kin}} +  \| H_{M_R}\ddot q + n_{M_R} \|_{W_{dyn}}
$$

where:

* $J_{gen} = J_m - J_b H_b^{-1} H_m$ is the **generalized Jacobian** mapping joint accelerations to EE acceleration (task space reduced as needed), $H_b, H_m$ are submatrices of the inertia matrix of the entire aerial manipulator relative to the base and the manipulator respectively.
* $\dot{v}_{des}$ is the desired task-space acceleration with feedback terms,
* $H_{M_R}$ is the **reaction-moment inertia submatrix** (rows 3–6) of the **manipulator-only inertia matrix**,
* $n_{M_R}$ is the corresponding nonlinear term (Coriolis + centrifugal + gravity contribution, consistent with `H_MR`),
* $λ_W$ = $W_{kin}/W_{dyn}$ is a scalar weight tuning the trade-off between tracking and reaction minimization.

No equality constraints are required; joint limits are added later as box constraints.

---

## Conceptual Differences w.r.t. Cocuzza et al. (2012)

The control algorithm is similar to that presented in the paper in `media/paper-LS.pdf` but with some differences:

* The base is **not free-floating**: it is a drone subject to gravity and control forces.
* Momentum conservation **does not hold**; reaction minimization is **local** and formulated explicitly in acceleration space.
* Only the **reaction moment** (not force) is minimized, consistent with aerial platform disturbance sensitivity.

---

## Prototype

I would like to prototype the resolution in a python script first (you can modify the already existing script `test_acc_control_QP.py`). For now I had problems in solving the problem in the redundant case (I do not understand why just keeping the first three rows of the system of equation does not work). 

Do like in chapter VI.B of `paper-LS`.  

---

## More Implementation Details

In order to compute the manipulator inertia matrix you will need to import the model of the manipulator alone for `pinocchio`. For this step you can copy how it is done in `clik1_node_pkg/clik_uam_node.cpp`.

As QP solver use **OSQP** with the `Eigen-OSQP` C++ wrapper as used in `clik1_node_pkg/clik_uam_node.cpp`.

---

## Consistency Checks (Critical)

Copilot **must verify** the following:

* `H_MR` is extracted from the **manipulator-only** inertia matrix, **not** the full UAM matrix
* The moment rows correspond to the UAV body frame
* `n` matches the same rows and reference frame as `H_MR`
* Units are consistent (Nm, rad/s²)

---

## Debugging Aids

Expected behavior:

* Increasing `λ_W` reduces base angular disturbance
* Tracking error increases gracefully, not explosively

---

## References (for internal consistency)

* Cocuzza et al., *Least-Squares-Based Reaction Control of Space Manipulators*, JGCD 2012 (`paper-LS`)
* Pedrocco et al., *Trajectory Tracking Control of an Aerial Manipulator*, Appl. Sci. 2024 (`paper_MP`)

---

