# Instructions: QP-based Reaction-Minimizing algorithm for UAM

## Objective

Modify `clik_uam_node.cpp` to replace the current closed-form / pseudoinverse CLIK with a **Quadratic Programming (QP)** formulation suitable for an **aerial manipulator (UAM)** (always at acceleration level).

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

## High-Level Control Flow (per control cycle)

1. Read current state:

   * Joint position `q`, velocity `qd`
   * Base pose and twist (as already done in `clik_uam_node`)

2. Compute kinematics:

   * End-effector pose, velocity
   * `Jgen`, `Jgen_dot`

3. Build desired task acceleration:

   ```
   vd_des = xdd_ref
            + Kp * (x_ref - x)
            + Kd * (xd_ref - xd)
            - Jgen_dot * qd
   ```

4. Compute manipulator dynamics (manipulator only):

   * Inertia matrix `H_m`
   * Nonlinear term `n_m = C(q,qd) + g(q)`

5. Extract reaction-related terms:

   * `H_MR = H_m.block(3, 0, 3, nq)`
   * `n_mr     = n_m.segment(3, 3)`

6. Assemble and solve QP for `qdd`. Implement both velocity and position joint limits. Regarding position joint limits implement the constraints as how you think it's better.

7. Integrate/forward `qdd` to low-level joint controller

---

## Detailed Implementation Instructions

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

