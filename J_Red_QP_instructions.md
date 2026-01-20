# Instructions: QP-based Reaction-Minimizing algorithm for UAM

## Objective

Modify `clik_uam_node.cpp` so to control the arm when in redundant mode. In this case I would like to exploit redundancy by asking the arm to track a 2D trajectory with its end-effector (only x-y-z coordinates, without caring about EE orientation) so that at least it has one degree of redundancy. 

The controller shall compute **joint accelerations** `qdd` by solving, at each control step, the following optimization problem:

$$
\argmin_{\ddot q} \| J\ddot q - \dot{v}_{des} \|_{W_{kin}}
$$

where:

* $J$ is the **classic Jacobian** mapping joint accelerations to EE acceleration (task space reduced as needed),
* $\dot{v}_{des}$ is the desired task-space acceleration with feedback terms, $\dot{v}_{des}=\dot{v}_{ref}+K_P e + K_D \dot{e} - \dot{J}\dot{q}$


No equality constraints are required; joint limits are added later as box constraints.

---

## Conceptual Differences w.r.t. Cocuzza et al. (2012)

The control algorithm is similar to that presented in the paper in `media/paper-LS.pdf` but with some differences:

* The base is **not free-floating**: it is a drone subject to gravity and control forces.
* Momentum conservation **does not hold**; reaction minimization is **local** and formulated explicitly in acceleration space.
* Only the **reaction moment** (not force) is minimized, consistent with aerial platform disturbance sensitivity.

---

## Prototype

I would like to prototype the resolution in a python script first (you can modify the already existing script `test_acc_control_onlyJ_QP.py`). For now I had problems in solving the problem in the redundant case (I do not understand why just keeping the first three rows of the system of equation does not work). 

Do like in chapter VI.B of `paper-LS`.  

---

## More Implementation Details

In order to compute the manipulator inertia matrix you will need to import the model of the manipulator alone for `pinocchio`. For this step you can copy how it is done in `clik1_node_pkg/clik_uam_node.cpp`.

As QP solver use **OSQP** with the `Eigen-OSQP` C++ wrapper as used in `clik1_node_pkg/clik_uam_node.cpp`.


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

