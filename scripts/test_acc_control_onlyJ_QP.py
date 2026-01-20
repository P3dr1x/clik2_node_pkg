#!/usr/bin/env python3
"""test_acc_control_onlyJ_QP.py

Prototipo Python per controllo in accelerazione con QP usando SOLO il Jacobiano
classico (geometrico) dell'end-effector.

Obiettivo (come da J_Red_QP_instructions.md):
    min_qdd || J*qdd - vdot_des ||^2
    vdot_des = vdot_ref + Kp*e + Kd*e_dot - Jdot*qd

Estensione (seconda versione nelle istruzioni): aggiunge un termine di costo
per minimizzare il momento di reazione tra braccio e base:
    + || H_MR*qdd + n_MR ||^2
con H_MR e n_MR ricavati dal modello del manipolatore con base free-flyer.

La base può essere:
    - FIXED (default)
    - fluttuante in modalità "reactionless_dyn" imponendo conservazione del momento
        (v_base = -Hb^{-1}Hm*qd_arm)

Dipendenze: pinocchio, numpy. OSQP opzionale.
"""
import os
import sys
import time
import argparse
import numpy as np

import pinocchio as pin
import pinocchio.visualize

EE_FRAME = "mobile_wx250s/ee_gripper_link"


ARM_JOINTS = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]

# Giunti gripper (nel wx250s.urdf). Se lockati nel modello usato per H_MR/n_MR,
# l'inerzia del gripper viene considerata come carico rigido senza aggiungere DOF al QP.
GRIPPER_JOINTS = ["gripper", "left_finger", "right_finger"]


def find_urdf_and_pkg_dir():
    """Restituisce (urdf_filename, pkg_dir) per t960a.urdf.
    Prova nel path install della workspace e poi nella workspace sorgente.
    Allineato agli altri script in "scripts/".
    """
    ws_install = "/home/mattia/interbotix_ws/install"
    cand1 = os.path.join(ws_install, "clik2_node_pkg", "share", "clik2_node_pkg", "model", "t960a.urdf")
    pkg1 = os.path.join(ws_install, "clik2_node_pkg", "model")

    here = os.path.dirname(os.path.abspath(__file__))
    cand2 = os.path.normpath(os.path.join(here, "..", "model", "t960a.urdf"))
    pkg2 = os.path.normpath(os.path.join(here, "..", "model"))

    if os.path.exists(cand1):
        return cand1, pkg1
    if os.path.exists(cand2):
        return cand2, pkg2

    raise FileNotFoundError(f"URDF t960a.urdf non trovato. Cercati: {cand1} e {cand2}")


def main():
    # =============================
    # CLI
    # =============================
    parser = argparse.ArgumentParser(description="Simulazione CLIK2 QP (accelerazioni) Jacobiano-only con base fixed o reactionless")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--realtime",
        dest="realtime",
        action="store_true",
        help="Abilita sleep(dt) ad ogni passo per visualizzazione in tempo reale (default).",
    )
    group.add_argument(
        "--no-realtime",
        dest="realtime",
        action="store_false",
        help="Disabilita sleep; esecuzione il più veloce possibile.",
    )
    parser.set_defaults(realtime=True)
    parser.add_argument(
        "--rt-scale",
        type=float,
        default=1.0,
        help="Fattore di scala del tempo reale (1=tempo reale, >1 più lento, <1 più veloce).",
    )

    # Modalità evoluzione base (solo per prototipo)
    # - reactionless_dyn: base si muove in modo coerente con conservazione del momento
    #   (v_base = -Hb^{-1}Hm*qd_arm, come in test_acc_control_QP.py)
    # - fixed: base bloccata
    parser.add_argument(
        "--base-mode",
        choices=["reactionless_dyn", "fixed"],
        default="fixed",
        help=(
            "Come evolve la base: "
            "'reactionless_dyn' impone v_base = -Hb^{-1}Hm*qd_arm (conservazione del momento); "
            "'fixed' tiene la base ferma (default)."
        ),
    )

    parser.add_argument(
        "--k-err-pos",
        type=float,
        default=20.0,
        help="Guadagno feedback errore posizione EE (k_err_pos).",
    )
    parser.add_argument(
        "--k-err-vel",
        type=float,
        default=20.0,
        help="Guadagno feedback errore velocità EE (k_err_vel).",
    )

    parser.add_argument(
        "--w-kin",
        type=float,
        default=10.0,
        help="Peso del costo cinematico (tracking) nel QP.",
    )
    parser.add_argument(
        "--w-dyn",
        type=float,
        default=1.0,
        help="Peso del costo dinamico (reaction-moment) nel QP.",
    )

    parser.add_argument(
        "--dyn-lock-gripper",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Se attivo (default), blocca i giunti del gripper nel modello usato per calcolare H_MR e n_MR. "
            "Così l'inerzia del gripper è inclusa nel costo dinamico senza aggiungere variabili al QP."
        ),
    )

    parser.add_argument(
        "--dyn-disable-gravity",
        action="store_true",
        help=(
            "Se attivo, azzera la gravità nel modello usato per calcolare H_MR e n_MR. "
            "Utile come test diagnostico: n_MR non include il contributo del peso."
        ),
    )

    # Traiettoria desiderata EE: cerchio nel piano X-Z (WORLD)
    parser.add_argument(
        "--circle-radius",
        type=float,
        default=0.10,
        help="Raggio della traiettoria circolare dell'EE [m] (piano X-Z in WORLD).",
    )
    parser.add_argument(
        "--circle-period",
        type=float,
        default=12.0,
        help="Periodo per completare un giro completo della traiettoria [s].",
    )

    # Penalizzazione sul movimento di giunti specifici (su qdd)
    parser.add_argument(
        "--w-qdd-waist",
        type=float,
        default=10,
        help="Peso su qdd del giunto 'waist' (aumenta => meno movimento).",
    )
    parser.add_argument(
        "--w-qdd-forearm-roll",
        type=float,
        default=25,
        help="Peso su qdd del giunto 'forearm_roll' (aumenta => meno movimento).",
    )
    parser.add_argument(
        "--w-qdd-wrist-rotate",
        type=float,
        default=25,
        help="Peso su qdd del giunto 'wrist_rotate' (aumenta => meno movimento).",
    )

    # Solo due casi, come da istruzioni: task 6D e task 3D (solo posizione)
    parser.add_argument(
        "--task",
        choices=["3d", "6d"],
        default="3d",
        help="Task-space da tracciare: '3d' solo posizione (x,y,z), '6d' posizione+orientazione.",
    )
    args = parser.parse_args()

    # =============================
    # CONFIG
    # =============================
    rate_hz = 120.0
    dt = 1.0 / rate_hz

    T_total = float(args.circle_period)  # [s] durata simulazione
    radius = float(args.circle_radius)
    circle_period = float(args.circle_period)
    if circle_period <= 0.0:
        raise ValueError("--circle-period deve essere > 0")

    k_err_pos = float(args.k_err_pos)
    k_err_vel = float(args.k_err_vel)
    Kp = np.eye(6) * k_err_pos
    Kd = np.eye(6) * k_err_vel

    w_kin = float(args.w_kin)
    w_dyn = float(args.w_dyn)
    if w_kin < 0.0 or w_dyn < 0.0:
        raise ValueError("I pesi --w-kin e --w-dyn devono essere >= 0")
    qp_lambda_reg = 1e-2  # regolarizzazione QP per stabilità numerica

    # Penalizzazione su qdd per ridurre movimenti inutili in ridondanza
    w_qdd_by_name = {
        "waist": float(args.w_qdd_waist),
        "forearm_roll": float(args.w_qdd_forearm_roll),
        "wrist_rotate": float(args.w_qdd_wrist_rotate),
    }

    # Modalità task: 3D (solo lineare) oppure 6D
    task_is_3d = (str(args.task).lower() == "3d")

    joint_vel_limit_default = 2.0  # [rad/s] fallback se URDF non ha velocityLimit

    # Limiti box su accelerazioni (qdd) per i giunti del gruppo ARM_JOINTS.
    # Richiesta: 15 rad/s^2 per waist e shoulder, poi +0.5 rad/s^2 per ogni giunto successivo.
    qdd_limit_arm = np.zeros(len(ARM_JOINTS), dtype=float)
    qdd_limit_arm[0] = 15.0  # waist
    qdd_limit_arm[1] = 15.0  # shoulder
    for j in range(2, len(ARM_JOINTS)):
        qdd_limit_arm[j] = 15.0 + 0.5 * float(j - 1)

    # =============================
    # MODELS
    # =============================
    urdf_filename, pkg_dir = find_urdf_and_pkg_dir()
    model, cmodel, vmodel = pin.buildModelsFromUrdf(
        urdf_filename,
        package_dirs=[pkg_dir],
        root_joint=pin.JointModelFreeFlyer(),
    )

    wx_urdf = os.path.join(os.path.dirname(urdf_filename), "wx250s.urdf")
    if not os.path.exists(wx_urdf):
        wx_urdf = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model", "wx250s.urdf"))
    if not os.path.exists(wx_urdf):
        raise FileNotFoundError(f"URDF manipolatore non trovato (wx250s.urdf): {wx_urdf}")
    # Modello manipolatore-only "pieno" (include anche gripper e dita).
    model_man = pin.buildModelFromUrdf(wx_urdf, pin.JointModelFreeFlyer())

    # Modello usato per il costo dinamico: opzionalmente ridotto lockando i giunti del gripper.
    model_dyn = model_man
    if args.dyn_lock_gripper:
        q0_lock = pin.neutral(model_man)
        joints_to_lock = []
        for jn in GRIPPER_JOINTS:
            if model_man.existJointName(jn):
                joints_to_lock.append(model_man.getJointId(jn))
        if len(joints_to_lock) > 0:
            model_dyn = pin.buildReducedModel(model_man, joints_to_lock, q0_lock)

    # Opzione diagnostica: rimuove il contributo della gravità in nonLinearEffects() del costo dinamico.
    if args.dyn_disable_gravity:
        try:
            model_dyn.gravity.linear[:] = 0.0
        except Exception:
            try:
                model_dyn.gravity.linear = np.zeros(3)
            except Exception:
                pass

    data = model.createData()
    data_dyn = model_dyn.createData()

    q = pin.neutral(model)
    v = np.zeros(model.nv)

    if not model.existFrame(EE_FRAME):
        raise RuntimeError(f"Frame EE non trovato: {EE_FRAME}")
    ee_frame_id = model.getFrameId(EE_FRAME)

    if not model.existFrame("mobile_wx250s/base_link"):
        raise RuntimeError("Frame 'mobile_wx250s/base_link' non trovato nel modello completo")
    arm_base_frame_id_full = model.getFrameId("mobile_wx250s/base_link")

    # =============================
    # INDICI GIUNTI
    # =============================
    def arm_joint_indices(mdl, joint_names):
        idx_v = []
        idx_q = []
        for jn in joint_names:
            if not mdl.existJointName(jn):
                raise RuntimeError(f"Joint non trovato nel modello: {jn}")
            jid = mdl.getJointId(jn)
            if int(mdl.joints[jid].nq) != 1:
                raise RuntimeError(f"Joint {jn} ha nq != 1")
            idx_v.append(int(mdl.joints[jid].idx_v))
            idx_q.append(int(mdl.joints[jid].idx_q))
        return idx_v, idx_q

    idx_v_arm, idx_q_arm = arm_joint_indices(model, ARM_JOINTS)
    idx_v_arm_dyn, idx_q_arm_dyn = arm_joint_indices(model_dyn, ARM_JOINTS)
    n_arm = len(ARM_JOINTS)

    # Diagonale pesi su qdd (nell'ordine ARM_JOINTS)
    w_qdd = np.zeros(n_arm)
    for j, jname in enumerate(ARM_JOINTS):
        w_qdd[j] = float(w_qdd_by_name.get(jname, 0.0))
    if np.any(w_qdd < 0.0):
        raise ValueError("I pesi --w-qdd-* devono essere >= 0")

    # Limiti URDF (se presenti)
    have_pos_lim = hasattr(model_dyn, "lowerPositionLimit") and (len(model_dyn.lowerPositionLimit) == model_dyn.nq)
    have_vel_lim = hasattr(model_dyn, "velocityLimit") and (len(model_dyn.velocityLimit) == model_dyn.nv)
    q_lower = np.full(n_arm, -np.inf)
    q_upper = np.full(n_arm, +np.inf)
    v_limit = np.zeros(n_arm)
    for i in range(n_arm):
        if have_pos_lim:
            q_lower[i] = float(model_dyn.lowerPositionLimit[idx_q_arm_dyn[i]])
            q_upper[i] = float(model_dyn.upperPositionLimit[idx_q_arm_dyn[i]])
        if have_vel_lim:
            v_limit[i] = float(model_dyn.velocityLimit[idx_v_arm_dyn[i]])

    # =============================
    # QP SOLVER (OSQP opzionale)
    # =============================
    try:
        import osqp
        import scipy.sparse as sp

        have_osqp = True

        def solve_box_qp(P, g, l, u):
            A = sp.eye(P.shape[0], format="csc")
            prob = osqp.OSQP()
            prob.setup(P=sp.csc_matrix(P), q=g, A=A, l=l, u=u, verbose=False, polish=False, warm_start=True)
            res = prob.solve()
            if res.info.status_val in (1, 2):
                return np.array(res.x).reshape(-1)
            return None

    except Exception:
        have_osqp = False

        def solve_box_qp(P, g, l, u):
            try:
                x = -np.linalg.solve(P, g)
            except np.linalg.LinAlgError:
                x = -np.linalg.lstsq(P, g, rcond=None)[0]
            return np.minimum(np.maximum(x, l), u)

    # =============================
    # MESHCat (opzionale)
    # =============================
    have_viz = False
    have_meshcat_overlays = False
    try:
        viz = pin.visualize.MeshcatVisualizer(model, cmodel, vmodel)
        viz.initViewer(open=True)
        viz.loadViewerModel()
        viz.displayCollisions(False)
        viz.displayVisuals(True)
        viz.display(q)
        have_viz = True

        # Prova a stampare l'URL del viewer (stile test_Jext_pinocchio.py)
        try:
            viewer_url = viz.viewer.url()
            if viewer_url:
                print(f"MeshCat URL: {viewer_url}")
                print("Apri il link sopra se il browser non si è aperto automaticamente.")
        except Exception:
            pass

        # Overlay per visualizzare traiettoria EE (reale/desiderata) direttamente in MeshCat
        try:
            import meshcat.geometry as mg

            viz.viewer["ee_actual"].set_object(
                mg.Sphere(0.01),
                mg.MeshLambertMaterial(color=0x00FF00, opacity=0.7),
            )
            viz.viewer["ee_desired"].set_object(
                mg.Sphere(0.01),
                mg.MeshLambertMaterial(color=0xFF0000, opacity=0.7),
            )
            have_meshcat_overlays = True
        except Exception as e:
            print(f"Overlay traiettoria MeshCat non disponibili: {e}")
        time.sleep(1.0)
    except Exception as e:
        print(f"MeshCat non disponibile: {e}")

    # Stato iniziale EE
    pin.forwardKinematics(model, data, q, v)
    pin.updateFramePlacements(model, data)
    T_we0 = data.oMf[ee_frame_id]
    p0 = np.array(T_we0.translation).reshape(3)
    q_des = pin.Quaternion(np.array(T_we0.rotation))

    # =============================
    # LOG
    # =============================
    t_log = []
    base_p_log = []
    ee_p_log = []
    ee_p_des_log = []
    err_pos_log = []

    # Norma del residuo di reazione: ||H_MR*qdd + n_MR|| (3x1)
    react_moment_norm_log = []

    # Buffer per Jdot (solo Jacobiano classico: colonne braccio)
    Jm_prev = None

    print("\nAvvio simulazione UAM (Jacobiano-only, CLIK2 QP)...")
    print(
        f"task={args.task} | have_osqp={have_osqp} | rate_hz={rate_hz} | "
        f"realtime={args.realtime} | rt_scale={args.rt_scale} | base_mode={args.base_mode} | "
        f"w_kin={w_kin} | w_dyn={w_dyn} | dyn_disable_gravity={args.dyn_disable_gravity}"
    )

    # Traiettoria circolare nel piano X-Z (WORLD), velocità angolare costante
    center = p0.copy()
    center[0] = p0[0] - radius
    omega = (2.0 * np.pi) / circle_period

    # Loop deterministico con numero di passi fissato (stile test_Jext_pinocchio.py)
    N_steps = int(round(T_total / dt))
    draw_hz = min(30.0, rate_hz)
    draw_stride = max(1, int(round(rate_hz / draw_hz)))
    last_draw = time.time()
    ee_path_pts = []
    ee_des_path_pts = []

    for i in range(N_steps + 1):
        loop_start = time.time()
        t = i * dt

        # ========= Stato giunti (uniche variabili controllabili) =========
        qd_arm = v[idx_v_arm]

        # ========= Base: fixed oppure reactionless (conservazione del momento) =========
        Hb_inv_Hm = None
        if args.base_mode == "reactionless_dyn":
            # Dinamica full: M(q) e blocchi Hb/Hm per coupling base-arm
            pin.crba(model, data, q)
            M = np.array(data.M)
            M = (M + M.T) * 0.5
            Hb = M[:6, :6]
            Hm = M[:6, idx_v_arm]
            Hb_inv_Hm = np.linalg.solve(Hb, Hm)

            # v_base = -Hb^{-1}Hm * qd_arm
            v[0:6] = -(Hb_inv_Hm @ qd_arm)
        else:
            v[0:6] = 0.0

        # ========= Traiettoria desiderata in WORLD (pos/vel/acc lineare, orientazione costante) =========
        # Cerchio a velocità angolare costante (nessuno start/stop quintico).
        theta = omega * t
        theta_dot = omega
        theta_ddot = 0.0

        p_des = center.copy()
        p_des[0] = center[0] + radius * np.cos(theta)
        p_des[2] = center[2] + radius * np.sin(theta)

        v_lin_des = np.array(
            [
                -radius * np.sin(theta) * theta_dot,
                0.0,
                radius * np.cos(theta) * theta_dot,
            ]
        )
        a_lin_des = np.array(
            [
                -radius * np.cos(theta) * (theta_dot**2) + (-radius * np.sin(theta) * theta_ddot),
                0.0,
                -radius * np.sin(theta) * (theta_dot**2) + (radius * np.cos(theta) * theta_ddot),
            ]
        )

        omega_des = np.zeros(3)
        alpha_des = np.zeros(3)
        v_des = np.hstack([v_lin_des, omega_des])
        a_des = np.hstack([a_lin_des, alpha_des])



        # ========= Kinematics =========
        pin.forwardKinematics(model, data, q, v)
        pin.updateFramePlacements(model, data)
        T_we = data.oMf[ee_frame_id]
        p_cur = np.array(T_we.translation).reshape(3)
        R_cur = np.array(T_we.rotation)

        # velocità EE (WORLD): Jacobiano geometrico (classico), ordine sempre [lin; ang]
        V_ee = pin.getFrameVelocity(model, data, ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        v_ee_meas = np.hstack([np.array(V_ee.linear).reshape(3), np.array(V_ee.angular).reshape(3)])

        # ========= Errori (come nodo) =========
        e_pos = p_des - p_cur
        R_des = np.array(q_des.matrix())
        R_err = R_des @ R_cur.T
        e_ang = pin.log3(R_err)
        e_pose = np.hstack([e_pos, e_ang])
        e_vel = v_des - v_ee_meas

        fb = (Kp @ e_pose) + (Kd @ e_vel)
        if task_is_3d:
            fb[3:] = 0.0

        # ========= Jacobiano classico =========
        pin.computeJointJacobians(model, data, q)
        J = pin.computeFrameJacobian(model, data, q, ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

        # Colonne braccio del Jacobiano classico
        Jm = J[:, idx_v_arm]

        # Jdot*qd (differenze finite su Jm)
        if Jm_prev is None:
            Jm_prev = Jm.copy()
            Jm_dot_qd = np.zeros(6)
        else:
            Jm_dot = (Jm - Jm_prev) / dt
            Jm_dot_qd = Jm_dot @ qd_arm
            Jm_prev = Jm.copy()

        # Come da istruzioni: vdot_des = vdot_ref + Kp*e + Kd*e_dot - Jdot*qd
        vdot_des = a_des + fb - Jm_dot_qd

        # ========= Termine dinamico (reaction moment) =========
        # Ricava H_MR e n_MR dal modello del manipolatore con base free-flyer.
        # Se --dyn-lock-gripper è attivo, il modello dinamico ha i giunti gripper/dita lockati
        # (l'inerzia dei link del gripper resta inclusa come carico rigido).
        q_dyn = pin.neutral(model_dyn)
        v_dyn = np.zeros(model_dyn.nv)

        T_w_O = data.oMf[arm_base_frame_id_full]
        q_dyn[0:3] = np.array(T_w_O.translation).reshape(3)
        qO = pin.Quaternion(np.array(T_w_O.rotation))
        q_dyn[3:7] = np.array([qO.x, qO.y, qO.z, qO.w])

        for j in range(n_arm):
            q_dyn[idx_q_arm_dyn[j]] = q[idx_q_arm[j]]

        V_O_local = pin.getFrameVelocity(model, data, arm_base_frame_id_full, pin.ReferenceFrame.LOCAL)
        v_dyn[0:3] = np.array(V_O_local.linear).reshape(3)
        v_dyn[3:6] = np.array(V_O_local.angular).reshape(3)
        for j in range(n_arm):
            v_dyn[idx_v_arm_dyn[j]] = v[idx_v_arm[j]]

        pin.normalize(model_dyn, q_dyn)
        pin.crba(model_dyn, data_dyn, q_dyn)
        M_dyn = np.array(data_dyn.M)
        M_dyn = (M_dyn + M_dyn.T) * 0.5
        nle_dyn = np.array(pin.nonLinearEffects(model_dyn, data_dyn, q_dyn, v_dyn)).reshape(-1)

        # Reaction moment: righe angolari del free-flyer (3..5), colonne dei giunti braccio.
        H_mr = M_dyn[3:6, idx_v_arm_dyn]
        n_mr = nle_dyn[3:6]

        # ========= QP: tracking + minimizzazione momento di reazione =========
        if task_is_3d:
            J_task = Jm[0:3, :]
            v_task = vdot_des[0:3]
        else:
            J_task = Jm
            v_task = vdot_des

        P = (w_kin * (J_task.T @ J_task)) + (w_dyn * (H_mr.T @ H_mr))
        P = 0.5 * (P + P.T)
        if np.any(w_qdd > 0.0):
            P = P + np.diag(w_qdd)
        P = P + np.eye(n_arm) * qp_lambda_reg
        g = (w_dyn * (H_mr.T @ n_mr)) - (w_kin * (J_task.T @ v_task))

        # bounds su qdd da limiti vel/pos
        l = np.full(n_arm, -np.inf)
        u = np.full(n_arm, +np.inf)
        dt2 = dt * dt
        for j in range(n_arm):
            vlim = v_limit[j] if (have_vel_lim and v_limit[j] > 0.0) else joint_vel_limit_default
            dq_min = -abs(vlim)
            dq_max = +abs(vlim)
            l_vel = (dq_min - qd_arm[j]) / dt
            u_vel = (dq_max - qd_arm[j]) / dt
            l[j] = max(l[j], l_vel)
            u[j] = min(u[j], u_vel)
            if have_pos_lim and np.isfinite(q_lower[j]) and np.isfinite(q_upper[j]):
                q_i = q[idx_q_arm[j]]
                dq_i = qd_arm[j]
                l_pos = (2.0 * (q_lower[j] - q_i - dt * dq_i)) / dt2
                u_pos = (2.0 * (q_upper[j] - q_i - dt * dq_i)) / dt2
                l[j] = max(l[j], l_pos)
                u[j] = min(u[j], u_pos)

            # Vincoli box su accelerazione (sempre attivi): -qdd_lim <= qdd <= +qdd_lim
            qdd_lim = float(qdd_limit_arm[j])
            l[j] = max(l[j], -abs(qdd_lim))
            u[j] = min(u[j], +abs(qdd_lim))
            if l[j] > u[j]:
                l[j] = u[j]

        qdd_arm = solve_box_qp(P, g, l, u)
        if qdd_arm is None:
            qdd_arm = np.zeros(n_arm)

        # Diagnostica: quanto è "piccolo" il termine di reazione effettivamente ottenuto
        # (indipendente da w_dyn: utile anche per valutare il compromesso col tracking).
        react_moment = (H_mr @ qdd_arm) + n_mr
        react_moment_norm = float(np.linalg.norm(react_moment))

        # ========= Integrazione: comando solo qdd_arm; la base si muove "di riflesso" =========
        # Aggiorna velocità giunti (uniche controllabili)
        qd_arm_next = qd_arm + qdd_arm * dt

        if args.base_mode == "reactionless_dyn":
            if Hb_inv_Hm is None:
                raise RuntimeError("Hb_inv_Hm non inizializzato in modalità reactionless_dyn")
            v_base_next = -(Hb_inv_Hm @ qd_arm_next)
        else:
            v_base_next = np.zeros(6)

        v_next = v.copy()
        v_next[0:6] = v_base_next
        for j in range(n_arm):
            v_next[idx_v_arm[j]] = qd_arm_next[j]

        q = pin.integrate(model, q, v_next * dt)
        pin.normalize(model, q)
        v = v_next

        # ========= LOG =========
        t_log.append(t)
        base_p_log.append(q[0:3].copy())
        ee_p_log.append(p_cur.copy())
        ee_p_des_log.append(p_des.copy())
        err_pos_log.append(float(np.linalg.norm(e_pos)))

        react_moment_norm_log.append(react_moment_norm)

        if (i % int(rate_hz)) == 0:
            print(
                f"t={t:5.2f} | ||e_pos||={np.linalg.norm(e_pos):.4f} | ||qdd||={np.linalg.norm(qdd_arm):.2f} | "
                f"||H_MR*qdd+n_MR||={react_moment_norm:.3f}"
            )

        now = time.time()
        if have_viz and (i % draw_stride) == 0:
            try:
                viz.display(q)
                if have_meshcat_overlays:
                    T_act = np.eye(4)
                    T_act[0:3, 3] = p_cur
                    viz.viewer["ee_actual"].set_transform(T_act)

                    T_des = np.eye(4)
                    T_des[0:3, 3] = p_des
                    viz.viewer["ee_desired"].set_transform(T_des)

                    # Aggiorna le polilinee della traiettoria (decimata alla frequenza di draw)
                    ee_path_pts.append(p_cur.copy())
                    ee_des_path_pts.append(p_des.copy())

                    if len(ee_path_pts) >= 2:
                        import meshcat.geometry as mg

                        pts_act = np.array(ee_path_pts, dtype=float).T
                        pts_des = np.array(ee_des_path_pts, dtype=float).T

                        viz.viewer["ee_path_actual"].set_object(
                            mg.Line(
                                mg.PointsGeometry(pts_act),
                                mg.LineBasicMaterial(color=0x00FF00, linewidth=2),
                            )
                        )
                        viz.viewer["ee_path_desired"].set_object(
                            mg.Line(
                                mg.PointsGeometry(pts_des),
                                mg.LineBasicMaterial(color=0xFF0000, linewidth=2),
                            )
                        )
            except Exception:
                pass
            last_draw = now
        # Sleep opzionale per visualizzazione in tempo reale (stile test_Jext_pinocchio.py)
        if args.realtime:
            dt_sleep = max(0.0, dt * float(args.rt_scale))
            elapsed = time.time() - loop_start
            remaining = dt_sleep - elapsed
            if remaining > 0:
                time.sleep(remaining)

    # =============================
    # PLOT
    # =============================
    try:
        import matplotlib.pyplot as plt

        t_arr = np.array(t_log)
        base_p = np.vstack(base_p_log)
        ee_p = np.vstack(ee_p_log)
        ee_p_des = np.vstack(ee_p_des_log)
        err_arr = np.array(err_pos_log)
        react_moment_norm_arr = np.array(react_moment_norm_log, dtype=float) if len(react_moment_norm_log) else None

        print("\nDifferenza posizione EE fine-inizio:")
        delta_ee = ee_p[-1, :] - ee_p[0, :]
        print(f"  dx = {delta_ee[0]: .6f} m")
        print(f"  dy = {delta_ee[1]: .6f} m")
        print(f"  dz = {delta_ee[2]: .6f} m")

        fig3d = plt.figure()
        ax3d = fig3d.add_subplot(111, projection="3d")
        ax3d.plot(ee_p_des[:, 0], ee_p_des[:, 1], ee_p_des[:, 2], "k--", linewidth=2.0, label="EE desired")
        ax3d.plot(ee_p[:, 0], ee_p[:, 1], ee_p[:, 2], label="EE actual")
        ax3d.scatter(ee_p[0, 0], ee_p[0, 1], ee_p[0, 2], c="g", s=40, label="start")
        ax3d.scatter(ee_p[-1, 0], ee_p[-1, 1], ee_p[-1, 2], c="r", s=40, label="end")
        #ax3d.scatter(ee_p_des[-1, 0], ee_p_des[-1, 1], ee_p_des[-1, 2], c="k", s=30, label="desired end")
        ax3d.set_xlabel("X [m]")
        ax3d.set_ylabel("Y [m]")
        ax3d.set_zlabel("Z [m]")
        ax3d.set_title("Traiettoria EE")
        ax3d.legend(loc="best")

        if react_moment_norm_arr is not None:
            fig_rm, ax_rm = plt.subplots(1, 1)
            ax_rm.plot(t_arr, react_moment_norm_arr, linewidth=2.0)
            ax_rm.set_xlabel("t [s]")
            ax_rm.set_ylabel("||H_MR*qdd + n_MR||")
            ax_rm.set_title("Norma del momento di reazione (proxy)")
            ax_rm.grid(True)

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Plot non disponibile: {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Errore: {e}")
        sys.exit(1)
