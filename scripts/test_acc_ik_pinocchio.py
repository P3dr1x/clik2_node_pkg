#!/usr/bin/env python3
"""
Test IK per CLIK2

Requisiti Python: pinocchio, numpy
"""
import os
import sys
import time
import numpy as np

import pinocchio as pin
import pinocchio.visualize

EE_FRAME = "mobile_wx250s/ee_gripper_link"


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
    # Carica modello URDF con base flottante
    urdf_filename, pkg_dir = find_urdf_and_pkg_dir()
    model, cmodel, vmodel = pin.buildModelsFromUrdf(
        urdf_filename,
        package_dirs=[pkg_dir],
        root_joint=pin.JointModelFreeFlyer(),
    )
    # >>> DISATTIVA GRAVITÀ <<<
    model.gravity = pin.Motion.Zero()   # equivalente a gravity = [0,0,0]
    print(f"Gravità: {model.gravity}")

    data = model.createData()
    data_0 = model.createData()

    # Configurazione iniziale: neutra
    q = pin.neutral(model)

    # Verifica frame EE e id (serve anche per la stampa a q neutra)
    if not model.existFrame(EE_FRAME):
        raise RuntimeError(f"Frame EE non trovato: {EE_FRAME}")
    ee_frame_id = model.getFrameId(EE_FRAME)

    # # Imposta una posa non banale della base (per evidenziare la differenza LOCAL vs WORLD)
    # # Traslazione
    # q[0:3] = np.array([0.3, -0.2, 0.4])
    # # Rotazione da RPY (roll, pitch, yaw)
    # R = pin.rpy.rpyToMatrix(0.2, -0.1, 0.5)
    # quat = pin.Quaternion(R)  # XYZW
    # q[3:7] = np.array([quat.x, quat.y, quat.z, quat.w])


    # Visualizzazione con MeshCat (posa drone+braccio)
    try:
        viz = pin.visualize.MeshcatVisualizer(model, cmodel, vmodel)
        # Prova ad aprire il browser automaticamente; se fallisce, resta in background
        viz.initViewer(open=True)

        # Carica il modello
        viz.loadViewerModel()

        # Mostra solo i visual (non le collisioni)
        viz.displayCollisions(False)
        viz.displayVisuals(True)

        # Visualizza la posa corrente
        viz.display(q)

        # Prova a stampare l'URL del viewer
        viewer_url = None
        try:
            viewer_url = viz.viewer.url()
        except Exception:
            pass
        if viewer_url:
            print(f"MeshCat URL: {viewer_url}")
            print("Apri il link sopra se il browser non si è aperto automaticamente.")
        else:
            print("MeshCat: visualizzazione inizializzata e posa mostrata (URL non disponibile).")

        # Mantieni la finestra viva per qualche secondo per permettere il rendering
        time.sleep(5.0)
    except Exception as e:
        print(f"MeshCat non disponibile: {e}")

    rate_hz = 1000.0
    dt = 1.0 / rate_hz
    # Durata complessiva della traiettoria
    T_total = 12.0  # [s]
    # Lunghezza del segmento desiderato (m) lungo l'asse X in WORLD
    segment_length = 0.10 # [m]
    lamb_pinv = 1e-5
    #eps_Hb = 1e-9

    # Logging
    t_log = []
    base_p_log = []  # posizione base
    yaw_log = []
    err_Jdot_log = []   # norma di [err_lin; err_ang]
    err_z1_log = []     # norma di (z1 + Ag_dot*qd)
    h0dot_norm_log = []   # norma di h0_dot (a=0)
    hdot_norm_log = []    # norma di h_dot (a=acc_ff)
    Agdotv_norm_log = []  # norma di Ag_dot_v
    joints_log = []  # angoli giunti del braccio (ultimi nv-6)
    ee_p_log = []    # posizione end-effector

    # Parametri traiettoria circolare (WORLD): piano x-z, orientazione costante
    # Centro scelto per passare da p0 al tempo t=0 sul bordo lungo +X
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    T_we0 = data.oMf[ee_frame_id]
    p0 = np.array(T_we0.translation).reshape(3)

    # --- Stampa matrici d'inerzia e CMM all'istante iniziale per confronto ---
    # try:
    #     M_world0 = pin.crba(model, data, q)
    #     M_world0 = (M_world0 + M_world0.T) * 0.5
    #     Hb0 = M_world0[:6, :6]
    #     # assumiamo che le colonne successive corrispondano ai giunti mobili
    #     m_arm = model.nv - 6 - 3
    #     Hbm0 = M_world0[:6, 6:6+m_arm]
    #     print("--- Istante iniziale: matrici d'inerzia ---")
    #     print("Hb0:\n", Hb0)
    #     print("Hbm0:\n", Hbm0)
    # except Exception as e:
    #     print("Errore calcolo Hb/Hbm iniziali:", e)

    # compute Centroidal Momentum Matrix (CMM) if available and print sub-blocks
    # try:
    #     # proviamo diverse API possibili
    #     try:
    #         pin.computeCentroidalMap(model, data, q)
    #     except Exception:
    #         try:
    #             pin.computeCentroidalMomentum(model, data, q)
    #         except Exception:
    #             pass
    #     Ag = getattr(data, 'Ag', None)
    #     if Ag is None:
    #         # some pinocchio versions store centroidal matrix in data.Ag or data.A
    #         Ag = getattr(data, 'A', None)
    #     if Ag is not None:
    #         print("--- Istante iniziale: Centroidal Momentum Matrix (Ag) ---")
    #         print("Ag shape:", np.shape(Ag))
    #         # submatrici corrispondenti a base e giunti
    #         Ag_base = Ag[:, :6]
    #         Ag_joints = Ag[:, 6:6+m_arm]
    #         print("Ag_base (6x6):\n", Ag_base)
    #         print("Ag_joints (6x{}):\n".format(m_arm), Ag_joints)
    #     else:
    #         print("Centroidal Momentum Matrix non disponibile (data.Ag assente).")
    # except Exception as e:
    #     print("Errore calcolo CMM iniziale:", e)

    # Parametri traiettoria lineare (WORLD): segmento di ampiezza `segment_length` in X
    # con profilo quintico in tempo per avere vel/acc nulle agli estremi.

    t0 = time.time()
    last_draw = t0
    print("\nAvvio loop IK a livello delle accelerazioni...")
    # Solo cinematica inversa accelerazioni (senza feedback K,Kd)
    m_arm = model.nv - 6 - 3  # DoF manipolatore (escludendo base e gripper)
    qdot_arm = np.zeros(m_arm)
    v_full = np.zeros(model.nv)
    acc_ff = np.zeros(model.nv)
    a_zero = np.zeros(model.nv)
    v_base = np.zeros(6)
    print(f"Controller clik2 feed-forward (solo IK) m_arm={m_arm}")

    while True:
        t = time.time() - t0

        # Cinematica e Jacobiani
        pin.forwardKinematics(model, data, q, v_full, acc_ff)
        pin.updateFramePlacements(model, data)

        # pin.forwardKinematics(model, data_0, q, v_full, a_zero)
        # pin.updateFramePlacements(model, data_0)

        # Jacobiano frame EE espresso in LOCAL_WORLD_ALIGNED (righe [lin; ang] usate qui)
        J = pin.computeFrameJacobian(model, data, q, ee_frame_id, pin.ReferenceFrame.WORLD)

        # Posizione end-effector (WORLD)
        p_ee = np.array(data.oMf[ee_frame_id].translation).reshape(3)

        # Partizioni Jacobiano
        Jb = J[:, :6]
        Jm = J[:, 6:6+m_arm]

        # Matrice di momento centroidale (CMM) e partizioni
        pin.computeCentroidalMap(model, data, q)
        Ag = data.Ag
        Ag_b = Ag[:, :6]
        Ag_m = Ag[:, 6:6+m_arm]
        # Calcolo derivata
        pin.computeCentroidalMapTimeVariation(model, data, q, v_full)
        dAg = data.dAg

         # ========= Traiettoria desiderata lineare verticale (pos, vel, acc) =========
        # Usiamo un profilo quintico in posizione per avere vel/acc nulle agli estremi
        if t <= T_total:
            # parametro temporale normalizzato s in [0,1]
            s = t / T_total
            # polinomio quintico: s(t) in [0,1]
            s3 = s**3
            s4 = s**4
            s5 = s**5
            s_pos = 10.0 * s3 - 15.0 * s4 + 6.0 * s5
            s_vel = (30.0 * s**2 - 60.0 * s**3 + 30.0 * s**4) / T_total
            s_acc = (60.0 * s - 180.0 * s**2 + 120.0 * s**3) / (T_total**2)

            p_des = p0.copy()
            # segmento di ampiezza `segment_length` lungo X in WORLD
            p_des[0] += segment_length * s_pos

            v_lin_des = np.zeros(3)
            v_lin_des[0] = segment_length * s_vel

            a_lin_des = np.zeros(3)
            a_lin_des[0] = segment_length * s_acc
            omega_des = np.zeros(3)
            alpha_des = np.zeros(3)
        else:
            # Mantieni posizione finale
            p_des = p0.copy()
            p_des[0] += segment_length
            v_lin_des = np.zeros(3)
            a_lin_des = np.zeros(3)
            omega_des = np.zeros(3)
            alpha_des = np.zeros(3)

        acc_ee_des = np.hstack([a_lin_des, alpha_des])

        Ab_inv_Am = np.linalg.solve(Ag_b, Ag_m)

        # ========= Termine z1 (solo feed-forward) =========
        pin.computeCentroidalMomentumTimeVariation(model, data, q, v_full, a_zero)
        h0_dot = data.dhg  # centroidal momentum time derivative with a=0
        h0dot_norm_log.append(np.linalg.norm(h0_dot))
        # Calcola anche Ag_dot * v_full (qd) per confronto: dh = Ag_dot * v + Ag * a
        pin.computeCentroidalMomentumTimeVariation(model, data, q, v_full, acc_ff)
        h_dot = data.dhg
        hdot_norm_log.append(np.linalg.norm(h_dot))

        # Stima Ag_dot * v_full = dAg @ v_full
        Ag_dot_v = dAg @ v_full
        Agdotv_norm_log.append(np.linalg.norm(Ag_dot_v))
        z1 = - h0_dot
        #z1 = - Ag_dot_v

        # errore tra z1 e -Ag_dot*v_full: z1 + Ag_dot*v_full dovrebbe essere ~0
        err_z1 = - h0_dot + Ag_dot_v
        err_z1_log.append(np.linalg.norm(err_z1))

        # ========= Accelerazione nominale xdd0 =========
        pin.forwardKinematics(model, data, q, v_full, a_zero)
        pin.updateFramePlacements(model, data)
        a_ee = pin.getFrameClassicalAcceleration(model, data, ee_frame_id, pin.ReferenceFrame.WORLD)
        a_ee_sp_0 = pin.getFrameAcceleration(model, data, ee_frame_id, pin.ReferenceFrame.WORLD)
        xdd0 = np.hstack([a_ee_sp_0.linear, a_ee_sp_0.angular])

        try:
            # Aggiorna le derivate dei Jacobiani delle giunzioni
            pin.computeJointJacobiansTimeVariation(model, data, q, v_full)

            # Jacobiano tempo‑variante del frame EE nel WORLD
            Jdot_ee = pin.getFrameJacobianTimeVariation(
                model, data, ee_frame_id, pin.ReferenceFrame.WORLD
            )

            # Termine Jdot * v
            a_Jdot_v = Jdot_ee @ v_full

            # Confronto (lineare + angolare) tra accelerazione spaziale e Jdot*v
            err_lin = a_ee_sp_0.linear - a_Jdot_v[:3]
            err_ang = a_ee_sp_0.angular - a_Jdot_v[3:]
            err_vec = np.hstack([err_lin, err_ang])
            err_Jdot_log.append(np.linalg.norm(err_vec))

        except Exception as e:
            print("Errore nel calcolo di Jdot o nella verifica a_ee_sp_0:", e)

        # ========= Termine z2 (solo feed-forward) =========
        z2_ff = acc_ee_des - xdd0

        # ========= Sistema aumentato =========
        Aug = np.vstack([
            np.hstack([Ag_b, Ag_m]),
            np.hstack([Jb, Jm])
        ])
        rhs_ff = np.hstack([z1, z2_ff])
        try:
            acc_ff[:6+m_arm] = np.linalg.lstsq(Aug, rhs_ff, rcond=None)[0]
        except Exception as e:
            print("Errore risoluzione Aug:", e)
            break
        qdd_total = acc_ff[6:6+m_arm]

        # ========= Integrazione =========
        q = pin.integrate(model, q, v_full * dt)

        qdot_arm += qdd_total * dt
        qdot_arm = np.clip(qdot_arm, -2.0, 2.0)
        #Hb_inv_Hbm = np.linalg.solve(Hb, Hbm)
        #v_base = - Ab_inv_Am @ qdot_arm  # aggiorna con nuova velocità giunti
        v_base += acc_ff[:6] * dt  # integrazione euler su v_base
        v_full[:6] = v_base
        v_full[6:6+m_arm] = qdot_arm

        # ========= Diagnostica sintetica =========
        if int(t * 10) % 10 == 0:
            # spostamento verticale rispetto alla posizione iniziale (piano z)
            dz = p_ee[2] - p0[2]
            print(f"t={t:5.2f} | dz={dz:.4f}")
            #print(f"  h_0  = {h0_dot}")
            #print(f"  h_diff  = {h_dot-h0_dot}")
            #print(f"  acc_ff = {acc_ff}")

        # Visualizzazione a rate fisso
        now = time.time()
        if 'viz' in locals() and (now - last_draw) >= dt:
            try:
                viz.display(q)
            except Exception:
                pass
            last_draw = now

        # Log configurazioni
        T_wb = data.oMi[1]
        base_p = np.array(T_wb.translation).reshape(3)
        R_wb = T_wb.rotation
        # estrai yaw da R_wb (Z-Y-X yaw-pitch-roll)
        yaw = np.arctan2(R_wb[1,0], R_wb[0,0])
        t_log.append(t)
        base_p_log.append(base_p)
        yaw_log.append(yaw)
        joints_log.append(np.array(q[6:6+m_arm]))  # solo i primi 6 giunti del braccio
        ee_p_log.append(p_ee)

        # Esci
        if t >= T_total:
            break
        # Rispetta dt
        time.sleep(max(0.0, dt - (time.time() - now)))

    # Plot traiettoria EE (3D) e, opzionale, componenti x,y,z vs tempo
    try:
        import matplotlib.pyplot as plt
        t_arr = np.array(t_log)
        ee_p_arr = np.vstack(ee_p_log)

        # Differenza finale in posizione EE rispetto all'inizio
        delta_ee = ee_p_arr[-1, :] - ee_p_arr[0, :]
        print("\nDifferenza posizione EE fine-inizio:")
        print(f"  dx = {delta_ee[0]: .6f} m")
        print(f"  dy = {delta_ee[1]: .6f} m")
        print(f"  dz = {delta_ee[2]: .6f} m")
        joints_arr = np.vstack(joints_log)

        # Plot norma errore ||[err_lin; err_ang]|| nel tempo
        if len(err_Jdot_log) == len(t_arr):
            err_arr = np.array(err_Jdot_log)
            plt.figure()
            plt.plot(t_arr, err_arr)
            plt.xlabel('t [s]')
            plt.ylabel('||err_Jdot||')
            plt.title('Norma errore a_ee_sp_0 vs Jdot*v')

        # Plot norma errore ||z1 + Ag_dot*v|| nel tempo
        if len(err_z1_log) == len(t_arr):
            errz_arr = np.array(err_z1_log)
            plt.figure()
            plt.plot(t_arr, errz_arr)
            plt.xlabel('t [s]')
            plt.ylabel('||- h0_dot + Ag_dot*v||')
            plt.title('Differenza h0_dot - Ag_dot*v')

        # Plot norme di h0_dot, h_dot e Ag_dot_v nella stessa finestra
        if (len(h0dot_norm_log) == len(t_arr)
                and len(hdot_norm_log) == len(t_arr)
                and len(Agdotv_norm_log) == len(t_arr)):
            h0dot_arr = np.array(h0dot_norm_log)
            hdot_arr = np.array(hdot_norm_log)
            Agdotv_arr = np.array(Agdotv_norm_log)
            plt.figure()
            plt.plot(t_arr, h0dot_arr, label='||h0_dot|| (a=0)')
            plt.plot(t_arr, hdot_arr, label='||h_dot|| (a=acc_ff)')
            plt.plot(t_arr, Agdotv_arr, label='||Ag_dot*v||')
            plt.xlabel('t [s]')
            plt.ylabel('norma')
            plt.title('Norme di h0_dot, h_dot e Ag_dot*v')
            plt.legend(loc='best')

        # Figura 1: traiettoria 3D dell'end-effector (WORLD)
        fig3d = plt.figure()
        ax3d = fig3d.add_subplot(111, projection='3d')
        ax3d.plot(ee_p_arr[:, 0], ee_p_arr[:, 1], ee_p_arr[:, 2], label='EE path (WORLD)')
        ax3d.scatter(ee_p_arr[0, 0], ee_p_arr[0, 1], ee_p_arr[0, 2], c='g', s=40, label='start')
        ax3d.scatter(ee_p_arr[-1, 0], ee_p_arr[-1, 1], ee_p_arr[-1, 2], c='r', s=40, label='end')
        ax3d.set_xlabel('X [m]')
        ax3d.set_ylabel('Y [m]')
        ax3d.set_zlabel('Z [m]')
        ax3d.set_title('Traiettoria EE in WORLD')
        ax3d.legend(loc='best')
        try:
            # Aspetto isometrico
            ranges = ee_p_arr.max(axis=0) - ee_p_arr.min(axis=0)
            max_range = max(ranges.max(), 1e-6)
            x_mid = (ee_p_arr[:, 0].max() + ee_p_arr[:, 0].min()) / 2.0
            y_mid = (ee_p_arr[:, 1].max() + ee_p_arr[:, 1].min()) / 2.0
            z_mid = (ee_p_arr[:, 2].max() + ee_p_arr[:, 2].min()) / 2.0
            ax3d.set_box_aspect((1, 1, 1))
            ax3d.set_xlim(x_mid - max_range / 2, x_mid + max_range / 2)
            ax3d.set_ylim(y_mid - max_range / 2, y_mid + max_range / 2)
            ax3d.set_zlim(z_mid - max_range / 2, z_mid + max_range / 2)
        except Exception:
            pass

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
