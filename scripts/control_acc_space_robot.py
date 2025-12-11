#!/usr/bin/env python3
"""
Controllo a livello delle accelerazioni per manipolatore su base flottante

Dipendenze: pinocchio, numpy, (opzionale) matplotlib, meshcat

Idea generale (ispirata alla letteratura su robot spaziali):
- Stato generalizzato: q, v per base flottante (6 DoF) + giunti del braccio
- Dinamica centroidale: h_dot = Ag(q) * a + dAg(q, v) * v
- Task EE a livello accelerazioni: xdd = J(q) * a + Jdot(q, v) * v
- Si risolve un sistema aumentato per a = [a_base; a_joints]
  con vincoli su h_dot (p.es. regolazione verso 0 o verso un riferimento) e
  xdd_des per l'end-effector.

Questo script implementa un controllo feed-forward basato su:
- Regolazione del momento centroidale verso 0: h_dot_des = - dAg * v
- Accelerazione desiderata del frame EE: xdd_des da profilo quintico su posizione

Nota: non sono inclusi feedback K, Kd; il focus è mostrare l'algoritmo.

Modalità disponibili via CLI:
- "augmented" (default): risolve il sistema aumentato su tutti i DoF [Ag; J].
- "freefloating": non comanda attivamente la base; usa il vincolo di momentum
    per ricavare a_b da a_m: a_b = -Ag_b^{-1}(Ag_m a_m + dAg v), e risolve solo
    il task EE per a_m con eliminazione della base.
"""

import os
import time
import sys
import numpy as np
import argparse

import pinocchio as pin
import pinocchio.visualize
try:
    import meshcat.geometry as g
except Exception:
    g = None

EE_FRAME = "mobile_wx250s/ee_gripper_link"


def find_urdf_and_pkg_dir():
    """Restituisce (urdf_filename, pkg_dir) per t960a.urdf.
    Cerca prima nell'install della ws e poi nella sorgente.
    Allineato allo schema di `test_acc_ik_pinocchio.py`.
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


def quintic_profile(t, T, p0, p1):
    """Profili s, s_dot, s_ddot per interpolazione quintica tra p0 e p1 in [0, T]."""
    if t <= 0:
        return p0, 0.0, 0.0
    if t >= T:
        return p1, 0.0, 0.0
    s = t / T
    s2 = s * s
    s3 = s2 * s
    s4 = s3 * s
    s5 = s4 * s
    s_pos = 10.0 * s3 - 15.0 * s4 + 6.0 * s5
    s_vel = (30.0 * s2 - 60.0 * s3 + 30.0 * s4) / T
    s_acc = (60.0 * s - 180.0 * s2 + 120.0 * s3) / (T * T)
    p = p0 + (p1 - p0) * s_pos
    v = (p1 - p0) * s_vel
    a = (p1 - p0) * s_acc
    return p, v, a


def main():
    parser = argparse.ArgumentParser(description="Controllo accelerazioni per manipolatore su base flottante")
    parser.add_argument("--mode", choices=["augmented", "freefloating"], default="augmented",
                        help="augmented: risoluzione su tutti i DoF; freefloating: elimina la base via momentum.")
    parser.add_argument("--no-browser", action="store_true",
                        help="Non apre automaticamente il browser MeshCat; stampa comunque l'URL.")
    parser.add_argument("--realtime", action="store_true",
                        help="Abilita sleep(dt) ad ogni passo per visualizzazione in tempo reale.")
    parser.add_argument("--rt-scale", type=float, default=1.0,
                        help="Fattore di scala del tempo reale (1=tempo reale, >1 più lento, <1 più veloce).")
    parser.add_argument("--traj", choices=["line", "circle"], default="line",
                        help="Seleziona la traiettoria desiderata dell'EE: line o circle.")
    parser.add_argument("--radius", type=float, default=0.08,
                        help="Raggio [m] per traiettoria circolare (quando --traj=circle).")
    parser.add_argument("--revs", type=float, default=1.0,
                        help="Numero di giri completi per traiettoria circolare.")
    parser.add_argument("--no-ee-ff", action="store_true",
                        help="Se impostato, DISABILITA il feedforward di accelerazione EE (lascia solo il feedback di posa).")
    args = parser.parse_args()

    # Carica modello URDF con base flottante
    urdf_filename, pkg_dir = find_urdf_and_pkg_dir()
    model, cmodel, vmodel = pin.buildModelsFromUrdf(
        urdf_filename,
        package_dirs=[pkg_dir],
        root_joint=pin.JointModelFreeFlyer(),
    )

    # Imposta gravità a zero (ambiente spaziale / base flottante)
    model.gravity = pin.Motion.Zero()
    data = model.createData()

    # Configuration neutra e stati
    q = pin.neutral(model)
    nv = model.nv
    v = np.zeros(nv)
    a = np.zeros(nv)

    # Identificatori frame e dimensioni
    if not model.existFrame(EE_FRAME):
        raise RuntimeError(f"Frame EE non trovato: {EE_FRAME}")
    ee_frame_id = model.getFrameId(EE_FRAME)

    # Visualizzazione opzionale via MeshCat
    viz = None
    try:
        viz = pin.visualize.MeshcatVisualizer(model, cmodel, vmodel)
        viz.initViewer(open=not args.no_browser)
        viz.loadViewerModel()
        viz.displayCollisions(False)
        viz.displayVisuals(True)
        viz.display(q)
        # Prova a stampare l'URL del viewer
        viewer_url = None
        try:
            viewer_url = viz.viewer.url()
        except Exception:
            pass
        if viewer_url:
            print(f"MeshCat viewer inizializzato. URL: {viewer_url}")
            if args.no_browser:
                print("Apri l'URL sopra nel tuo browser per vedere la scena.")
        else:
            print("MeshCat viewer inizializzato (URL non disponibile).")
        # Oggetti ausiliari per capire il moto: assi base ed EE, e traiettorie
        if g is not None:
            try:
                viz.viewer["frames/base"].set_object(g.AxisHelper(0.15))
                viz.viewer["frames/ee"].set_object(g.AxisHelper(0.10))
                viz.viewer["plots/ee_path"].set_object(
                    g.Line(g.PointsGeometry(np.zeros((3, 2))), g.LineMaterial(color=0xff0000, linewidth=2))
                )
                viz.viewer["plots/ee_des_path"].set_object(
                    g.Line(g.PointsGeometry(np.zeros((3, 2))), g.LineMaterial(color=0x00aa00, linewidth=2))
                )
            except Exception:
                pass
    except Exception as e:
        print(f"MeshCat non disponibile: {e}")
    
    time.sleep(3.0)  # attesa per viewer

    # Preparazione traiettoria desiderata EE (WORLD): spostamento su X
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    T_we0 = data.oMf[ee_frame_id]
    p0 = np.array(T_we0.translation).reshape(3)
    # Centro per traiettoria circolare in piano XZ, tale da iniziare in p0
    # p_des(0) = p0 se c = p0 - [R, 0, 0] e theta0=0
    circle_center = p0.copy()
    circle_center[0] = p0[0] - max(args.radius, 1e-6)
    circle_center[1] = p0[1]
    circle_center[2] = p0[2]

    # Parametri controllo
    rate_hz = 100.0 # [Hz]
    dt = 1.0 / rate_hz  # [s]
    # Frequenza di aggiornamento viewer (<= rate di controllo)
    draw_hz = min(30.0, rate_hz)
    draw_stride = max(1, int(round(rate_hz / draw_hz)))
    T_total = 12.0 # [s]
    segment_length = 0.10  # [m]

    # Partizioni DoF: base (6) + manipolatore (include TUTTI i DoF oltre la base)
    # Nota: includiamo anche gli ultimi 3 DoF (gripper/finger) perché contribuiscono a h
    m_arm = nv - 6

    # Log
    t_log = []
    ee_p_log = []
    err_momentum_log = []
    cond_aug_log = []
    ee_path_pts = []
    ee_des_path_pts = []
    p_des_log = []
    h0dot_norm_log = []

    v_lin_des = np.zeros(3)
    a_lin_des = np.zeros(3)
    omega_des = np.zeros(3)
    alpha_des = np.zeros(3)

    # Guadagni feedback su errore di posa e velocità EE (analogo a K/Kd nel nodo C++)
    k_err_pos = 20.0
    k_err_vel = 2.0
    K_ee = k_err_pos * np.eye(6)
    Kd_ee = k_err_vel * np.eye(6)

    print(f"Avvio controllo accelerazioni (mode={args.mode}, feed-forward, EE + momentum)...")
    N_steps = int(round(T_total / dt))

    for i in range(N_steps + 1):  # includi istante finale
        loop_start = time.time()
        t = i * dt

        # Cinematica/spazio
        pin.forwardKinematics(model, data, q, v, a)
        pin.updateFramePlacements(model, data)

        # Jacobiano EE nel WORLD
        J = pin.computeFrameJacobian(model, data, q, ee_frame_id, pin.ReferenceFrame.WORLD)
        Jb = J[:, :6]
        Jm = J[:, 6:]

        # Derivata Jacobiano EE
        pin.computeJointJacobiansTimeVariation(model, data, q, v)
        Jdot = pin.getFrameJacobianTimeVariation(model, data, ee_frame_id, pin.ReferenceFrame.WORLD)

        # Stato corrente EE
        p_ee = np.array(data.oMf[ee_frame_id].translation).reshape(3)

        # Pianificazione desiderata della traiettoria
        if args.traj == "line":
            # Spostamento lungo X con profilo quintico
            p_des, v_des_x, a_des_x = quintic_profile(t, T_total, p0[0], p0[0] + segment_length)
            p_des_vec = p0.copy(); p_des_vec[0] = p_des
            v_lin_des[:] = 0.0
            a_lin_des[:] = 0.0
            v_lin_des[0] = v_des_x
            a_lin_des[0] = a_des_x
        else:
            # Traiettoria circolare in piano XZ: p = c + R [cosθ, 0, sinθ]
            # θ(t) con profilo quintico da 0 a 2π*revs per avere velocità/accelerazione nulle agli estremi
            theta, theta_dot, theta_ddot = quintic_profile(t, T_total, 0.0, 2.0 * np.pi * args.revs)
            R = max(args.radius, 1e-6)
            ct, st = np.cos(theta), np.sin(theta)
            # Posizione desiderata 3D (piano XZ)
            p_des_vec = np.array([circle_center[0] + R * ct,
                                  circle_center[1],
                                  circle_center[2] + R * st])
            # Velocità lineare desiderata per moto circolare parametrico nel piano XZ
            # v = d/dt p = R * ([-sinθ, 0, cosθ] * θ_dot)
            v_lin_des = np.array([
                R * (-st * theta_dot),
                0.0,
                R * ( ct * theta_dot)
            ])
            # Accelerazione lineare desiderata per moto circolare parametrico nel piano XZ
            # a = R * ([-cosθ, 0, -sinθ] * θ_dot^2 + [-sinθ, 0, cosθ] * θ_ddot)
            a_lin_des = np.array([
                R * (-ct * (theta_dot**2) - st * theta_ddot),
                0.0,
                R * (-st * (theta_dot**2) + ct * theta_ddot)
            ])
        # Mantieni EE orizzontale: accelerazione angolare desiderata nulla (orientazione costante)
        # xdd_ff contiene solo il termine di feedforward da traiettoria
        xdd_ff = np.hstack([a_lin_des, alpha_des])

        # Log desiderato per plot offline
        p_des_log.append(p_des_vec.copy())

        # Accelerazione spaziale del frame EE dovuta allo stato corrente (senza nuovo a)
        a_ee_sp = pin.getFrameAcceleration(model, data, ee_frame_id, pin.ReferenceFrame.WORLD)
        xdd0 = np.hstack([a_ee_sp.linear, a_ee_sp.angular])

        # Errore di posa EE (WORLD) usando log su SE(3)
        T_we = data.oMf[ee_frame_id]
        R_cur = T_we.rotation
        p_cur = np.array(T_we.translation).reshape(3)

        # Orientazione desiderata: manteniamo quella corrente (alpha_des = 0)
        R_des = R_cur
        T_des = pin.SE3(R_des, p_des_vec)
        T_err = T_des * T_we.inverse()
        # pin.log6 restituisce un oggetto pin.Motion; usiamo linear() e angular()
        log6_err = pin.log6(T_err)

        # Rimappa in [lin; ang] come nel nodo C++
        e6 = np.zeros(6)
        e6[0:3] = np.array(log6_err.linear).reshape(3)   # parte lineare
        e6[3:6] = np.array(log6_err.angular).reshape(3)  # parte angolare

        # Velocità EE desiderata e misurata (WORLD)
        v_ee_des = np.hstack([v_lin_des, omega_des])
        v_ee_motion = pin.getFrameVelocity(model, data, ee_frame_id, pin.ReferenceFrame.WORLD)
        v_ee_meas = np.hstack([np.array(v_ee_motion.linear).reshape(3),
                       np.array(v_ee_motion.angular).reshape(3)])
        e_v6 = v_ee_des - v_ee_meas

        # Accelerazione desiderata per feedback (posizione + velocità) dell'EE
        xdd_fb = K_ee @ e6 + Kd_ee @ e_v6

        # Composizione: di default usa feedback + feedforward;
        # se --no-ee-ff è passato, usa solo il feedback sulla posa.
        if args.no_ee_ff:
            xdd_des = xdd_fb
        else:
            xdd_des = xdd_ff + xdd_fb

        # Centroidal momentum matrix Ag e derivata dAg
        pin.computeCentroidalMap(model, data, q)
        Ag = data.Ag
        Ag_b = Ag[:, :6]
        Ag_m = Ag[:, 6:]
        pin.computeCentroidalMapTimeVariation(model, data, q, v)
        dAg = data.dAg

        # Prova calolo h0_dot con formula del momentum variation
        pin.computeCentroidalMomentumTimeVariation(model, data, q, v, np.zeros(nv))
        h0_dot = data.dhg  # centroidal momentum time derivative with a=0
        h0dot_norm_log.append(np.linalg.norm(h0_dot))

        # Calcolo della variazione del momento centroidale totale del sistema (se >0 => aumento momento del sistema => spinta esterna) 
        pin.computeCentroidalMomentumTimeVariation(model, data, q, v, a)
        h_dot = data.dhg
        #h_dot_pred = h_dot

        # Target centroidale: annulla h_dot (regola momentum verso costante)
        # h_dot = dAg*v + Ag*a  -> impongo h_dot_des = 0 => Ag*a = - dAg*v
        z1 = - dAg @ v
        #z1 = - h0_dot  # Questo sembra non piacergli. h_dot diventa diverso da zero (traj seguita cmq però). Perchè??

        # Target EE accelerazioni: xdd = J*a + Jdot*v  -> J*a = xdd_des - Jdot*v
        z2 = xdd_des - Jdot @ v

        if args.mode == "augmented":
            # Sistema aumentato su tutti i DoF: [Ag_b Ag_m; Jb Jm] a = [-dAg v; xdd_des - Jdot v]
            Aug = np.vstack([
                np.hstack([Ag_b, Ag_m]),
                np.hstack([Jb, Jm])
            ])
            rhs = np.hstack([z1, z2])

            try:
                u, s, vh = np.linalg.svd(Aug, full_matrices=False)
                cond = s[0] / max(s[-1], 1e-12)
                cond_aug_log.append(cond)
                cond_label = "cond(Aug)"
                a_sol = np.linalg.pinv(Aug, rcond=1e-8) @ rhs
            except Exception as e:
                print(f"Errore nella risoluzione del sistema aumentato (pinv): {e}")
                break

            # Assegna accelerazioni piene
            a[:] = a_sol

        else:  # freefloating
            # Risolviamo LO STESSO sistema aumentato, ma non usiamo direttamente a_b per controllare la base.
            Aug = np.vstack([
                np.hstack([Ag_b, Ag_m]),
                np.hstack([Jb, Jm])
            ])
            rhs = np.hstack([z1, z2])

            try:
                u, s, vh = np.linalg.svd(Aug, full_matrices=False)
                cond = s[0] / max(s[-1], 1e-12)
                cond_aug_log.append(cond)
                cond_label = "cond(Aug)"
                a_sol = np.linalg.pinv(Aug, rcond=1e-8) @ rhs
            except Exception as e:
                print(f"Errore nella risoluzione del sistema aumentato (pinv, freefloating): {e}")
                break

            # Prendi solo a_m dalla soluzione aumentata
            a_m = a_sol[6:]

            # Ricava accelerazione base dal vincolo di momentum: a_b = -Ag_b^{-1}(Ag_m a_m + dAg v)
            try:
                a_b = - np.linalg.solve(Ag_b, Ag_m @ a_m + dAg @ v)
            except np.linalg.LinAlgError:
                a_b = - (np.linalg.pinv(Ag_b, rcond=1e-8) @ (Ag_m @ a_m + dAg @ v))

            # Assembla accelerazioni piene
            a[:6] = a_b
            a[6:] = a_m

        # Aggiorna misura errore momentum previsto
        h_dot_pred = Ag @ a + dAg @ v
        err_momentum_log.append(np.linalg.norm(h_dot_pred))

        # Integrazione semplice (Euler) con dt nominale (passi deterministici)
        v = v + a * dt
        v = np.clip(v, -2.0, 2.0)
        q = pin.integrate(model, q, v * dt)

        # Riallinea la velocità della base al vincolo di momentum nel caso freefloating: v_b = -Ag_b^{-1} Ag_m v_m
        if args.mode == "freefloating":
            qd_m = v[6:]
            try:
                v[:6] = - np.linalg.solve(Ag_b, Ag_m @ qd_m)
            except np.linalg.LinAlgError:
                v[:6] = - (np.linalg.pinv(Ag_b, rcond=1e-8) @ (Ag_m @ qd_m))

        # Visualizza
        if viz is not None and (i % draw_stride) == 0:
            try:
                viz.display(q)
                # Aggiorna assi base ed EE
                if g is not None:
                    try:
                        # Base: usa il primo corpo dopo il free-flyer (indice 1)
                        T_wb = data.oMi[1]
                        viz.viewer["frames/base"].set_transform(T_wb.homogeneous)
                        # EE frame
                        T_we = data.oMf[ee_frame_id]
                        viz.viewer["frames/ee"].set_transform(T_we.homogeneous)
                        # Aggiorna traiettorie
                        ee_path_pts.append(np.array(T_we.translation).reshape(3))
                        ee_des_path_pts.append(p_des_vec.copy())
                        if len(ee_path_pts) >= 2:
                            P = np.array(ee_path_pts).T
                            viz.viewer["plots/ee_path"].set_object(
                                g.Line(g.PointsGeometry(P), g.LineMaterial(color=0xff0000, linewidth=2))
                            )
                        if len(ee_des_path_pts) >= 2:
                            Pd = np.array(ee_des_path_pts).T
                            viz.viewer["plots/ee_des_path"].set_object(
                                g.Line(g.PointsGeometry(Pd), g.LineMaterial(color=0x00aa00, linewidth=2))
                            )
                    except Exception:
                        pass
            except Exception:
                pass
            pass

        # Log
        t_log.append(t)
        ee_p_log.append(p_ee)

        # Stampa sintetica
        if (i % max(1, int(round(0.5 / dt)))) == 0:
            dx = p_ee[0] - p0[0]
            print(f"t={t:5.2f} | (dx,dy,dz)=({dx:.4f},{p_ee[1]-p0[1]:.4f},{p_ee[2]-p0[2]:.4f})")
        # Sleep opzionale per visualizzazione in tempo reale
        if args.realtime:
            dt_sleep = max(0.0, dt * args.rt_scale)
            elapsed = time.time() - loop_start
            remaining = dt_sleep - elapsed
            if remaining > 0:
                time.sleep(remaining)

    # Report finale
    ee_p_arr = np.vstack(ee_p_log) if len(ee_p_log) else np.zeros((0, 3))
    if ee_p_arr.shape[0] >= 2:
        delta = ee_p_arr[-1, :] - ee_p_arr[0, :]
        print("\nDifferenza posizione EE fine-inizio:")
        print(f"  dx = {delta[0]: .6f} m")
        print(f"  dy = {delta[1]: .6f} m")
        print(f"  dz = {delta[2]: .6f} m")

    # Plot opzionale
    try:
        import matplotlib.pyplot as plt
        t_arr = np.array(t_log)
        cond_arr = np.array(cond_aug_log)
        # plt.figure()
        # plt.plot(t_arr, cond_arr)
        # plt.xlabel('t [s]')
        # plt.ylabel('cond(Aug)')
        # plt.title('Condizionamento del sistema aumentato')

        if len(err_momentum_log) == len(t_arr):
            plt.figure()
            plt.plot(t_arr, np.array(err_momentum_log), label='||h_dot||')
            if len(h0dot_norm_log) == len(t_arr):
                plt.plot(t_arr, np.array(h0dot_norm_log), '--', label='||h0_dot|| (a=0)')
            plt.xlabel('t [s]')
            plt.ylabel('||h_dot||')
            plt.title('Norma momento centroidale h_dot')
            plt.legend(loc='best')

        if ee_p_arr.shape[0] >= 2:
            fig3d = plt.figure()
            ax3d = fig3d.add_subplot(111, projection='3d')
            ax3d.plot(ee_p_arr[:, 0], ee_p_arr[:, 1], ee_p_arr[:, 2], label='EE path (WORLD)')
            # Traiettoria desiderata
            if len(p_des_log) >= 2:
                ee_des_arr = np.vstack(p_des_log)
                ax3d.plot(ee_des_arr[:, 0], ee_des_arr[:, 1], ee_des_arr[:, 2], 'g--', label='EE desired (WORLD)')
            ax3d.scatter(ee_p_arr[0, 0], ee_p_arr[0, 1], ee_p_arr[0, 2], c='g', s=40, label='start')
            ax3d.scatter(ee_p_arr[-1, 0], ee_p_arr[-1, 1], ee_p_arr[-1, 2], c='r', s=40, label='end')
            ax3d.set_xlabel('X [m]')
            ax3d.set_ylabel('Y [m]')
            ax3d.set_zlabel('Z [m]')
            ax3d.set_title('Traiettoria EE in WORLD (reale vs desiderata)')
            # Imposta stessa scala sugli assi 3D
            try:
                if len(p_des_log) >= 2:
                    ee_all = np.vstack([ee_p_arr, np.vstack(p_des_log)])
                else:
                    ee_all = ee_p_arr
                ranges = ee_all.max(axis=0) - ee_all.min(axis=0)
                max_range = max(ranges.max(), 1e-9)
                mid = (ee_all.max(axis=0) + ee_all.min(axis=0)) / 2.0
                ax3d.set_xlim(mid[0] - max_range/2, mid[0] + max_range/2)
                ax3d.set_ylim(mid[1] - max_range/2, mid[1] + max_range/2)
                ax3d.set_zlim(mid[2] - max_range/2, mid[2] + max_range/2)
                # Disponibile da Matplotlib 3.4+: mantiene aspect cubico
                ax3d.set_box_aspect((1, 1, 1))
            except Exception:
                pass
            ax3d.legend(loc='best')
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
