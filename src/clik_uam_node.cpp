#include "clik2_node_pkg/clik_uam_node.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <type_traits> // per std::is_convertible_v usato internamente dai macro RCLCPP
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/crba.hpp"    // inertia matrix (CRBA)
#include "pinocchio/algorithm/rnea.hpp"    // nonLinearEffects, gravity, RNEA
#include "pinocchio/algorithm/centroidal.hpp"
#include "pinocchio/algorithm/model.hpp"               // buildReducedModel
#include "pinocchio/algorithm/joint-configuration.hpp" // neutral, normalize
#include "pinocchio/spatial/se3.hpp" // per SE3 log6
#include "pinocchio/spatial/explog.hpp" // per log3 su SO(3)
#include "interbotix_xs_msgs/msg/joint_group_command.hpp" // Include necessario per il nuovo tipo di messaggio
#include "ament_index_cpp/get_package_share_directory.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include <Eigen/SVD>
#include <Eigen/Dense>
#include <iomanip>
#include <algorithm> // std::clamp
#include <sstream>   // debug formatting
#include <cmath>     // std::isfinite, std::abs
#include <chrono>
#include <limits>

/// @brief 
ClikUamNode::ClikUamNode() : Node("clik_uam_node"), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_)
{
    RCLCPP_INFO(this->get_logger(), "Nodo clik_uam_node avviato.");

    this->declare_parameter<bool>("use_gazebo_pose", true);
    this->get_parameter("use_gazebo_pose", use_gazebo_pose_);
    this->declare_parameter<bool>("use_gz_odom", true);
    use_gz_odom_ = this->get_parameter("use_gz_odom").as_bool();

    // Parametri per instradare i topic e la modalità reale/simulata
    this->declare_parameter<std::string>("robot_name", "mobile_wx250s");
    this->declare_parameter<bool>("real_system", false);
    real_system_ = this->get_parameter("real_system").as_bool();

    // Carica il modello URDF
    // NOTA: il percorso del file URDF potrebbe dover essere reso un parametro
    const auto pkg_share = ament_index_cpp::get_package_share_directory("clik2_node_pkg");
    const std::string urdf_filename = pkg_share + "/model/t960a.urdf";
    
    RCLCPP_INFO(this->get_logger(), "Caricamento modello URDF da: %s", urdf_filename.c_str());
    
    try {
        pinocchio::urdf::buildModel(urdf_filename, pinocchio::JointModelFreeFlyer(), model_);
        data_ = pinocchio::Data(model_);
        
        RCLCPP_INFO(this->get_logger(), "Modello URDF caricato con successo.");
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Errore nel caricamento del modello URDF: %s", e.what());
        rclcpp::shutdown();
        return;
    }

    if (!model_.existFrame("mobile_wx250s/ee_gripper_link")) {
        RCLCPP_ERROR(this->get_logger(), "Frame 'mobile_wx250s/ee_gripper_link' assente nel modello.");
        rclcpp::shutdown();
        return;
    }
    ee_frame_id_ = model_.getFrameId("mobile_wx250s/ee_gripper_link");
    RCLCPP_INFO(this->get_logger(), "End-effector frame ID: %d", static_cast<int>(ee_frame_id_));

    // Frame base braccio (terna O) - serve per usare la twist della base del braccio (non del drone)
    if (!model_.existFrame("mobile_wx250s/base_link")) {
        RCLCPP_ERROR(this->get_logger(), "Frame 'mobile_wx250s/base_link' assente nel modello.");
        rclcpp::shutdown();
        return;
    }
    arm_base_frame_id_full_ = model_.getFrameId("mobile_wx250s/base_link");

    // Giunti controllati (gruppo "arm"). Definiti prima di precomputare gli indici.
    arm_joints_ = {"waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"};

    // Carica modello del SOLO manipolatore (FreeFlyer + braccio) per estrazione H_MR e n (reaction moment)
    {
        const std::string urdf_man_filename = pkg_share + "/model/wx250s.urdf";
        RCLCPP_INFO(this->get_logger(), "Caricamento modello manipolatore URDF da: %s", urdf_man_filename.c_str());
        try {
            // Costruisci il modello completo (include anche gripper e dita), poi riducilo lockando i giunti del gripper.
            pinocchio::Model model_man_full;
            pinocchio::urdf::buildModel(urdf_man_filename, pinocchio::JointModelFreeFlyer(), model_man_full);

            // Lock dei giunti gripper (come nello script test_acc_control_QP.py)
            const std::vector<std::string> gripper_joints = {"gripper", "left_finger", "right_finger"};
            std::vector<pinocchio::JointIndex> joints_to_lock;
            joints_to_lock.reserve(gripper_joints.size());
            for (const auto &jn : gripper_joints) {
                const pinocchio::JointIndex jid = model_man_full.getJointId(jn);
                if (jid > 0) {
                    joints_to_lock.push_back(jid);
                }
            }

            if (!joints_to_lock.empty()) {
                const Eigen::VectorXd q0_lock = pinocchio::neutral(model_man_full);
                model_man_ = pinocchio::buildReducedModel(model_man_full, joints_to_lock, q0_lock);
                RCLCPP_INFO(this->get_logger(), "Modello manipolatore ridotto: lockati %zu giunti gripper", joints_to_lock.size());
            } else {
                model_man_ = model_man_full;
                RCLCPP_WARN(this->get_logger(), "Giunti gripper non trovati nel modello manipolatore; uso modello completo");
            }

            // Richiesta: ignora la gravità nel modello del braccio (n_MR non include il contributo del peso)
            model_man_.gravity.linear().setZero();
            model_man_.gravity.angular().setZero();

            data_man_ = pinocchio::Data(model_man_);
        } catch (const std::exception &e) {
            RCLCPP_ERROR(this->get_logger(), "Errore nel caricamento del modello manipolatore URDF: %s", e.what());
            rclcpp::shutdown();
            return;
        }

        // Precalcolo indici q/v nel modello manipolatore per gli stessi arm_joints_
        const int n_arm = static_cast<int>(arm_joints_.size());
        idx_v_arm_man_.resize(n_arm);
        idx_q_arm_man_.resize(n_arm);
        for (int i = 0; i < n_arm; ++i) {
            const std::string &jname = arm_joints_[static_cast<size_t>(i)];
            if (!model_man_.existJointName(jname)) {
                RCLCPP_ERROR(this->get_logger(), "Joint '%s' non esiste nel modello manipolatore.", jname.c_str());
                rclcpp::shutdown();
                return;
            }
            const pinocchio::JointIndex jid = model_man_.getJointId(jname);
            const int idx_v = static_cast<int>(model_man_.joints[jid].idx_v());
            const int idx_q = static_cast<int>(model_man_.joints[jid].idx_q());
            const int nq_j = static_cast<int>(model_man_.joints[jid].nq());
            if (nq_j != 1) {
                RCLCPP_ERROR(this->get_logger(), "Joint '%s' nel modello manipolatore ha nq!=1 (nq=%d)", jname.c_str(), nq_j);
                rclcpp::shutdown();
                return;
            }
            idx_v_arm_man_[i] = idx_v;
            idx_q_arm_man_[i] = idx_q;
        }

        q_man_.resize(model_man_.nq);
        q_man_.setZero();
        v_man_.resize(model_man_.nv);
        v_man_.setZero();

        // Frame end-effector nel modello manipolatore (usato per Jacobiano classico del braccio)
        if (model_man_.existFrame("mobile_wx250s/ee_gripper_link")) {
            ee_frame_id_man_ = model_man_.getFrameId("mobile_wx250s/ee_gripper_link");
            have_ee_frame_man_ = true;
        } else if (model_man_.existFrame("ee_gripper_link")) {
            ee_frame_id_man_ = model_man_.getFrameId("ee_gripper_link");
            have_ee_frame_man_ = true;
        } else {
            have_ee_frame_man_ = false;
            RCLCPP_WARN(this->get_logger(), "Frame EE non trovato in model_man_ (wx250s.urdf). Userò lo Jacobiano del modello completo come fallback.");
        }
    }

    // SOTTOSCRIZIONI STATO DRONE
    if (use_gazebo_pose_) {
        RCLCPP_INFO(this->get_logger(), "Utilizzo della posa da Gazebo (/world/default/dynamic_pose/info).");
        gazebo_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseArray>(
            "/world/default/dynamic_pose/info", 10, std::bind(&ClikUamNode::gazebo_pose_callback, this, std::placeholders::_1));
    } else {
        RCLCPP_INFO(this->get_logger(), "Utilizzo della posa dal Motion Capture (/t960a/pose - PoseStamped da natnet_ros2).");
        real_drone_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/t960a/pose", 10, std::bind(&ClikUamNode::real_drone_pose_callback, this, std::placeholders::_1));
    }

    // Sottoscrizione alla velocità/odometria del drone. Regole:
    // - Se use_gz_odom == false: usa /real_t960a_twist (da real_drone_vel_pub) e NON iscriversi a /model/t960a_0/odometry
    // - Se use_gz_odom == true e use_gazebo_pose_ == true: usa /model/t960a_0/odometry
    // - Se real_system_ == true: comunque iscriviti a /real_t960a_twist

    if (!use_gz_odom_ || real_system_ || !use_gazebo_pose_)
    {
        // Twist già trasformato in FLU (da real_drone_vel_pub, quando disponibile)
        real_drone_twist_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/real_t960a_twist", rclcpp::SensorDataQoS(),
            std::bind(&ClikUamNode::real_drone_twist_callback, this, std::placeholders::_1));
        RCLCPP_INFO(this->get_logger(), "Iscrizione a /real_t960a_twist (Twist FLU) [use_gz_odom=%d, real_system=%d, use_gazebo_pose=%d]",
                    use_gz_odom_, real_system_, use_gazebo_pose_);
    }

    if (use_gz_odom_ && use_gazebo_pose_)
    {
        // Simulazione: usa l'odometria del modello gazebo bridgiata su ROS2
        gazebo_odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/model/t960a_0/odometry", 10,
            std::bind(&ClikUamNode::gazebo_odometry_callback, this, std::placeholders::_1));
        RCLCPP_INFO(this->get_logger(), "Iscrizione a /model/t960a_0/odometry (nav_msgs/Odometry) [use_gz_odom=false, use_gazebo_pose=true]");
    }

    // Joint states
    {
        std::string joint_states_topic;
        // if (real_system_) {
        //     joint_states_topic = "/" + robot_name_ + "/joint_states";
        // } else {
        joint_states_topic = "/joint_states";
        // }
        joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            joint_states_topic, 10, std::bind(&ClikUamNode::joint_state_callback, this, std::placeholders::_1));
        RCLCPP_INFO(this->get_logger(), "Mi sottoscrivo a %s", joint_states_topic.c_str());
    }

    // Publisher comandi: se reale, JointGroupCommand su /<robot_name>/commands/joint_group; se SITL, Float64MultiArray su /arm_controller/commands
    if (real_system_) {
        const std::string cmd_topic = "/commands/joint_group";
        joint_group_pub_ = this->create_publisher<interbotix_xs_msgs::msg::JointGroupCommand>(cmd_topic, 10);
        RCLCPP_INFO(this->get_logger(), "real_system=true -> pubblicherò su %s (JointGroupCommand)", cmd_topic.c_str());
    } else {
        arm_controller_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/arm_controller/commands", 10);
        RCLCPP_INFO(this->get_logger(), "real_system=false -> pubblicherò su /arm_controller/commands (Float64MultiArray)");
    }

    // Pose corrente EE (rispetto al frame world)
    ee_world_pose_pub_ = this->create_publisher<geometry_msgs::msg::Pose>("/ee_world_pose", 10);

    // Subscription alla posa desiderata pubblicata dal planner (QoS latched)
    desired_ee_global_pose_sub_ = this->create_subscription<geometry_msgs::msg::Pose>(
        "/desired_ee_global_pose", rclcpp::QoS(10),  // QoS volatile: non riceve messaggi precedenti all'avvio
        std::bind(&ClikUamNode::desired_pose_callback, this, std::placeholders::_1));

    // Subscription alla velocità desiderata EE (default: zero se non presente)
    desired_ee_velocity_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
        "/desired_ee_velocity", rclcpp::QoS(10),
        std::bind(&ClikUamNode::desired_velocity_callback, this, std::placeholders::_1));

    // Subscription all'accelerazione desiderata EE (default: zero se non presente)
    desired_ee_accel_sub_ = this->create_subscription<geometry_msgs::msg::Accel>(
        "/desired_ee_accel", rclcpp::QoS(10),
        std::bind(&ClikUamNode::desired_accel_callback, this, std::placeholders::_1));

    // Allocazioni
    q_.resize(model_.nq);
    q_.setZero();
    qd_.resize(model_.nv);
    qd_.setZero();
    J_.resize(6, model_.nv);
    error_pose_ee_.resize(6);
    error_vel_ee_.resize(6);

    declare_parameter("k_err_pos_", 20.0);
    declare_parameter("k_err_vel_", 20.0);
    this->declare_parameter<double>("joint_vel_limit", 3.14);
    // NOTE: dichiarato solo per compatibilità con vecchie config, ma non usato nei vincoli QP
    this->declare_parameter<double>("joint_acc_limit", 30.0);
    this->declare_parameter<double>("lambda_w", 10.0);
    this->declare_parameter<double>("w_kin", 1.0);
    this->declare_parameter<double>("w_dyn", 1.0);
    this->declare_parameter<double>("qp_lambda_reg", 1e-2);
    k_err_pos_ = get_parameter("k_err_pos_").as_double();
    k_err_vel_ = get_parameter("k_err_vel_").as_double();
    joint_vel_limit_ = this->get_parameter("joint_vel_limit").as_double();
    lambda_w_ = this->get_parameter("lambda_w").as_double();
    w_kin_ = this->get_parameter("w_kin").as_double();
    w_dyn_ = this->get_parameter("w_dyn").as_double();
    qp_lambda_reg_ = this->get_parameter("qp_lambda_reg").as_double();

    if (w_kin_ < 0.0) {
        RCLCPP_WARN(this->get_logger(), "w_kin < 0 non consentito; imposto w_kin=0");
        w_kin_ = 0.0;
    }
    if (w_dyn_ < 0.0) {
        RCLCPP_WARN(this->get_logger(), "w_dyn < 0 non consentito; imposto w_dyn=0");
        w_dyn_ = 0.0;
    }
    K_matrix_ = Eigen::MatrixXd::Identity(6, 6) * k_err_pos_;
    Kd_matrix_ = Eigen::MatrixXd::Identity(6, 6) * k_err_vel_;

    // Log iniziale dei guadagni di controllo posizione/velocità EE
    RCLCPP_INFO(this->get_logger(), "Guadagni EE: k_err_pos_=%.3f, k_err_vel_=%.3f", k_err_pos_, k_err_vel_);
    RCLCPP_INFO(this->get_logger(), "Pesi QP: w_kin=%.3f, w_dyn=%.3f (lambda_w legacy=%.3f)", w_kin_, w_dyn_, lambda_w_);

    // Pesi per la pseudoinversa
    this->declare_parameter<double>("shoulder_weight", 15.0);
    this->declare_parameter<double>("forearm_weight", 25.0);
    this->declare_parameter<double>("wrist_weight", 25.0);
    double shoulder_w = get_parameter("shoulder_weight").as_double();
    double forearm_w  = get_parameter("forearm_weight").as_double();
    double wrist_w    = get_parameter("wrist_weight").as_double();
    W_diag_.resize(arm_joints_.size());
    for (size_t i = 0; i < arm_joints_.size(); ++i) {
        const std::string &jn = arm_joints_[i];
        if (jn == "shoulder")       W_diag_[static_cast<Eigen::Index>(i)] = shoulder_w;
        else if (jn == "forearm_roll") W_diag_[static_cast<Eigen::Index>(i)] = forearm_w;
        else if (jn == "wrist_rotate") W_diag_[static_cast<Eigen::Index>(i)] = wrist_w;
        else W_diag_[static_cast<Eigen::Index>(i)] = 1.0;
    }

    // Opzione per sfruttare ridondanza cinematica: segui solo la traiettoria di posizione (ignora orientazione)
    this->declare_parameter<bool>("redundant", false);
    redundant_ = this->get_parameter("redundant").as_bool();

    // In modalità redundant, scala il feedback sull'orientazione EE (0 = nullo, 1 = pieno)
    this->declare_parameter<double>("redundant_ang_fb_scale", 0.05);
    redundant_ang_fb_scale_ = this->get_parameter("redundant_ang_fb_scale").as_double();

    // Precalcolo indici q/v dei giunti del braccio nel modello completo
    {
        const int n_arm = static_cast<int>(arm_joints_.size());
        idx_v_arm_.resize(n_arm);
        idx_q_arm_.resize(n_arm);
        for (int i = 0; i < n_arm; ++i) {
            const std::string &jname = arm_joints_[static_cast<size_t>(i)];
            if (!model_.existJointName(jname)) {
                RCLCPP_ERROR(this->get_logger(), "Joint '%s' non esiste nel modello.", jname.c_str());
                rclcpp::shutdown();
                return;
            }
            const pinocchio::JointIndex jid = model_.getJointId(jname);
            const int idx_v = static_cast<int>(model_.joints[jid].idx_v());
            const int idx_q = static_cast<int>(model_.joints[jid].idx_q());
            const int nq_j = static_cast<int>(model_.joints[jid].nq());
            if (nq_j != 1) {
                RCLCPP_ERROR(this->get_logger(), "Joint '%s' nel modello ha nq!=1 (nq=%d)", jname.c_str(), nq_j);
                rclcpp::shutdown();
                return;
            }
            idx_v_arm_[i] = idx_v;
            idx_q_arm_[i] = idx_q;
        }
    }

    // Estrazione limiti posizione/velocità (ordine arm_joints_) dal modello manipolatore
    {
        const int n_arm = static_cast<int>(arm_joints_.size());
        q_lower_arm_.resize(n_arm);
        q_upper_arm_.resize(n_arm);
        v_limit_arm_.resize(n_arm);
        q_lower_arm_.setZero();
        q_upper_arm_.setZero();
        v_limit_arm_.setZero();

        have_position_limits_ = (model_man_.lowerPositionLimit.size() == static_cast<Eigen::Index>(model_man_.nq)) &&
                               (model_man_.upperPositionLimit.size() == static_cast<Eigen::Index>(model_man_.nq));
        have_velocity_limits_ = (model_man_.velocityLimit.size() == static_cast<Eigen::Index>(model_man_.nv));

        for (int i = 0; i < n_arm; ++i) {
            const int iq = idx_q_arm_man_[i];
            const int iv = idx_v_arm_man_[i];
            if (have_position_limits_) {
                q_lower_arm_[i] = model_man_.lowerPositionLimit[static_cast<Eigen::Index>(iq)];
                q_upper_arm_[i] = model_man_.upperPositionLimit[static_cast<Eigen::Index>(iq)];
            } else {
                q_lower_arm_[i] = -std::numeric_limits<double>::infinity();
                q_upper_arm_[i] = +std::numeric_limits<double>::infinity();
            }
            if (have_velocity_limits_) {
                v_limit_arm_[i] = model_man_.velocityLimit[static_cast<Eigen::Index>(iv)];
            } else {
                v_limit_arm_[i] = 0.0;
            }
        }
    }

    // Setup QP (OSQP-Eigen): variabili = accelerazioni giunti del braccio, vincoli = box (A=I)
    {
        qp_n_ = static_cast<int>(arm_joints_.size());

        qp_gradient_.resize(qp_n_);
        qp_l_.resize(qp_n_);
        qp_u_.resize(qp_n_);
        qp_solution_.resize(qp_n_);
        qp_solution_.setZero();

        qp_P_dense_.resize(qp_n_, qp_n_);
        qp_P_dense_.setZero();
        qp_J_task_.resize(6, qp_n_);
        qp_J_task_.setZero();
        qp_v_task_.resize(6);
        qp_v_task_.setZero();
        qp_H_mr_.resize(3, qp_n_);
        qp_H_mr_.setZero();
        qp_n_mr_.resize(3);
        qp_n_mr_.setZero();

        // A = I (box constraints)
        qp_A_.resize(qp_n_, qp_n_);
        qp_A_.setIdentity();
        qp_A_.makeCompressed();

        // Hessian sparsity: full upper-triangular pattern
        qp_hessian_.resize(qp_n_, qp_n_);
        qp_hessian_.reserve(Eigen::Index(qp_n_ * (qp_n_ + 1) / 2));
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(static_cast<size_t>(qp_n_ * (qp_n_ + 1) / 2));
        for (int j = 0; j < qp_n_; ++j) {
            for (int i = 0; i <= j; ++i) {
                triplets.emplace_back(i, j, (i == j) ? qp_lambda_reg_ : 0.0);
            }
        }
        qp_hessian_.setFromTriplets(triplets.begin(), triplets.end());
        qp_hessian_.makeCompressed();

        qp_solver_.settings()->setVerbosity(false);
        qp_solver_.settings()->setWarmStart(true);
        qp_solver_.settings()->setPolish(false);
        qp_solver_.data()->setNumberOfVariables(qp_n_);
        qp_solver_.data()->setNumberOfConstraints(qp_n_);
        qp_solver_.data()->setLinearConstraintsMatrix(qp_A_);

        qp_initialized_ = false;
    }

    // Timer di update() controllo parametrico
    this->declare_parameter<double>("control_rate_hz", 100.0);
    double rate_hz = this->get_parameter("control_rate_hz").as_double();
    rate_hz = std::max(1.0, rate_hz); // salvaguardia: minimo 1 Hz
    const auto period_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(1.0 / rate_hz));
    control_timer_ = this->create_wall_timer(period_ns, std::bind(&ClikUamNode::update, this));
    RCLCPP_INFO(this->get_logger(), "control_rate_hz=%.2f Hz (periodo=%.3f ms)", rate_hz, 1000.0 / rate_hz);
    last_update_time_ = this->now();
    this->declare_parameter<double>("desired_timeout_sec", 0.5);
    desired_timeout_sec_ = this->get_parameter("desired_timeout_sec").as_double();}

void ClikUamNode::desired_pose_callback(const geometry_msgs::msg::Pose::SharedPtr msg) {
    desired_ee_pose_world_ = *msg;
    desired_ee_pose_world_ready_ = true;
    last_desired_msg_time_ = this->now();
    have_desired_msg_ = true;
    //RCLCPP_INFO(this->get_logger(), "Nuova posa desiderata ricevuta: x=%.3f y=%.3f z=%.3f", msg->position.x, msg->position.y, msg->position.z);
}

void ClikUamNode::desired_velocity_callback(const geometry_msgs::msg::Twist::SharedPtr msg) {
    desired_ee_velocity_world_ = *msg;
    desired_ee_velocity_ready_ = true; // indica che siamo in una fase di tracking
    last_desired_msg_time_ = this->now();
    have_desired_msg_ = true;
}

void ClikUamNode::desired_accel_callback(const geometry_msgs::msg::Accel::SharedPtr msg) {
    desired_ee_accel_world_ = *msg;
    desired_ee_accel_ready_ = true; // indica che siamo in una fase di tracking
    last_desired_msg_time_ = this->now();
    have_desired_msg_ = true;
}

void ClikUamNode::vehicle_local_position_callback(const px4_msgs::msg::VehicleLocalPosition::SharedPtr msg)
{
    // Conversione da NED (PX4) a FLU (Forward, Left, Up)
    // NED: x=N (Forward), y=E (Right), z=D (Down)
    // FLU: x=Forward -> x_ned, y=Left -> -y_ned, z=Up -> -z_ned
    vehicle_local_position_ = *msg; // copia originale
    vehicle_local_position_.x = msg->y;        // Forward
    vehicle_local_position_.y = msg->x;       // Left
    vehicle_local_position_.z = -msg->z;       // Up

    has_vehicle_local_position_ = true;
}

void ClikUamNode::vehicle_attitude_callback(const px4_msgs::msg::VehicleAttitude::SharedPtr msg)
{
    // Conversione da FRD (PX4) a FLU (ROS2)
    // Eigen::Quaterniond frd_quat(msg->q[0], msg->q[1], msg->q[2], msg->q[3]);
    // FRD (Forward, Right, Down) -> FLU (Forward, Left, Up)
    // La conversione consiste nell'invertire gli assi Y e Z
    // Eigen::Quaterniond flu_quat(frd_quat.w(), frd_quat.x(), -frd_quat.y(), -frd_quat.z());
    Eigen::Quaterniond flu_quat(msg->q[0], msg->q[1], -msg->q[2], -msg->q[3]);


    vehicle_attitude_ = *msg; // Copio il messaggio originale
    // Sovrascrivo con il quaternione convertito
    vehicle_attitude_.q[0] = flu_quat.w();
    vehicle_attitude_.q[1] = flu_quat.x();
    vehicle_attitude_.q[2] = flu_quat.y();
    vehicle_attitude_.q[3] = flu_quat.z();

    has_vehicle_attitude_ = true;
}

void ClikUamNode::vehicle_odometry_callback(const px4_msgs::msg::VehicleOdometry::SharedPtr msg)
{
    // PX4 fornisce twist nel frame del corpo FRD (Forward, Right, Down)
    // Convertiamo a FLU: (x = x, y = -y, z = -z) sia per linear che per angular
    vehicle_odom_ = *msg;
    has_vehicle_odom_ = true;
}

void ClikUamNode::gazebo_odometry_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
    // In simulazione assumiamo che la twist di Gazebo sia espressa nel frame WORLD
    // già in convenzione FLU, quindi non servono conversioni di frame qui.
    gazebo_odom_ = *msg;
    has_gazebo_odom_ = true;
}

void ClikUamNode::real_drone_twist_callback(const geometry_msgs::msg::Twist::SharedPtr msg)
{
    // Twist già espresso in:
    // - lineare: WORLD-FLU con heading iniziale rimosso (da real_drone_vel_pub)
    // - angolare: body FLU
    real_drone_twist_ = *msg;
    has_real_drone_twist_ = true;
}

void ClikUamNode::real_drone_pose_callback(const geometry_msgs::msg::PoseStamped::ConstSharedPtr msg)
{
    // Assumiamo che /t960a/pose sia in frame world FLU (come configurato in natnet_ros2)
    vehicle_local_position_.x = msg->pose.position.x;
    vehicle_local_position_.y = msg->pose.position.y;
    vehicle_local_position_.z = msg->pose.position.z;

    vehicle_attitude_.q[0] = msg->pose.orientation.w;
    vehicle_attitude_.q[1] = msg->pose.orientation.x;
    vehicle_attitude_.q[2] = msg->pose.orientation.y;
    vehicle_attitude_.q[3] = msg->pose.orientation.z;
    has_vehicle_local_position_ = true;
    has_vehicle_attitude_ = true;
}

void ClikUamNode::gazebo_pose_callback(const geometry_msgs::msg::PoseArray::SharedPtr msg)
{
    if (!msg->poses.empty())
    {
        const auto& pose = msg->poses[0];
        vehicle_local_position_.x = pose.position.x;
        vehicle_local_position_.y = pose.position.y;
        vehicle_local_position_.z = pose.position.z;

        vehicle_attitude_.q[0] = pose.orientation.w;
        vehicle_attitude_.q[1] = pose.orientation.x;
        vehicle_attitude_.q[2] = pose.orientation.y;
        vehicle_attitude_.q[3] = pose.orientation.z;

        has_vehicle_local_position_ = true;
        has_vehicle_attitude_ = true;
    }
}

void ClikUamNode::joint_state_callback(const sensor_msgs::msg::JointState::SharedPtr msg)
{
    // Copia il messaggio
    current_joint_state_ = *msg;

    // Se il topic non fornisce le velocità o contiene NaN/Inf, stimale via differenze finite
    const bool names_ok = !current_joint_state_.name.empty();
    const bool pos_ok   = !current_joint_state_.position.empty();
    const bool vel_absent = current_joint_state_.velocity.empty();
    bool vel_nonfinite = false;
    if (!vel_absent) {
        for (const auto &v : current_joint_state_.velocity) {
            if (!std::isfinite(v)) { vel_nonfinite = true; break; }
        }
    }

    if ((vel_absent || vel_nonfinite) && names_ok && pos_ok)
    {
        // Usa timestamp del messaggio se disponibile; altrimenti usa ora
        rclcpp::Time stamp = current_joint_state_.header.stamp;
        if (stamp.nanoseconds() == 0) {
            stamp = this->now();
        }

        std::vector<double> vel_est(current_joint_state_.name.size(), 0.0);
        double dt = 0.0;
        if (last_joint_state_time_.nanoseconds() != 0) {
            dt = (stamp - last_joint_state_time_).seconds();
        }

        if (dt > 1e-5) {
            for (size_t i = 0; i < current_joint_state_.name.size(); ++i) {
                const auto &jn = current_joint_state_.name[i];
                const double q_now = current_joint_state_.position[i];
                auto it = prev_joint_positions_.find(jn);
                if (it != prev_joint_positions_.end() && std::isfinite(q_now)) {
                    const double q_prev = it->second;
                    vel_est[i] = (q_now - q_prev) / dt;
                } else {
                    vel_est[i] = 0.0; // prima misura o posizione non finita: velocità => 0
                }
            }
        } else {
            // dt nullo/negativo: mantieni 0
            std::fill(vel_est.begin(), vel_est.end(), 0.0);
        }

        // Scrivi le velocità stimate nel messaggio corrente per l'update()
        current_joint_state_.velocity = vel_est;

        // Aggiorna memoria delle posizioni
        for (size_t i = 0; i < current_joint_state_.name.size(); ++i) {
            prev_joint_positions_[current_joint_state_.name[i]] = current_joint_state_.position[i];
        }
        last_joint_state_time_ = stamp;
    } else {
        // Aggiorna memoria posizioni e timestamp anche se le velocità sono fornite e finite
        if (names_ok && pos_ok) {
            for (size_t i = 0; i < current_joint_state_.name.size(); ++i) {
                prev_joint_positions_[current_joint_state_.name[i]] = current_joint_state_.position[i];
            }
            rclcpp::Time stamp = current_joint_state_.header.stamp;
            if (stamp.nanoseconds() == 0) {
                stamp = this->now();
            }
            last_joint_state_time_ = stamp;
        }
    }

    has_current_joint_state_ = true;
}

void ClikUamNode::update()
{
    if (!desired_ee_pose_world_ready_ || !has_current_joint_state_ || !has_vehicle_local_position_ || !has_vehicle_attitude_)
    {
        if (!waiting_log_printed_) {
            RCLCPP_INFO(this->get_logger(), "In attesa di: posa_desiderata=%d joint_state=%d posa_drone=%d attitude_drone=%d",
                        desired_ee_pose_world_ready_, has_current_joint_state_, has_vehicle_local_position_, has_vehicle_attitude_);
            waiting_log_printed_ = true;
        }
        return;
    }
    // appena tutti i dati disponibili, reset del flag
    if (waiting_log_printed_) {
        RCLCPP_INFO(this->get_logger(), "Dati pronti. Avvio controllo.");
        waiting_log_printed_ = false;
    }

    // Timestamp corrente per calcoli di differenziazione
    // (conta le iterazioni dell'update dopo il superamento della guardia)
    update_iterations_after_guard_++;
    rclcpp::Time now = this->now();

    // dt del controllo (serve per vincoli discretizzati del QP e integrazione)
    double dt = (now - last_update_time_).seconds();
    last_update_time_ = now;
    dt = std::clamp(dt, 1e-4, 0.02);

    // 0. AGGIORNA STATO DEL DRONE

    q_[0] = vehicle_local_position_.x;
    q_[1] = vehicle_local_position_.y;
    q_[2] = vehicle_local_position_.z;
    q_[3] = vehicle_attitude_.q[1]; // x
    q_[4] = vehicle_attitude_.q[2]; // y
    q_[5] = vehicle_attitude_.q[3]; // z
    q_[6] = vehicle_attitude_.q[0]; // w

    // --- Twist base (drone): usa odometria se disponibile, altrimenti differenziazione ---
    geometry_msgs::msg::Pose current_drone_pose;
    current_drone_pose.position.x = vehicle_local_position_.x;
    current_drone_pose.position.y = vehicle_local_position_.y;
    current_drone_pose.position.z = vehicle_local_position_.z;
    current_drone_pose.orientation.x = vehicle_attitude_.q[1];
    current_drone_pose.orientation.y = vehicle_attitude_.q[2];
    current_drone_pose.orientation.z = vehicle_attitude_.q[3];
    current_drone_pose.orientation.w = vehicle_attitude_.q[0];

    // Velocità WORLD del drone
    Eigen::Vector3d vlin_base_world = Eigen::Vector3d::Zero();
    Eigen::Vector3d omega_base_world = Eigen::Vector3d::Zero();

    // Sorgenti disponibili:
    //  - Twist FLU pubblicato da real_drone_vel_pub su /real_t960a_twist
    //  - Odometria Gazebo (WORLD-FLU) su /model/t960a_0/odometry
    // Nessuna più conversione NED/ENU e nessuna differenziazione numerica

    if (has_real_drone_twist_)
    {
        // Parte lineare in WORLD-FLU (heading iniziale già rimosso)
        vlin_base_world = Eigen::Vector3d(
            real_drone_twist_.linear.x,
            real_drone_twist_.linear.y,
            real_drone_twist_.linear.z
        );
        // La parte angolare viene letta direttamente come body FLU più sotto
    }
    else if (use_gazebo_pose_ && has_gazebo_odom_)
    {
        // Simulazione: assumiamo che la twist di Gazebo sia già in WORLD-FLU
        vlin_base_world = Eigen::Vector3d(
            gazebo_odom_.twist.twist.linear.x,
            gazebo_odom_.twist.twist.linear.y,
            gazebo_odom_.twist.twist.linear.z
        );

        omega_base_world = Eigen::Vector3d(
            gazebo_odom_.twist.twist.angular.x,
            gazebo_odom_.twist.twist.angular.y,
            gazebo_odom_.twist.twist.angular.z
        );
    }

    // --- VERIFICARE: Stima velocità generalizzata v (Pinocchio richiede LOCAL per la base free-flyer) ---
    // Converti WORLD -> LOCAL usando R^T per la parte lineare.
    // La parte angolare, nel caso reale, arriva già in body FLU da real_drone_vel_pub.
    Eigen::Quaterniond q_world_base(vehicle_attitude_.q[0], vehicle_attitude_.q[1], vehicle_attitude_.q[2], vehicle_attitude_.q[3]);
    Eigen::Matrix3d Rwb = q_world_base.normalized().toRotationMatrix();
    Eigen::Vector3d vlin_base_local = Rwb.transpose() * vlin_base_world;

    Eigen::Vector3d omega_base_local;
    if (has_real_drone_twist_)
    {
        // Twist angolare già in frame body FLU
        omega_base_local = Eigen::Vector3d(
            real_drone_twist_.angular.x,
            real_drone_twist_.angular.y,
            real_drone_twist_.angular.z
        );
    }
    else
    {
        // Caso Gazebo o assenza di real_drone_twist_: ruota da WORLD a LOCAL
        omega_base_local = Rwb.transpose() * omega_base_world;
    }


    // 1. AGGIORNO STATO GIUNTI BRACCIO
    const int n_arm = static_cast<int>(arm_joints_.size());
    Eigen::VectorXd q_arm_meas(n_arm);
    Eigen::VectorXd qd_arm_meas(n_arm);
    q_arm_meas.setZero();
    qd_arm_meas.setZero();

    // Estrai posizioni/velocità per i soli giunti del braccio nell'ordine arm_joints_
    qd_.setZero();
    for (int i = 0; i < n_arm; ++i) {
        const std::string &jn = arm_joints_[static_cast<size_t>(i)];
        double q_i = 0.0;
        double dq_i = 0.0;
        bool found = false;
        for (size_t k = 0; k < current_joint_state_.name.size(); ++k) {
            if (current_joint_state_.name[k] == jn) {
                if (k < current_joint_state_.position.size()) q_i = current_joint_state_.position[k];
                if (k < current_joint_state_.velocity.size()) dq_i = current_joint_state_.velocity[k];
                found = true;
                break;
            }
        }
        if (!found) {
            q_i = 0.0;
            dq_i = 0.0;
        }
        q_arm_meas[i] = q_i;
        qd_arm_meas[i] = dq_i;
        q_[idx_q_arm_[i]] = q_i;
        qd_[idx_v_arm_[i]] = dq_i;
    }
    pinocchio::normalize(model_, q_);

    // Generalized velocity misurata (usa velocità drone misurate + velocità giunti misurate)
    Eigen::VectorXd v_gen_meas(model_.nv); v_gen_meas.setZero();
    v_gen_meas.segment<3>(0) = vlin_base_local;  // base lineare 
    v_gen_meas.segment<3>(3) = omega_base_local; // base angolare 
    // copia giunti braccio (ordine pinocchio)
    for (int i = 0; i < n_arm; ++i) {
        v_gen_meas[idx_v_arm_[i]] = qd_arm_meas[i];
    }

    // 2. CINEMATICA DIRETTA PER POSA GLOBALE CORRENTE EE

    pinocchio::forwardKinematics(model_, data_, q_, v_gen_meas);
    pinocchio::updateFramePlacements(model_, data_);

    const pinocchio::SE3& ee_placement = data_.oMf[ee_frame_id_];

    geometry_msgs::msg::Pose current_ee_pose_world;
    current_ee_pose_world.position.x = ee_placement.translation().x();
    current_ee_pose_world.position.y = ee_placement.translation().y();
    current_ee_pose_world.position.z = ee_placement.translation().z();
    // Costruisco il quaternion dall'orientazione dell'EE e lo normalizzo
    Eigen::Quaterniond ee_cur_quat(ee_placement.rotation());
    ee_cur_quat.normalize();
    current_ee_pose_world.orientation.x = ee_cur_quat.x();
    current_ee_pose_world.orientation.y = ee_cur_quat.y();
    current_ee_pose_world.orientation.z = ee_cur_quat.z();
    current_ee_pose_world.orientation.w = ee_cur_quat.w();

    // Pubblica la posa assoluta dell'end-effector
    ee_world_pose_pub_->publish(current_ee_pose_world);

    // 3. CALCOLO MATRICI JACOBIANI E CMM 

    // Jacobiano del frame in LOCAL_WORLD_ALIGNED (coerente con v_gen_meas: base in LOCAL)
    pinocchio::computeFrameJacobian(model_, data_, q_, ee_frame_id_, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J_);

    const Eigen::Index m_total = static_cast<Eigen::Index>(model_.nv) - 6; // DoF manipolatore = nv - 6
    const Eigen::Index m_arm = static_cast<Eigen::Index>(arm_joints_.size());

    // 4. LETTURA (con conversione messaggio) DEI VALORI DESIDERATI

    // Costruisci vettore velocità desiderata in WORLD [lin; ang]
    Eigen::VectorXd v_ee_des(6); v_ee_des.setZero();
    v_ee_des(0) = desired_ee_velocity_world_.linear.x;
    v_ee_des(1) = desired_ee_velocity_world_.linear.y;
    v_ee_des(2) = desired_ee_velocity_world_.linear.z;
    v_ee_des(3) = desired_ee_velocity_world_.angular.x;
    v_ee_des(4) = desired_ee_velocity_world_.angular.y;
    v_ee_des(5) = desired_ee_velocity_world_.angular.z;

    // --- Accelerazione desiderata end-effector (WORLD) ---
    Eigen::VectorXd acc_ee_des(6); acc_ee_des.setZero();
    acc_ee_des(0) = desired_ee_accel_world_.linear.x;
    acc_ee_des(1) = desired_ee_accel_world_.linear.y;
    acc_ee_des(2) = desired_ee_accel_world_.linear.z;
    acc_ee_des(3) = desired_ee_accel_world_.angular.x;
    acc_ee_des(4) = desired_ee_accel_world_.angular.y;
    acc_ee_des(5) = desired_ee_accel_world_.angular.z;


    // 5. CALCOLO ERRORE POSE END-EFFECTOR

    // Errore posa come in clik1_node_pkg / branch Jgen-MRmin:
    // - posizione: e_pos = p_des - p_cur (WORLD)
    // - orientazione: e_ang = log3(R_des * R_cur^T) (WORLD)
    Eigen::Quaterniond ee_des_quat(
        desired_ee_pose_world_.orientation.w,
        desired_ee_pose_world_.orientation.x,
        desired_ee_pose_world_.orientation.y,
        desired_ee_pose_world_.orientation.z
    );
    ee_des_quat.normalize();
    Eigen::Matrix3d R_des = ee_des_quat.toRotationMatrix();
    Eigen::Vector3d p_des(
        desired_ee_pose_world_.position.x,
        desired_ee_pose_world_.position.y,
        desired_ee_pose_world_.position.z
    );

    const Eigen::Vector3d p_cur = ee_placement.translation();
    const Eigen::Vector3d e_pos = p_des - p_cur; // world

    const Eigen::Matrix3d R_cur = ee_placement.rotation();
    const Eigen::Matrix3d R_err_world = R_des * R_cur.transpose();
    const Eigen::Vector3d e_ang = pinocchio::log3(R_err_world); // world

    error_pose_ee_.head<3>() = e_pos;
    error_pose_ee_.tail<3>() = e_ang;

    // 6. CALCOLO ERRORE DI VELOCITÀ END-EFFECTOR

    // Stima velocità EE diretta da modello (LOCAL_WORLD_ALIGNED)
    const pinocchio::Motion &ee_vel_motion = pinocchio::getFrameVelocity(model_, data_, ee_frame_id_, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED);
    Eigen::VectorXd v_ee_meas(6); // [lin; ang] per coerenza con v_ee_des
    v_ee_meas.head<3>() = ee_vel_motion.linear();
    v_ee_meas.tail<3>() = ee_vel_motion.angular();

    error_vel_ee_ = v_ee_des - v_ee_meas;


    // === Stato manipolatore-only (base = mobile_wx250s/base_link) ===
    // Serve per: (1) Jacobiano classico del braccio nel costo cinematico (J),
    //            (2) H_MR e n_MR nel costo dinamico.
    q_man_.setZero();
    v_man_.setZero();

    // Posa di O (mobile_wx250s/base_link) nel mondo dal modello completo
    const pinocchio::SE3 &T_w_O = data_.oMf[arm_base_frame_id_full_];
    q_man_.segment<3>(0) = T_w_O.translation();
    Eigen::Quaterniond q_O(T_w_O.rotation());
    q_O.normalize();
    q_man_[3] = q_O.x();
    q_man_[4] = q_O.y();
    q_man_[5] = q_O.z();
    q_man_[6] = q_O.w();

    // Copia posizioni giunti braccio (ordine arm_joints_)
    for (int i = 0; i < n_arm; ++i) {
        q_man_[idx_q_arm_man_[i]] = q_arm_meas[i];
    }

    // Twist della base del braccio (LOCAL di O) dal modello completo
    const pinocchio::Motion V_O_local = pinocchio::getFrameVelocity(
        model_, data_, arm_base_frame_id_full_, pinocchio::ReferenceFrame::LOCAL);
    v_man_.segment<3>(0) = V_O_local.linear();
    v_man_.segment<3>(3) = V_O_local.angular();

    // Copia velocità giunti braccio
    for (int i = 0; i < n_arm; ++i) {
        v_man_[idx_v_arm_man_[i]] = v_gen_meas[idx_v_arm_[i]];
    }
    pinocchio::normalize(model_man_, q_man_);

    // Jacobiano del manipolatore (solo colonne dei giunti arm) per il costo cinematico
    Eigen::Matrix<double, 6, Eigen::Dynamic> J_arm(6, n_arm);
    J_arm.setZero();
    if (have_ee_frame_man_) {
        Eigen::Matrix<double, 6, Eigen::Dynamic> Jm_full(6, model_man_.nv);
        Jm_full.setZero();
        pinocchio::computeFrameJacobian(
            model_man_, data_man_, q_man_, ee_frame_id_man_,
            pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, Jm_full);
        for (int i = 0; i < n_arm; ++i) {
            J_arm.col(i) = Jm_full.col(idx_v_arm_man_[i]);
        }
    } else {
        // Fallback: estrai Jacobiano dei soli giunti arm dal modello completo
        for (int i = 0; i < n_arm; ++i) {
            J_arm.col(i) = J_.col(idx_v_arm_[i]);
        }
    }


    // 7. CALCOLO vdot_des (task accel) secondo istruzioni: vdot_des = xdd_ref + Kp*e + Kd*e_dot - Jdot*v
    // Nota: qui manteniamo il termine -Jdot*v del modello completo (con base free-flyer) come nel codice precedente.
    pinocchio::computeJointJacobiansTimeVariation(model_, data_, q_, v_gen_meas);
    Eigen::Matrix<double, 6, Eigen::Dynamic> Jdot(6, model_.nv);
    Jdot.setZero();
    pinocchio::getFrameJacobianTimeVariation(model_, data_, ee_frame_id_, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, Jdot);
    const Eigen::VectorXd Jdot_v = Jdot * v_gen_meas;

    Eigen::VectorXd fb = K_matrix_ * error_pose_ee_ + Kd_matrix_ * error_vel_ee_;
    if (redundant_) {
        // Task 3D: ignoriamo orientazione e feedback angolare
        fb.tail(3).setZero();
    }

    Eigen::VectorXd vdot_des = acc_ee_des + fb - Jdot_v;

    // 8. COSTRUZIONE QP (come test_acc_control_QP.py):
    // min_qdd  || J*qdd - vdot_des ||^2_{w_kin} + || H_MR*qdd + n_MR ||^2_{w_dyn}
    // In modalità redundant: task 3D (solo posizione) -> usa solo le prime 3 righe (lineari).
    qp_J_task_.setZero();
    qp_v_task_.setZero();
    if (redundant_) {
        // Solo parte lineare (x,y,z)
        qp_v_task_.head<3>() = vdot_des.head<3>();
        for (int i = 0; i < n_arm; ++i) {
            qp_J_task_.col(i).head<3>() = J_arm.col(i).head<3>();
        }
    } else {
        qp_v_task_ = vdot_des;
        for (int i = 0; i < n_arm; ++i) {
            qp_J_task_.col(i) = J_arm.col(i);
        }
    }

    // Dinamica manipolatore-only per H_MR e n_mr (righe 3..5: momento base in frame body/LOCAL)
    q_man_.setZero();
    v_man_.setZero();
    // NOTA: la base del manipolatore-only corrisponde a mobile_wx250s/base_link (non al link base del drone)
    // Posa di O (mobile_wx250s/base_link) nel mondo dal modello completo
    const pinocchio::SE3 &T_w_O = data_.oMf[arm_base_frame_id_full_];
    q_man_.segment<3>(0) = T_w_O.translation();
    Eigen::Quaterniond q_O(T_w_O.rotation());
    q_O.normalize();
    q_man_[3] = q_O.x();
    q_man_[4] = q_O.y();
    q_man_[5] = q_O.z();
    q_man_[6] = q_O.w();

    // Copia posizioni giunti braccio (ordine arm_joints_)
    for (int i = 0; i < n_arm; ++i) {
        q_man_[idx_q_arm_man_[i]] = q_arm_meas[i];
    }

    // Twist della base del braccio (LOCAL di O) dal modello completo
    const pinocchio::Motion V_O_local = pinocchio::getFrameVelocity(
        model_, data_, arm_base_frame_id_full_, pinocchio::ReferenceFrame::LOCAL);
    v_man_.segment<3>(0) = V_O_local.linear();
    v_man_.segment<3>(3) = V_O_local.angular();

    // Copia velocità giunti braccio
    for (int i = 0; i < n_arm; ++i) {
        v_man_[idx_v_arm_man_[i]] = v_gen_meas[idx_v_arm_[i]];
    }
    pinocchio::normalize(model_man_, q_man_);

    pinocchio::crba(model_man_, data_man_, q_man_);
    data_man_.M.triangularView<Eigen::StrictlyLower>() = data_man_.M.transpose().triangularView<Eigen::StrictlyLower>();
    pinocchio::nonLinearEffects(model_man_, data_man_, q_man_, v_man_);

    qp_n_mr_ = data_man_.nle.segment<3>(3);
    for (int i = 0; i < n_arm; ++i) {
        qp_H_mr_.col(i) = data_man_.M.block<3, 1>(3, idx_v_arm_man_[i]);
    }

    qp_P_dense_.noalias() = (w_kin_ * (qp_J_task_.transpose() * qp_J_task_)) + (w_dyn_ * (qp_H_mr_.transpose() * qp_H_mr_));
    qp_P_dense_.diagonal().array() += qp_lambda_reg_;
    qp_P_dense_ = 0.5 * (qp_P_dense_ + qp_P_dense_.transpose());

    // OSQP-Eigen usa forma 0.5 x^T P x + g^T x
    // => g = w_dyn * H^T n - w_kin * J^T v
    qp_gradient_.noalias() = (w_dyn_ * (qp_H_mr_.transpose() * qp_n_mr_)) - (w_kin_ * (qp_J_task_.transpose() * qp_v_task_));

    // 9. Vincoli box su qdd (derivati da limiti vel/pos discretizzati)
    const double inf = std::numeric_limits<double>::infinity();
    const double dt2 = dt * dt;
    for (int i = 0; i < n_arm; ++i) {
        double l_i = -inf;
        double u_i = +inf;

        // velocity limits -> bounds on acceleration
        double vlim = 0.0;
        if (have_velocity_limits_) vlim = v_limit_arm_[i];
        if (!(vlim > 0.0)) vlim = joint_vel_limit_;
        const double dq_min = -std::abs(vlim);
        const double dq_max = +std::abs(vlim);
        const double l_vel = (dq_min - qd_arm_meas[i]) / dt;
        const double u_vel = (dq_max - qd_arm_meas[i]) / dt;
        l_i = std::max(l_i, l_vel);
        u_i = std::min(u_i, u_vel);

        // position limits -> bounds on acceleration using q_{k+1} = q + dt*dq + 0.5*dt^2*qdd
        if (have_position_limits_ && std::isfinite(q_lower_arm_[i]) && std::isfinite(q_upper_arm_[i])) {
            const double l_pos = (2.0 * (q_lower_arm_[i] - q_arm_meas[i] - dt * qd_arm_meas[i])) / dt2;
            const double u_pos = (2.0 * (q_upper_arm_[i] - q_arm_meas[i] - dt * qd_arm_meas[i])) / dt2;
            l_i = std::max(l_i, l_pos);
            u_i = std::min(u_i, u_pos);
        }

        qp_l_[i] = l_i;
        qp_u_[i] = u_i;
        if (qp_l_[i] > qp_u_[i]) {
            qp_l_[i] = qp_u_[i];
        }
    }

    // Aggiorna Hessian (upper-triangular) mantenendo pattern fisso
    for (int j = 0; j < n_arm; ++j) {
        for (int i = 0; i <= j; ++i) {
            qp_hessian_.coeffRef(i, j) = qp_P_dense_(i, j);
        }
    }

    qp_solution_.setZero();
    bool qp_ok = false;
    if (!qp_initialized_) {
        qp_solver_.data()->setHessianMatrix(qp_hessian_);
        qp_solver_.data()->setGradient(qp_gradient_);
        qp_solver_.data()->setLowerBound(qp_l_);
        qp_solver_.data()->setUpperBound(qp_u_);
        qp_initialized_ = qp_solver_.initSolver();
        if (!qp_initialized_) {
            RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "OSQP-Eigen initSolver() fallito");
        }
    } else {
        qp_solver_.updateHessianMatrix(qp_hessian_);
        qp_solver_.updateGradient(qp_gradient_);
        qp_solver_.updateLowerBound(qp_l_);
        qp_solver_.updateUpperBound(qp_u_);
    }

    if (qp_initialized_) {
        const auto flag = qp_solver_.solveProblem();
        if (flag == OsqpEigen::ErrorExitFlag::NoError) {
            qp_solution_ = qp_solver_.getSolution();
            qp_ok = true;
        }
    }

    if (!qp_ok) {
        qp_solution_.setZero();
    }

    // Componi vettore accelerazioni totali del manipolatore (arm + eventuali DoF extra a zero)
    Eigen::VectorXd qdd_total;
    qdd_total.resize(m_total);
    qdd_total.setZero();
    for (Eigen::Index i = 0; i < std::min<Eigen::Index>(m_total, m_arm); ++i) {
        qdd_total[i] = qp_solution_[static_cast<int>(i)];
    }


    if (!accel_buffers_initialized_) {
        q_pos_int_.resize(m_total); qd_int_.resize(m_total);
        q_pos_int_.setZero(); qd_int_.setZero();
        // inizializza feed-forward: per i primi 6 giunti (arm) usa i nomi; per gli altri lascia 0
        const Eigen::Index arm_count = static_cast<Eigen::Index>(arm_joints_.size());
        for (Eigen::Index i = 0; i < std::min(m_total, arm_count); ++i) {
            double current_pos = 0.0;
            const std::string &joint_name_to_find = arm_joints_[static_cast<size_t>(i)];
            for (size_t j = 0; j < current_joint_state_.name.size(); ++j) {
                if (current_joint_state_.name[j] == joint_name_to_find) { current_pos = current_joint_state_.position[j]; break; }
            }
            q_pos_int_[i] = current_pos;
        }
        // per i restanti DoF (p.es. gripper) mantieni zero come default sicuro
        accel_buffers_initialized_ = true;
    }

    // Integrazione secondo metodo di Eulero implicito con accelerazione totale
    qd_int_       += qdd_total * dt;

    // Sanifica e satura le velocità giunto rispetto al limite configurato
    auto sanitize_and_clamp_vel = [&](Eigen::VectorXd &v) {
        for (Eigen::Index i = 0; i < v.size(); ++i) {
            double &x = v[i];
            if (!std::isfinite(x)) x = 0.0;
            if (x > joint_vel_limit_) x = joint_vel_limit_;
            else if (x < -joint_vel_limit_) x = -joint_vel_limit_;
        }
    };
    sanitize_and_clamp_vel(qd_int_);

    // Aggiorna posizioni integrate
    q_pos_int_   += qd_int_ * dt;

    q_cmd_.resize(m_total);
    for (Eigen::Index i = 0; i < m_total; ++i) {
        double val = q_pos_int_[i];
        if (!std::isfinite(val)) val = 0.0; // evita NaN/Inf nel comando
        q_cmd_[i] = val;
    }

    // Prepara messaggio posizione target
    std_msgs::msg::Float64MultiArray command_msg;
    command_msg.data.resize(arm_joints_.size());
    for (size_t i = 0; i < arm_joints_.size(); ++i) {
        double val = 0.0;
        if (static_cast<int>(i) < q_cmd_.size()) val = q_cmd_[static_cast<int>(i)];
        command_msg.data[i] = val;
    }


    // Pubblicazione del messaggio
    if (real_system_) {
        interbotix_xs_msgs::msg::JointGroupCommand jgc;
        jgc.name = "arm";  // deve corrispondere al gruppo definito nel YAML (groups.arm)
        jgc.cmd.resize(command_msg.data.size());
        for (size_t i = 0; i < command_msg.data.size(); ++i) {
            jgc.cmd[i] = static_cast<float>(command_msg.data[i]);
        }
        if (joint_group_pub_) joint_group_pub_->publish(jgc);
    } else {
        if (arm_controller_pub_) arm_controller_pub_->publish(command_msg);
    }
}

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ClikUamNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
