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
#include "pinocchio/algorithm/rnea.hpp"    // nonLinearEffects, gravity, RNEA
#include "pinocchio/algorithm/centroidal.hpp"
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
    arm_joints_ = {"waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"};

    declare_parameter("k_err_pos_", 20.0);
    declare_parameter("k_err_vel_", 0.0);
    this->declare_parameter<double>("joint_vel_limit", 3.14);
    k_err_pos_ = get_parameter("k_err_pos_").as_double();
    k_err_vel_ = get_parameter("k_err_vel_").as_double();
    joint_vel_limit_ = this->get_parameter("joint_vel_limit").as_double();
    K_matrix_ = Eigen::MatrixXd::Identity(6, 6) * k_err_pos_;
    Kd_matrix_ = Eigen::MatrixXd::Identity(6, 6) * k_err_vel_;

    // Log iniziale dei guadagni di controllo posizione/velocità EE
    RCLCPP_INFO(this->get_logger(), "Guadagni EE: k_err_pos_=%.3f, k_err_vel_=%.3f", k_err_pos_, k_err_vel_);

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
    for (size_t i = 0; i < current_joint_state_.name.size(); ++i) {
        // Cerca il giunto nel modello di Pinocchio e aggiorna q_
        const auto& joint_name = current_joint_state_.name[i];
        if (model_.existJointName(joint_name)) {
            pinocchio::JointIndex joint_idx = model_.getJointId(joint_name);
            int joint_idx_int = static_cast<int>(joint_idx);
            if (joint_idx_int > 1 && (joint_idx_int - 2 + 7) < model_.nq) {
                q_[joint_idx_int - 2 + 7] = current_joint_state_.position[i];
            }
        }
    }
    pinocchio::normalize(model_, q_);

    // Estrai velocità giunti dai JointState (se disponibili)
    qd_.setZero();
    if (!current_joint_state_.velocity.empty()) {
        for (size_t i = 0; i < current_joint_state_.name.size(); ++i) {
            const auto &jn = current_joint_state_.name[i];
            if (model_.existJointName(jn)) {
                pinocchio::JointIndex jidx = model_.getJointId(jn);
                int jidx_int = static_cast<int>(jidx);
                // Mappatura empirica come per le posizioni (free-flyer joints occupano prime 7 posizioni in q, 6 in v)
                int col = jidx_int - 2; // corrispondente a ordine manipolatore dopo base (adattato al codice esistente)
                if (col >= 0 && (6 + col) < model_.nv && i < current_joint_state_.velocity.size()) {
                    qd_[6 + col] = current_joint_state_.velocity[i];  
                }
            }
        }
    }

    // Generalized velocity misurata (usa velocità drone misurate + velocità giunti misurate)
    Eigen::VectorXd v_gen_meas(model_.nv); v_gen_meas.setZero();
    v_gen_meas.segment<3>(0) = vlin_base_local;  // base lineare 
    v_gen_meas.segment<3>(3) = omega_base_local; // base angolare 
    // copia giunti
    for (int k = 6; k < model_.nv; ++k) v_gen_meas[k] = qd_[k];

    // Debug: stampa le componenti di v_gen_meas ([lin; ang] base in LOCAL e giunti)
    // {
    //     std::ostringstream oss;
    //     oss.setf(std::ios::fixed);
    //     oss.precision(4);
    //     oss << "v_gen_meas: base_local=["
    //         << v_gen_meas[0] << ", " << v_gen_meas[1] << ", " << v_gen_meas[2] << " | "
    //         << v_gen_meas[3] << ", " << v_gen_meas[4] << ", " << v_gen_meas[5] << "], joints=["
    //         << v_gen_meas.tail(model_.nv - 6).transpose() << "]";
    //     RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 200, "%s", oss.str().c_str());
        
    //     bool has_nan_base = false;
    //     for (int i = 0; i < 6; ++i) {
    //         if (!std::isfinite(v_gen_meas[i])) { has_nan_base = true; break; }
    //     }
    //     if (has_nan_base) {
    //         RCLCPP_WARN(this->get_logger(), "v_gen_meas contiene NaN/Inf nelle componenti base [0:5]");
    //     }
        
    //     bool has_nan_joints = false;
    //     for (int i = 6; i < model_.nv; ++i) {
    //         if (!std::isfinite(v_gen_meas[i])) { has_nan_joints = true; break; }
    //     }
    //     if (has_nan_joints) {
    //         RCLCPP_WARN(this->get_logger(), "v_gen_meas contiene NaN/Inf nelle componenti giunti [6:nv]");
    //     }
    // }

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

    // Jacobiano del frame nel WORLD. 
    pinocchio::computeFrameJacobian(model_, data_, q_, ee_frame_id_, pinocchio::ReferenceFrame::WORLD, J_);

    // Estrai i blocchi necessari
    const Eigen::Index m_total = static_cast<Eigen::Index>(model_.nv) - 6; // DoF manipolatore = nv - 6
    const Eigen::Index m_arm   = static_cast<Eigen::Index>(arm_joints_.size()); // 6 DoF dell'arm

    // Jacobian blocks
    Eigen::MatrixXd J_b = J_.leftCols(6);
    // Prealloca J_m con tutte le colonne del manipolatore
    Eigen::MatrixXd J_m(6, m_total);
    // Copia solo le prime m_arm colonne (dopo la base) e azzera le restanti (gripper)
    if (m_arm > 0) {
        J_m.block(0, 0, 6, m_arm) = J_.block(0, 6, 6, m_arm);
    }
    if (m_total > m_arm) {
        J_m.block(0, m_arm, 6, m_total - m_arm).setZero();
    }

    // Centroidal momentum matrix Ag e sue sottomatrici (come nello script Python)
    pinocchio::computeCentroidalMap(model_, data_, q_);
    Eigen::MatrixXd Ag = data_.Ag; // 6 x nv
    Ag_b_ = Ag.leftCols(6);                 // 6x6 base
    Ag_m_ = Ag.block(0, 6, 6, m_total);     // 6 x m_total (mantiene anche i DoF del gripper)

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

    // Costruisci SE3 desiderata in WORLD
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

    pinocchio::SE3 T_des(R_des, p_des);

    // Errore di posa SE3 in WORLD: T_err = T_des * T_cur^{-1}
    //pinocchio::SE3 T_err = T_des * ee_placement.inverse();
    pinocchio::SE3 T_err = ee_placement.inverse()* T_des;

    // Logaritmo su SE(3) restituisce un twist nel frame locale del termine di errore.
    // Convertiamolo al frame WORLD usando l'Adjoint della posa corrente dell'EE.
    const Eigen::Matrix<double,6,1> err_body_al = pinocchio::log6(T_err).toVector(); 
    const Eigen::Matrix<double,6,6> Ad_world_from_cur = ee_placement.toActionMatrix();
    const Eigen::Matrix<double,6,1> err_world_al = Ad_world_from_cur * err_body_al; 
    error_pose_ee_.head<3>() = err_world_al.head<3>(); // lin in WORLD
    error_pose_ee_.tail<3>() = err_world_al.tail<3>(); // ang in WORLD

    // 6. CALCOLO ERRORE DI VELOCITÀ END-EFFECTOR

    // Stima velocità EE diretta da modello (WORLD)
    const pinocchio::Motion &ee_vel_motion = pinocchio::getFrameVelocity(model_, data_, ee_frame_id_, pinocchio::ReferenceFrame::WORLD);
    Eigen::VectorXd v_ee_meas(6); // [lin; ang] per coerenza con v_ee_des
    v_ee_meas.head<3>() = ee_vel_motion.linear();
    v_ee_meas.tail<3>() = ee_vel_motion.angular();

    // Debug: stampa la velocità EE misurata e controlla componenti non finite
    // {
    //     std::ostringstream oss;
    //     oss.setf(std::ios::fixed);
    //     oss.precision(6);
    //     oss << "ee_vel_motion (WORLD) lin=["
    //         << v_ee_meas[0] << ", " << v_ee_meas[1] << ", " << v_ee_meas[2]
    //         << "] ang=[" << v_ee_meas[3] << ", " << v_ee_meas[4] << ", " << v_ee_meas[5] << "]";
    //     RCLCPP_INFO(this->get_logger(), "%s", oss.str().c_str());
    //     bool has_nan = false;
    //     for (int i = 0; i < 6; ++i) {
    //         if (!std::isfinite(v_ee_meas[i])) { has_nan = true; break; }
    //     }
    //     if (has_nan) {
    //         RCLCPP_WARN(this->get_logger(), "ee_vel_motion contiene NaN/Inf: controlla J, v_gen_meas, frame WORLD");
    //     }
    // }

    error_vel_ee_ = v_ee_des - v_ee_meas;


    // 7. CALCOLO z2=z_kin: xdd_des - Jdot*v
    // Costruisci velocità proiettata per feed-forward (momentum-conserving): v_b_ik = -Ag_b^{-1} Ag_m * v_m
    // Usa le velocità giunto integrate (qd_int_) invece delle velocità misurate.

    if (qd_int_.size() != m_total) { qd_int_.resize(m_total); qd_int_.setZero(); }
    // Usa solo i DoF dell'arm; i DoF del gripper restano a zero
    Eigen::VectorXd v_m = qd_int_;
    if (m_total > m_arm) {
        v_m.segment(m_arm, m_total - m_arm).setZero();
    }
    Eigen::VectorXd v_b_ik(6); v_b_ik.setZero();
    try {
        v_b_ik = -Ag_b_.colPivHouseholderQr().solve(Ag_m_ * v_m);
    } catch (...) {
        v_b_ik = -Ag_b_.completeOrthogonalDecomposition().solve(Ag_m_ * v_m);
    }
    // v_gen_ik: base proiettata + giunti integrati (qd_int_)
    Eigen::VectorXd v_gen_ik(model_.nv); v_gen_ik.setZero();
    v_gen_ik.head<6>() = v_b_ik;
    v_gen_ik.segment(6, m_total) = v_m;

    // Forward kinematics per ottenere Jdot relativo alla velocità proiettata

    pinocchio::computeJointJacobiansTimeVariation(model_, data_, q_, v_gen_ik);
    Eigen::Matrix<double,6,Eigen::Dynamic> Jdot(6, model_.nv);
    pinocchio::getFrameJacobianTimeVariation(model_, data_, ee_frame_id_, pinocchio::ReferenceFrame::WORLD, Jdot);
    Eigen::VectorXd Jdot_v = Jdot * v_gen_ik;
    // Feed-forward kinematico
    Eigen::VectorXd z2_ff = acc_ee_des - Jdot_v;
    Eigen::VectorXd z2_fb = K_matrix_ * error_pose_ee_ + Kd_matrix_ * error_vel_ee_;
    Eigen::VectorXd z2    = z2_ff + z2_fb; // totale (solo per riferimento)

    // Stampa dell'errore di posa dell'end-effector (tutte e 6 le componenti)
    // {
    //     std::ostringstream oss;
    //     oss.setf(std::ios::fixed);
    //     oss.precision(6);
    //     oss << "error_pose_ee_ = [";
    //     for (int i = 0; i < error_pose_ee_.size(); ++i) {
    //         oss << error_pose_ee_[i];
    //         if (i + 1 < error_pose_ee_.size()) oss << ", ";
    //     }
    //     oss << "]";
    //     RCLCPP_INFO(this->get_logger(), "%s", oss.str().c_str());
    // }

    // 8. CALCOLO z1 = z_dyn = h0_dot = - dAg * v

    // Nota: il calcolo precedente via h0_dot è commentato; al momento non restituisce il comportamento atteso.
    // pinocchio::computeCentroidalMomentumTimeVariation(model_, data_, q_, v_gen, Eigen::VectorXd::Zero(model_.nv));
    // Eigen::Matrix<double,6,1> h0_dot = data_.dhg; // h_dot con a=0
    // Eigen::VectorXd z1_old = -h0_dot;

    // Eigen::VectorXd a_zero = Eigen::VectorXd::Zero(model_.nv);
    // pinocchio::forwardKinematics(model_, data_, q_, v_gen_ik, a_zero);
    // pinocchio::updateFramePlacements(model_, data_);

    pinocchio::computeCentroidalMapTimeVariation(model_, data_, q_, v_gen_ik);
    Eigen::MatrixXd dAg = data_.dAg; // 6 x nv
    Eigen::VectorXd z1 = -dAg * v_gen_ik; 

    // 9. RISOLVO SISTEMA LINEARE PER qdd
        Eigen::MatrixXd Aug(12, 6 + m_total);
        Aug << Ag_b_, Ag_m_,
            J_b,   J_m;

    // Stampa una tantum delle sottomatrici al primo ciclo di update
    if (update_iterations_after_guard_ == 1) {
        std::ostringstream os;
        os.setf(std::ios::scientific);
        os.precision(3);
        os << "Ag_b_ (" << Ag_b_.rows() << "x" << Ag_b_.cols() << ")\n" << Ag_b_ << "\n";
        os << "Ag_m_ (" << Ag_m_.rows() << "x" << Ag_m_.cols() << ")\n" << Ag_m_ << "\n";
        os << "J_b (" << J_b.rows() << "x" << J_b.cols() << ")\n" << J_b << "\n";
        os << "J_m (" << J_m.rows() << "x" << J_m.cols() << ")\n" << J_m;
        RCLCPP_INFO(this->get_logger(), "Aug submatrici al primo ciclo:\n%s", os.str().c_str());
    }
    // Risolvo sistema
    Eigen::VectorXd qdd_total;
    if (redundant_) {
        // Mantieni la dinamica z1 completa (6 righe) e usa solo 3 righe di J (posizione)
        // Matrice aumentata rettangolare: [Ag_b Ag_m; J_b(3) J_m(3)] -> 9 x (6+m_total)
        Eigen::MatrixXd Aug_red(9, 6 + m_total);
        Aug_red << Ag_b_, Ag_m_,
                   J_b.topRows(3),   J_m.topRows(3);

        // RHS: prima 6 componenti di z1 (dinamica completa), poi 3 componenti di z2 (solo posizione)
        Eigen::VectorXd rhs_ff_red(9); rhs_ff_red << z1, z2_ff.head(3);
        Eigen::VectorXd rhs_fb_red(9); rhs_fb_red << Eigen::VectorXd::Zero(6), z2_fb.head(3);
        Eigen::VectorXd rhs_total_red = rhs_ff_red + rhs_fb_red;

        // Pseudoinversa pesata come in clik1: Pinv = Wc_inv * A^T * (A * Wc_inv * A^T)^{-1}
        // Costruisci Wc_inv = blockdiag(I6, W_inv) con W_inv derivata da W_diag_
        Eigen::VectorXd W_inv_vec(m_total); W_inv_vec.setOnes();
        // Applica pesi ai soli primi m_arm DoF (giunti del braccio); i restanti (gripper) rimangono 1.0
        for (Eigen::Index i = 0; i < std::min<Eigen::Index>(m_total, static_cast<Eigen::Index>(arm_joints_.size())); ++i) {
            double w = W_diag_[static_cast<Eigen::Index>(i)];
            W_inv_vec[i] = (w != 0.0 ? 1.0 / w : 1.0);
        }
        Eigen::MatrixXd Wc_inv = Eigen::MatrixXd::Identity(6 + m_total, 6 + m_total);
        Wc_inv.block(6, 6, m_total, m_total) = W_inv_vec.asDiagonal();

        Eigen::MatrixXd A = Aug_red;
        Eigen::MatrixXd AwAt = A * Wc_inv * A.transpose();          
        const double damping = 1e-6;
        Eigen::MatrixXd AwAt_reg = AwAt + damping * Eigen::MatrixXd::Identity(AwAt.rows(), AwAt.cols());
        Eigen::MatrixXd AwAt_inv = AwAt_reg.ldlt().solve(Eigen::MatrixXd::Identity(AwAt.rows(), AwAt.cols()));
        Eigen::MatrixXd Pinv = Wc_inv * A.transpose() * AwAt_inv;    // (6+m_total) x 9

        Eigen::VectorXd acc_total = Pinv * rhs_total_red;
        qdd_total = acc_total.tail(m_total);
    } else {
        Eigen::VectorXd rhs_ff(12); rhs_ff << z1, z2_ff;                 // feed-forward include dinamica base
        Eigen::VectorXd rhs_fb(12); rhs_fb << Eigen::VectorXd::Zero(6), z2_fb; // feedback solo su seconda metà
        // Risoluzione unica: qdd = Aug^{-1} * (rhs_ff + rhs_fb)
        Eigen::VectorXd rhs_total = rhs_ff + rhs_fb;
        Eigen::VectorXd acc_total = Aug.colPivHouseholderQr().solve(rhs_total);
        qdd_total = acc_total.tail(m_total); // accelerazioni giunti totali (ff + fb)
    }

    // // Se compaiono NaN/Inf nelle accelerazioni, stampa diagnostica dettagliata
    // auto has_nonfinite = [](const Eigen::VectorXd &v) {
    //     for (Eigen::Index i = 0; i < v.size(); ++i) {
    //         if (!std::isfinite(v[i])) {
    //             return true;
    //         }
    //     }
    //     return false;
    // };
    // const bool nan_total = has_nonfinite(qdd_total);
    // if (nan_total && update_iterations_after_guard_ <= 20) {
    //     // Indici non finiti
    //     auto nonfinite_indices = [](const Eigen::VectorXd &v) {
    //         std::ostringstream os; os << "[";
    //         bool first=true; for (Eigen::Index i=0;i<v.size();++i){ if(!std::isfinite(v[i])){ if(!first) os<<","; os<<i; first=false; }}
    //         os << "]"; return os.str();
    //     };
    //     // Norme scalari utili
    //     double n_z1 = z1.norm();
    //     double n_z2ff = z2_ff.norm();
    //     double n_z2fb = z2_fb.norm();
    //     // Check Jdot_v se calcolato
    //     // (Jdot_v già esiste sopra)
    //     double n_Jdotv = Jdot_v.norm();
    //     // Rank e SVD dell'Augmentata
    //     Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(Aug);
    //     int rank = qr.rank();
    //     Eigen::JacobiSVD<Eigen::MatrixXd> svd(Aug, Eigen::ComputeThinU | Eigen::ComputeThinV);
    //     const auto &S = svd.singularValues();
    //     double s_max = (S.size()>0? S(0) : 0.0);
    //     double s_min = (S.size()>0? S(S.size()-1) : 0.0);
    //     // Riepilogo breve su prime voci per i giunti del braccio
    //     size_t n_print = std::min(static_cast<size_t>(m_total), arm_joints_.size());
    //     std::ostringstream os_total;
    //     os_total.setf(std::ios::fixed); os_total.precision(3);
    //     for (size_t i=0;i<n_print;++i){
    //         os_total << (std::isfinite(qdd_total(static_cast<Eigen::Index>(i)))? qdd_total(static_cast<Eigen::Index>(i)) : std::numeric_limits<double>::quiet_NaN());
    //         if (i+1<n_print){ os_total << ", "; }
    //     }
    //     RCLCPP_WARN(this->get_logger(), "NaN in qdd_total rilevato. idx=%s",
    //                 nonfinite_indices(qdd_total).c_str());
    //     RCLCPP_WARN(this->get_logger(), "Norme: |z1|=%.3e |z2_ff|=%.3e |z2_fb|=%.3e |Jdot*v|=%.3e",
    //                 n_z1, n_z2ff, n_z2fb, n_Jdotv);
    //     RCLCPP_WARN(this->get_logger(), "Aug dims=%ldx%ld rank=%d s_max=%.3e s_min=%.3e", Aug.rows(), Aug.cols(), rank, s_max, s_min);
    //     RCLCPP_WARN(this->get_logger(), "qdd_total(first %zu)=[%s]", n_print, os_total.str().c_str());
    // }

    // Integrazione accelerazioni -> velocità -> posizione
    const rclcpp::Time now_int = this->now();
    double dt = (now_int - last_update_time_).seconds();
    last_update_time_ = now_int;
    dt = std::clamp(dt, 0.0, 0.02); // massimo 20 ms

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

    // Debug: stampa accelerazioni giunti (rad/s^2) e posizioni (deg) con throttle
    // {
    //     const size_t n_arm = arm_joints_.size();
    //     const size_t n_avail = static_cast<size_t>(std::min<Eigen::Index>(q_cmd_.size(), qdd_total.size()));
    //     const size_t n = std::min(n_arm, n_avail);
    //     if (n > 0) {
    //         std::ostringstream oss;
    //         oss.setf(std::ios::fixed); oss.precision(3);
    //         oss << "qdd[rad/s^2] & q[deg]: ";
    //         for (size_t i = 0; i < n; ++i) {
    //             double acc = std::isfinite(qdd_total[static_cast<Eigen::Index>(i)]) ? qdd_total[static_cast<Eigen::Index>(i)] : 0.0;
    //             double qdeg = (std::isfinite(q_cmd_[static_cast<Eigen::Index>(i)]) ? q_cmd_[static_cast<Eigen::Index>(i)] : 0.0) * 57.29577951308232;
    //             oss << arm_joints_[i] << ": qdd=" << acc << ", q=" << qdeg;
    //             if (i + 1 < n) oss << " | ";
    //         }
    //         RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500, "%s", oss.str().c_str());
    //     }
    // }

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
