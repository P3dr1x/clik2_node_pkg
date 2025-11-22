#include "clik2_node_pkg/clik_uam_node.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <memory>
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/crba.hpp" // Composite Rigid Body Algorithm (per inerzia)
#include "pinocchio/algorithm/rnea.hpp"    // nonLinearEffects, gravity, RNEA
#include "pinocchio/algorithm/energy.hpp"
#include "trajectory_msgs/msg/joint_trajectory_point.hpp"
#include "interbotix_xs_msgs/msg/joint_group_command.hpp" // Include necessario per il nuovo tipo di messaggio
#include "px4_ros_com/frame_transforms.h"
#include "ament_index_cpp/get_package_share_directory.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "tf2_ros/transform_broadcaster.h"

ClikUamNode::ClikUamNode() : Node("clik_uam_node"), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_)
{
    RCLCPP_INFO(this->get_logger(), "Nodo clik_uam_node avviato.");

    this->declare_parameter<bool>("use_gazebo_pose", true);
    this->get_parameter("use_gazebo_pose", use_gazebo_pose_);

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
        RCLCPP_INFO(this->get_logger(), "Utilizzo della posa dal nodo real_drone_pose_pub (/real_t960a_pose).");
        real_drone_pose_sub_ = this->create_subscription<geometry_msgs::msg::Pose>(
            "/real_t960a_pose", 10, std::bind(&ClikUamNode::real_drone_pose_callback, this, std::placeholders::_1));
    }

    // Odometria PX4 (twist body-fixed in FRD) => convertiamo a FLU
    vehicle_odom_sub_ = this->create_subscription<px4_msgs::msg::VehicleOdometry>(
        "/fmu/out/vehicle_odometry", 10, std::bind(&ClikUamNode::vehicle_odometry_callback, this, std::placeholders::_1));

    // Joint states: se reale, leggi da /<robot_name>/joint_states (xs_sdk); se SITL, da /joint_states (ros2_control)
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
    qd_.resize(model_.nv);
    inertia_matrix_.resize(model_.nv, model_.nv);
    J_.resize(6, model_.nv);
    error_pose_ee_.resize(6);
    desired_ee_velocity_vec_.setZero(6);
    arm_joints_ = {"waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"};

    declare_parameter("k_err_x_", 10.0);
    declare_parameter("k_err_xd_", 0.0);
    declare_parameter("use_odometry_twist_", true);
    k_err_x_ = get_parameter("k_err_x_").as_double();
    k_err_xd_ = get_parameter("k_err_xd_").as_double();
    use_odometry_twist_ = get_parameter("use_odometry_twist_").as_bool();
    K_matrix_ = Eigen::MatrixXd::Identity(6, 6) * k_err_x_;
    Kd_matrix_ = Eigen::MatrixXd::Identity(6, 6) * k_err_xd_;

    // Inizializza buffer comandi giunti
    q_cmd_.resize(6);
    qd_cmd_.resize(6);
    q_cmd_.setZero();
    qd_cmd_.setZero();

    // Timer controllo
    control_timer_ = this->create_wall_timer(std::chrono::milliseconds(10), std::bind(&ClikUamNode::update, this));  // 100 Hz
}

void ClikUamNode::desired_pose_callback(const geometry_msgs::msg::Pose::SharedPtr msg) {
    desired_ee_pose_world_ = *msg;
    desired_ee_pose_world_ready_ = true;
    RCLCPP_INFO(this->get_logger(), "Nuova posa desiderata ricevuta: x=%.3f y=%.3f z=%.3f", msg->position.x, msg->position.y, msg->position.z);
}

void ClikUamNode::desired_velocity_callback(const geometry_msgs::msg::Twist::SharedPtr msg) {
    desired_ee_velocity_world_ = *msg;
    desired_ee_velocity_ready_ = true; // indica che siamo in una fase di tracking
}

void ClikUamNode::desired_accel_callback(const geometry_msgs::msg::Accel::SharedPtr msg) {
    desired_ee_accel_world_ = *msg;
}

void ClikUamNode::vehicle_local_position_callback(const px4_msgs::msg::VehicleLocalPosition::SharedPtr msg)
{
    // ATTENZIONE: È GIUSTO FLU O ERA MEGLIO NED?
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
    // Conversione manuale da FRD (PX4) a FLU (ROS2)
    // FRD (Forward, Right, Down) -> FLU (Forward, Left, Up)
    // La conversione consiste nell'invertire gli assi Y e Z  //VERIFICARE SE È GIUSTO
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

void ClikUamNode::real_drone_pose_callback(const geometry_msgs::msg::Pose::SharedPtr msg)
{
    // Assumiamo che /real_t960a_pose sia già in frame world FLU con yaw eliminato (o come concordato)
    vehicle_local_position_.x = msg->position.x;
    vehicle_local_position_.y = msg->position.y;
    vehicle_local_position_.z = msg->position.z;

    vehicle_attitude_.q[0] = msg->orientation.w;
    vehicle_attitude_.q[1] = msg->orientation.x;
    vehicle_attitude_.q[2] = msg->orientation.y;
    vehicle_attitude_.q[3] = msg->orientation.z;
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
    current_joint_state_ = *msg;
    has_current_joint_state_ = true;
}

void ClikUamNode::update()
{
    if (!desired_ee_pose_world_ready_ || !has_current_joint_state_ || current_joint_state_.name.empty() || !has_vehicle_local_position_ || !has_vehicle_attitude_)
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

    // --- AGGIORNA STATO DEL ROBOT ---
    // 1. Popola il vettore di configurazione 'q' di Pinocchio
    q_[0] = vehicle_local_position_.x;
    q_[1] = vehicle_local_position_.y;
    q_[2] = vehicle_local_position_.z;
    q_[3] = vehicle_attitude_.q[1]; // x
    q_[4] = vehicle_attitude_.q[2]; // y
    q_[5] = vehicle_attitude_.q[3]; // z
    q_[6] = vehicle_attitude_.q[0]; // w

    // 2. Leggi lo stato attuale dei giunti del braccio
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

    // 3. Velocità (per ora a zero)
    qd_.setZero();

    // --- CALCOLO MATRICI INERZIA E JACOBIANI ---
    pinocchio::crba(model_, data_, q_);
    data_.M.triangularView<Eigen::StrictlyLower>() = data_.M.transpose().triangularView<Eigen::StrictlyLower>();
    inertia_matrix_ = data_.M;

    pinocchio::computeFrameJacobian(model_, data_, q_, ee_frame_id_, pinocchio::ReferenceFrame::WORLD, J_);

    // cinematica diretta per posa assoluta dell'end-effector
    pinocchio::forwardKinematics(model_, data_, q_);
    pinocchio::updateFramePlacements(model_, data_);
    const pinocchio::SE3& ee_placement = data_.oMf[ee_frame_id_];
    // conversione a geometry_msgs::Pose
    geometry_msgs::msg::Pose current_ee_pose_world;
    current_ee_pose_world.position.x = ee_placement.translation().x();
    current_ee_pose_world.position.y = ee_placement.translation().y();
    current_ee_pose_world.position.z = ee_placement.translation().z();
    Eigen::Quaterniond ee_q(ee_placement.rotation());
    current_ee_pose_world.orientation.x = ee_q.x();
    current_ee_pose_world.orientation.y = ee_q.y();
    current_ee_pose_world.orientation.z = ee_q.z();
    current_ee_pose_world.orientation.w = ee_q.w();

    // Pubblica la posa assoluta dell'end-effector
    ee_world_pose_pub_->publish(current_ee_pose_world);

    // Stima velocità EE via differenziazione numerica
    Eigen::VectorXd xdot_est(6); xdot_est.setZero();
    const rclcpp::Time now = this->now();
    if (have_last_ee_) {
        double dt_est = (now - last_ee_time_).seconds();
        if (dt_est > 1e-6) {
            // lineare
            xdot_est(0) = (current_ee_pose_world.position.x - last_ee_pose_world_.position.x) / dt_est;
            xdot_est(1) = (current_ee_pose_world.position.y - last_ee_pose_world_.position.y) / dt_est;
            xdot_est(2) = (current_ee_pose_world.position.z - last_ee_pose_world_.position.z) / dt_est;
            // angolare da delta quaternion
            Eigen::Quaterniond q_now(current_ee_pose_world.orientation.w, current_ee_pose_world.orientation.x, current_ee_pose_world.orientation.y, current_ee_pose_world.orientation.z);
            Eigen::Quaterniond q_prev(last_ee_pose_world_.orientation.w, last_ee_pose_world_.orientation.x, last_ee_pose_world_.orientation.y, last_ee_pose_world_.orientation.z);
            Eigen::Quaterniond dq = q_now * q_prev.conjugate();
            if (dq.w() < 0) dq.coeffs() *= -1.0; // percorso corto
            Eigen::AngleAxisd aa(dq);
            Eigen::Vector3d ang = aa.axis() * aa.angle();
            xdot_est.segment<3>(3) = ang / dt_est;
        }
    }
    last_ee_pose_world_ = current_ee_pose_world;
    last_ee_time_ = now;
    have_last_ee_ = true;

    // --- Twist base (drone): usa odometria se disponibile, altrimenti differenziazione ---
    geometry_msgs::msg::Pose current_drone_pose;
    current_drone_pose.position.x = vehicle_local_position_.x;
    current_drone_pose.position.y = vehicle_local_position_.y;
    current_drone_pose.position.z = vehicle_local_position_.z;
    current_drone_pose.orientation.x = vehicle_attitude_.q[1];
    current_drone_pose.orientation.y = vehicle_attitude_.q[2];
    current_drone_pose.orientation.z = vehicle_attitude_.q[3];
    current_drone_pose.orientation.w = vehicle_attitude_.q[0];

    // Velocità WORLD richieste dall'utente
    Eigen::Vector3d vlin_base_world = Eigen::Vector3d::Zero();
    Eigen::Vector3d omega_base_world = Eigen::Vector3d::Zero();
    if (use_odometry_twist_ && has_vehicle_odom_) {
        // msg twist è in FRD body. Converti a FLU body, poi ruota in WORLD.
        Eigen::Vector3d v_body_flu(vehicle_odom_.velocity[0], -vehicle_odom_.velocity[1], -vehicle_odom_.velocity[2]);
        Eigen::Vector3d w_body_flu(vehicle_odom_.angular_velocity[0], -vehicle_odom_.angular_velocity[1], -vehicle_odom_.angular_velocity[2]);
        Eigen::Quaterniond q_world_base(vehicle_attitude_.q[0], vehicle_attitude_.q[1], vehicle_attitude_.q[2], vehicle_attitude_.q[3]);
        Eigen::Matrix3d Rwb = q_world_base.normalized().toRotationMatrix();
        vlin_base_world = Rwb * v_body_flu;
        omega_base_world = Rwb * w_body_flu;
    } else if (have_last_drone_) {
        double dtb = (now - last_drone_time_).seconds();
        if (dtb > 1e-6) {
            vlin_base_world.x() = (current_drone_pose.position.x - last_drone_pose_world_.position.x) / dtb;
            vlin_base_world.y() = (current_drone_pose.position.y - last_drone_pose_world_.position.y) / dtb;
            vlin_base_world.z() = (current_drone_pose.position.z - last_drone_pose_world_.position.z) / dtb;

            Eigen::Quaterniond q_now(current_drone_pose.orientation.w, current_drone_pose.orientation.x, current_drone_pose.orientation.y, current_drone_pose.orientation.z);
            Eigen::Quaterniond q_prev(last_drone_pose_world_.orientation.w, last_drone_pose_world_.orientation.x, last_drone_pose_world_.orientation.y, last_drone_pose_world_.orientation.z);
            Eigen::Quaterniond dq = q_now * q_prev.conjugate();
            if (dq.w() < 0) dq.coeffs() *= -1.0;
            Eigen::AngleAxisd aa(dq);
            omega_base_world = aa.axis() * aa.angle() / dtb;
        }
    }
    last_drone_pose_world_ = current_drone_pose;
    last_drone_time_ = now;
    have_last_drone_ = true;

    // Estrai i blocchi necessari
    Eigen::MatrixXd H_b = inertia_matrix_.topLeftCorner(6, 6);
    Eigen::MatrixXd H_bm = inertia_matrix_.topRightCorner(6, model_.nv - 6);
    Eigen::MatrixXd J_b = J_.leftCols(6);
    Eigen::MatrixXd J_m = J_.rightCols(model_.nv - 6);

    // --- CALCOLO ERRORE DI POSA ---
    // Converti pose in SE3 di Pinocchio
    pinocchio::SE3 desired_pose_se3(
        pinocchio::SE3::Quaternion(desired_ee_pose_world_.orientation.w, desired_ee_pose_world_.orientation.x, desired_ee_pose_world_.orientation.y, desired_ee_pose_world_.orientation.z),
        Eigen::Vector3d(desired_ee_pose_world_.position.x, desired_ee_pose_world_.position.y, desired_ee_pose_world_.position.z)
    );
    pinocchio::SE3 current_pose_se3(
        pinocchio::SE3::Quaternion(current_ee_pose_world.orientation.w, current_ee_pose_world.orientation.x, current_ee_pose_world.orientation.y, current_ee_pose_world.orientation.z),
        Eigen::Vector3d(current_ee_pose_world.position.x, current_ee_pose_world.position.y, current_ee_pose_world.position.z)
    );

    // Calcola l'errore 6D
    error_pose_ee_ = pinocchio::log6(desired_pose_se3 * current_pose_se3.inverse()).toVector();

    // --- Costruzione riferimento di velocità desiderata e errore di velocità ---
    if (desired_ee_velocity_vec_.size() != 6) desired_ee_velocity_vec_.resize(6);
    desired_ee_velocity_vec_.setZero();
    if (desired_ee_velocity_ready_) {
        desired_ee_velocity_vec_(0) = desired_ee_velocity_world_.linear.x;
        desired_ee_velocity_vec_(1) = desired_ee_velocity_world_.linear.y;
        desired_ee_velocity_vec_(2) = desired_ee_velocity_world_.linear.z;
        desired_ee_velocity_vec_(3) = desired_ee_velocity_world_.angular.x;
        desired_ee_velocity_vec_(4) = desired_ee_velocity_world_.angular.y;
        desired_ee_velocity_vec_(5) = desired_ee_velocity_world_.angular.z;
    }
    error_vel_ee_.resize(6);
    error_vel_ee_ = desired_ee_velocity_vec_ - xdot_est;

    // --- Acc desiderata EE ---
    Eigen::VectorXd xdd_des(6); xdd_des.setZero();
    xdd_des(0) = desired_ee_accel_world_.linear.x;
    xdd_des(1) = desired_ee_accel_world_.linear.y;
    xdd_des(2) = desired_ee_accel_world_.linear.z;
    xdd_des(3) = desired_ee_accel_world_.angular.x;
    xdd_des(4) = desired_ee_accel_world_.angular.y;
    xdd_des(5) = desired_ee_accel_world_.angular.z;

    // --- Stima velocità generalizzata v (Pinocchio richiede LOCAL per la base free-flyer) ---
    // Converti WORLD -> LOCAL usando R^T
    Eigen::Quaterniond q_world_base(vehicle_attitude_.q[0], vehicle_attitude_.q[1], vehicle_attitude_.q[2], vehicle_attitude_.q[3]);
    Eigen::Matrix3d Rwb = q_world_base.normalized().toRotationMatrix();
    Eigen::Vector3d omega_base_local = Rwb.transpose() * omega_base_world;
    Eigen::Vector3d vlin_base_local = Rwb.transpose() * vlin_base_world;

    Eigen::VectorXd v_gen(model_.nv); v_gen.setZero();
    v_gen.segment<3>(0) = omega_base_local;
    v_gen.segment<3>(3) = vlin_base_local;
    // qdot giunti (se disponibili)
    // Nota: qui non effettuiamo il mapping preciso idx_v per semplicità; si può estendere usando model.idx_v
    // Manteniamo qd giunti a zero se non abbiamo una stima affidabile

    // --- xdd_EE_0 tramite Pinocchio: a=0, solo termini Jdot*v ---
    Eigen::VectorXd a_zero = Eigen::VectorXd::Zero(model_.nv);
    pinocchio::forwardKinematics(model_, data_, q_, v_gen, a_zero);
    pinocchio::updateFramePlacements(model_, data_);
    // Accelerazione spaziale del frame EE in WORLD
    pinocchio::Motion a_ee = pinocchio::getFrameAcceleration(model_, data_, ee_frame_id_, pinocchio::ReferenceFrame::WORLD);
    Eigen::VectorXd xdd0(6);
    // Motion memorizza [angular; linear]
    xdd0.segment<3>(0) = a_ee.linear();
    xdd0.segment<3>(3) = a_ee.angular();

    // --- h0_dot: componenti non lineari senza gravità (Coriolis) sui primi 6 gradi
    Eigen::VectorXd nle = pinocchio::nonLinearEffects(model_, data_, q_, v_gen);
    Eigen::VectorXd ggen(model_.nv); ggen.setZero();
    pinocchio::computeGeneralizedGravity(model_, data_, q_);
    ggen = data_.g; // Pinocchio riempie data_.g
    Eigen::VectorXd h0_dot = (nle - ggen).head(6);

    // --- z2: xdd_des - xdd0 + Kx*e + Kd*edot ---
    Eigen::VectorXd z2 = xdd_des - xdd0 + K_matrix_ * error_pose_ee_ + Kd_matrix_ * error_vel_ee_;

    // --- z1: h_dot - h0_dot, con h_dot ignorato -> z1 = -h0_dot ---
    Eigen::VectorXd z1 = -h0_dot;

    // --- Matrice aumentata e soluzione ---
    Eigen::Index m = model_.nv - 6; // DoF braccio
    Eigen::MatrixXd Aug(12, 6 + m);
    Aug << H_b, H_bm,
           J_b, J_m;
    Eigen::VectorXd rhs(12);
    rhs << z1, z2;
    Eigen::VectorXd acc = Aug.colPivHouseholderQr().solve(rhs);
    Eigen::VectorXd qdd = acc.tail(m);

    // Inizializza q_cmd_ con stato attuale alla prima iterazione
    if (q_cmd_.size() != static_cast<int>(arm_joints_.size())) {
        q_cmd_.resize(arm_joints_.size());
        qd_cmd_.resize(arm_joints_.size());
        for (size_t i = 0; i < arm_joints_.size(); ++i) {
            double current_pos = 0.0;
            const auto& joint_name_to_find = arm_joints_[i];
            for (size_t j = 0; j < current_joint_state_.name.size(); ++j) {
                if (current_joint_state_.name[j] == joint_name_to_find) {
                    current_pos = current_joint_state_.position[j];
                    break;
                }
            }
            q_cmd_[i] = current_pos;
            qd_cmd_[i] = 0.0;
        }
    }

    // Integrazione numerica semplice (Euler) su 100 Hz
    double dt = 0.01;
    for (size_t i = 0; i < arm_joints_.size(); ++i) {
        qd_cmd_[i] += qdd[i] * dt;
        q_cmd_[i] += qd_cmd_[i] * dt;
    }

    // Prepara messaggio posizione target
    std_msgs::msg::Float64MultiArray command_msg;
    command_msg.data.resize(arm_joints_.size());
    for (size_t i = 0; i < arm_joints_.size(); ++i) {
        command_msg.data[i] = q_cmd_[i];
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
