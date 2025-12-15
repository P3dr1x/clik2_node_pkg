#ifndef CLIK2_NODE_PKG_CLIK_UAM_NODE_HPP_
#define CLIK2_NODE_PKG_CLIK_UAM_NODE_HPP_

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "px4_msgs/msg/vehicle_local_position.hpp"
#include "px4_msgs/msg/vehicle_attitude.hpp"
#include "px4_msgs/msg/vehicle_odometry.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/accel.hpp"
#include "interbotix_xs_msgs/msg/joint_group_command.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "geometry_msgs/msg/transform_stamped.hpp" 
#include "pinocchio/multibody/model.hpp"
#include "pinocchio/multibody/data.hpp"
#include <unordered_map>

class ClikUamNode : public rclcpp::Node
{
public:
    ClikUamNode();

private:
    // Metodi
    // get_and_transform_desired_pose rimosso: ora la posa desiderata arriva dal planner esterno
    void vehicle_local_position_callback(const px4_msgs::msg::VehicleLocalPosition::SharedPtr msg);
    void vehicle_attitude_callback(const px4_msgs::msg::VehicleAttitude::SharedPtr msg);
    void vehicle_odometry_callback(const px4_msgs::msg::VehicleOdometry::SharedPtr msg);
    void gazebo_odometry_callback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void real_drone_pose_callback(const geometry_msgs::msg::PoseStamped::ConstSharedPtr msg);
    void real_drone_twist_callback(const geometry_msgs::msg::Twist::SharedPtr msg);
    void gazebo_pose_callback(const geometry_msgs::msg::PoseArray::SharedPtr msg);
    void joint_state_callback(const sensor_msgs::msg::JointState::SharedPtr msg);
    void desired_pose_callback(const geometry_msgs::msg::Pose::SharedPtr msg);
    void desired_velocity_callback(const geometry_msgs::msg::Twist::SharedPtr msg);
    void desired_accel_callback(const geometry_msgs::msg::Accel::SharedPtr msg);
    void update();

    // Variabili membro
    geometry_msgs::msg::Pose desired_ee_pose_world_;
    geometry_msgs::msg::Twist desired_ee_velocity_world_;
    Eigen::VectorXd desired_ee_velocity_vec_;
    geometry_msgs::msg::Accel desired_ee_accel_world_;
    bool desired_ee_pose_world_ready_ = false;
    bool desired_ee_velocity_ready_ = false;
    bool waiting_log_printed_ = false;

    // Subscribers ai topics dove sono pubblicati i dati di posa del drone
    rclcpp::Subscription<px4_msgs::msg::VehicleLocalPosition>::SharedPtr vehicle_local_position_sub_;
    rclcpp::Subscription<px4_msgs::msg::VehicleAttitude>::SharedPtr vehicle_attitude_sub_;
    rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr vehicle_odom_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr gazebo_odom_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr real_drone_pose_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr real_drone_twist_sub_;
    px4_msgs::msg::VehicleLocalPosition vehicle_local_position_;
    px4_msgs::msg::VehicleAttitude vehicle_attitude_;
    px4_msgs::msg::VehicleOdometry vehicle_odom_;
    nav_msgs::msg::Odometry gazebo_odom_;
    geometry_msgs::msg::Twist real_drone_twist_;
    bool has_vehicle_local_position_ = false;
    bool has_vehicle_attitude_ = false;
    bool has_vehicle_odom_ = false;
    bool has_gazebo_odom_ = false;
    bool has_real_drone_twist_ = false;
    bool use_gazebo_pose_;

    rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr gazebo_pose_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Pose>::SharedPtr desired_ee_global_pose_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr desired_ee_velocity_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Accel>::SharedPtr desired_ee_accel_sub_;

    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    // tf_broadcaster_ rimosso: non utilizzato in questo nodo

    // Pinocchio model and data
    pinocchio::Model model_;
    pinocchio::Data data_;
    pinocchio::Data::Matrix6x J_;
    pinocchio::FrameIndex ee_frame_id_;

    // Sottomatrici della Centroidal Momentum Matrix (Ag)
    Eigen::MatrixXd Ag_b_; // 6x6
    Eigen::MatrixXd Ag_m_; // 6xm (m = nv - 6)
    Eigen::VectorXd q_;
    Eigen::VectorXd qd_;
    Eigen::VectorXd error_pose_ee_;
    Eigen::VectorXd error_vel_ee_;
    Eigen::MatrixXd K_matrix_;
    Eigen::MatrixXd Kd_matrix_;

    // Parametri di controllo (proporzionale e derivativo)
    double k_err_pos_;
    double k_err_vel_;
    // Limite di velocità giunti (rad/s)
    double joint_vel_limit_ = 3.14;

    // Flag e timestamp per gestione messaggi desiderati
    bool desired_ee_accel_ready_ = false;
    bool have_desired_msg_ = false;
    rclcpp::Time last_desired_msg_time_;

    // Ridondanza cinematica (ignora orientazione se true)
    bool redundant_ = false;

    // Timing controllo
    rclcpp::Time last_update_time_;
    double desired_timeout_sec_ = 0.0;

    rclcpp::TimerBase::SharedPtr control_timer_;

    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
    // Publisher per Interbotix xs_sdk/xs_sdk_sim
    rclcpp::Publisher<interbotix_xs_msgs::msg::JointGroupCommand>::SharedPtr joint_group_pub_;
    // Publisher per ros2_control (SITL)
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr arm_controller_pub_;
    rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr ee_world_pose_pub_;

    sensor_msgs::msg::JointState current_joint_state_;
    bool has_current_joint_state_ = false;
    // Stima velocità giunti quando non fornita dal topic /joint_states
    std::unordered_map<std::string, double> prev_joint_positions_;
    rclcpp::Time last_joint_state_time_;

    // (Vecchi nomi k_err_x_, k_err_xd_ rimossi e sostituiti con k_err_pos_, k_err_vel_)

    std::vector<std::string> arm_joints_;
    // Pesi per pseudoinversa pesata dei giunti del braccio
    Eigen::VectorXd W_diag_;

    // Parametri runtime
    std::string robot_name_;
    bool real_system_ = true;

    // Buffer integrazione accelerazioni -> velocità -> posizione (giunti braccio)
    Eigen::VectorXd q_cmd_;    // posizioni comando braccio
    Eigen::VectorXd q_pos_int_; // posizione integrata (totale)
    Eigen::VectorXd qd_int_; // velocità integrata (totale)
    bool accel_buffers_initialized_ = false;

    // Stima velocità EE da differenziazione
    geometry_msgs::msg::Pose last_ee_pose_world_;
    rclcpp::Time last_ee_time_;
    bool have_last_ee_ = false;

    // Stima twist base (drone) da differenziazione
    geometry_msgs::msg::Pose last_drone_pose_world_;
    rclcpp::Time last_drone_time_;
    bool have_last_drone_ = false;

    // Contatore delle iterazioni dell'update dopo il superamento della guardia
    std::size_t update_iterations_after_guard_ = 0;
};

#endif // CLIK2_NODE_PKG_CLIK_UAM_NODE_HPP_
