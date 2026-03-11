#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>

// Nodo: real_drone_vel_pub
// Funzione: pubblica /real_t960a_twist (geometry_msgs/Twist)
//  - velocità lineare nel frame WORLD-FLU fissato alla direzione di marcia iniziale
//  - velocità angolare nel frame body FLU
// Usa:
//  - /fmu/out/vehicle_odometry per ricavare yaw iniziale (dal quaternione),
//    velocità lineare (NED) e angolare (body FRD)

class RealDroneVelPub : public rclcpp::Node {
public:
  RealDroneVelPub() : Node("real_drone_vel_pub") {
    using std::placeholders::_1;
    RCLCPP_INFO(get_logger(), "Avvio real_drone_vel_pub");

    // Parametri
    this->declare_parameter<bool>("use_mocap_omega", false);
    this->declare_parameter<double>("omega_lp_tau", 0.05); // [s], 0 => no filtro
    use_mocap_omega_ = this->get_parameter("use_mocap_omega").as_bool();
    omega_lp_tau_ = this->get_parameter("omega_lp_tau").as_double();

    twist_pub_ = create_publisher<geometry_msgs::msg::Twist>("/real_t960a_twist", 10);

    vehicle_odom_sub_ = create_subscription<px4_msgs::msg::VehicleOdometry>(
        "/fmu/out/vehicle_odometry", rclcpp::SensorDataQoS(),
        std::bind(&RealDroneVelPub::vehicle_odom_cb, this, _1));

    if (use_mocap_omega_) {
      mocap_pose_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
          "/t960a/pose", rclcpp::SensorDataQoS(),
          std::bind(&RealDroneVelPub::mocap_pose_cb, this, _1));
      RCLCPP_INFO(get_logger(), "use_mocap_omega=true -> stimo omega da /t960a/pose (dquat)");
    }

    RCLCPP_INFO(get_logger(), "omega_lp_tau=%.3f s", omega_lp_tau_);
  }

private:
  void vehicle_odom_cb(const px4_msgs::msg::VehicleOdometry::SharedPtr msg) {
    vehicle_odom_ = *msg;
    has_odom_ = true;

    // Inizializza una sola volta lo yaw di riferimento dal quaternione di odometry
    if (!yaw_offset_initialized_) {
      // PX4 VehicleOdometry.q è NED->FRD (w,x,y,z)
      Eigen::Quaterniond q_px4(vehicle_odom_.q[0], vehicle_odom_.q[1], vehicle_odom_.q[2], vehicle_odom_.q[3]);
      // Converti FRD -> FLU: (w, x, -y, -z)
      Eigen::Quaterniond q_flu(q_px4.w(), q_px4.x(), -q_px4.y(), -q_px4.z());
      q_flu.normalize();
      // Estrai yaw dal quaternione FLU
      Eigen::Vector3d eul = q_flu.toRotationMatrix().eulerAngles(2,1,0); // yaw, pitch, roll
      yaw_offset_ = eul[0];
      yaw_offset_initialized_ = true;
      RCLCPP_INFO(get_logger(), "Yaw iniziale (odometry) catturato: %.3f rad", yaw_offset_);
    }
    try_publish();
  }

  void mocap_pose_cb(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    // Stima omega body-FLU da delta quaternione.
    // Assunzione: orientation rappresenta R_wb (body rispetto world) in convenzione FLU.
    rclcpp::Time t = msg->header.stamp;
    if (t.nanoseconds() == 0) {
      t = this->now();
    }

    Eigen::Quaterniond q_wb(
        msg->pose.orientation.w,
        msg->pose.orientation.x,
        msg->pose.orientation.y,
        msg->pose.orientation.z);

    if (!std::isfinite(q_wb.w()) || !std::isfinite(q_wb.x()) || !std::isfinite(q_wb.y()) ||
        !std::isfinite(q_wb.z())) {
      return;
    }
    q_wb.normalize();

    if (!mocap_q_init_) {
      mocap_q_prev_ = q_wb;
      mocap_t_prev_ = t;
      mocap_q_init_ = true;
      return;
    }

    // Evita flip di segno del quaternione
    if (mocap_q_prev_.coeffs().dot(q_wb.coeffs()) < 0.0) {
      q_wb.coeffs() *= -1.0;
    }

    const double dt = (t - mocap_t_prev_).seconds();
    if (!(dt > 1e-4) || dt > 0.5) {
      mocap_q_prev_ = q_wb;
      mocap_t_prev_ = t;
      return;
    }

    // Rotazione relativa in coordinate body precedente: R_rel = R_prev^T * R_curr
    Eigen::Quaterniond q_rel = mocap_q_prev_.conjugate() * q_wb;
    q_rel.normalize();

    Eigen::AngleAxisd aa(q_rel);
    const double angle = aa.angle();
    const Eigen::Vector3d axis = aa.axis();
    if (!std::isfinite(angle) || !axis.allFinite()) {
      mocap_q_prev_ = q_wb;
      mocap_t_prev_ = t;
      return;
    }

    omega_mocap_raw_ = (angle / dt) * axis; // [rad/s] in body-FLU (approx)
    has_omega_mocap_ = true;

    mocap_q_prev_ = q_wb;
    mocap_t_prev_ = t;

    // Se l'odometria è già disponibile, pubblica appena arriva una nuova omega
    try_publish();
  }

  static Eigen::Vector3d lowpassIIR(const Eigen::Vector3d &x_raw, const Eigen::Vector3d &x_prev,
                                   double dt, double tau) {
    if (!(tau > 0.0) || !(dt > 0.0)) {
      return x_raw;
    }
    const double alpha = std::exp(-dt / tau);
    return alpha * x_prev + (1.0 - alpha) * x_raw;
  }

  void try_publish() {
    if (!(has_odom_ && yaw_offset_initialized_)) {
      return;
    }

    // Velocità lineare: campo velocity in NED [N, E, D]
    const double x_n = static_cast<double>(vehicle_odom_.velocity[0]);
    const double y_e = static_cast<double>(vehicle_odom_.velocity[1]);
    const double z_d = static_cast<double>(vehicle_odom_.velocity[2]);

    const double cos_y0 = std::cos(yaw_offset_);
    const double sin_y0 = std::sin(yaw_offset_);

    // Rotazione SOLO nel piano orizzontale per fissare il frame WORLD-FLU
    // X_world = forward rispetto heading iniziale
    // Y_world = left rispetto heading iniziale
    const double x_world =  cos_y0 * x_n + sin_y0 * y_e;
    const double y_world =  sin_y0 * x_n - cos_y0 * y_e;
    const double z_world = -z_d; // Down -> Up

    // Omega raw in body-FLU (da PX4 o da MoCap)
    Eigen::Vector3d omega_raw_flu = Eigen::Vector3d::Zero();
    if (use_mocap_omega_) {
      if (!has_omega_mocap_) {
        return;
      }
      omega_raw_flu = omega_mocap_raw_;
    } else {
      // Velocità angolare: campo angular_velocity in frame body FRD
      // Conversione FRD (Forward, Right, Down) -> FLU (Forward, Left, Up)
      const double wx_flu = static_cast<double>(vehicle_odom_.angular_velocity[0]);
      const double wy_flu = -static_cast<double>(vehicle_odom_.angular_velocity[1]);
      const double wz_flu = -static_cast<double>(vehicle_odom_.angular_velocity[2]);
      omega_raw_flu = Eigen::Vector3d(wx_flu, wy_flu, wz_flu);
    }

    // Low-pass su omega pubblicata
    const rclcpp::Time t_now = this->now();
    double dt_f = 0.0;
    if (omega_filt_init_) {
      dt_f = (t_now - omega_filt_t_prev_).seconds();
      dt_f = std::clamp(dt_f, 1e-4, 0.05);
      omega_filt_ = lowpassIIR(omega_raw_flu, omega_filt_, dt_f, omega_lp_tau_);
    } else {
      omega_filt_ = omega_raw_flu;
      omega_filt_init_ = true;
    }
    omega_filt_t_prev_ = t_now;

    geometry_msgs::msg::Twist twist;
    twist.linear.x  = x_world;
    twist.linear.y  = y_world;
    twist.linear.z  = z_world;
    twist.angular.x = omega_filt_.x();
    twist.angular.y = omega_filt_.y();
    twist.angular.z = omega_filt_.z();

    twist_pub_->publish(twist);
  }

  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr twist_pub_;
  rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr vehicle_odom_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr mocap_pose_sub_;

  // Parametri
  bool use_mocap_omega_{false};
  double omega_lp_tau_{0.05};

  px4_msgs::msg::VehicleOdometry vehicle_odom_;
  bool has_odom_{false};
  bool yaw_offset_initialized_{false};
  double yaw_offset_{0.0};

  // MoCap omega (raw)
  bool mocap_q_init_{false};
  Eigen::Quaterniond mocap_q_prev_{1.0, 0.0, 0.0, 0.0};
  rclcpp::Time mocap_t_prev_;
  Eigen::Vector3d omega_mocap_raw_{0.0, 0.0, 0.0};
  bool has_omega_mocap_{false};

  // Filtro omega (pubblicata)
  bool omega_filt_init_{false};
  Eigen::Vector3d omega_filt_{0.0, 0.0, 0.0};
  rclcpp::Time omega_filt_t_prev_;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<RealDroneVelPub>());
  rclcpp::shutdown();
  return 0;
}
