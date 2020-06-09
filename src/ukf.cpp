#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = false;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2.0;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 1.0;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
  is_initialized_ = false;
  
  time_us_ = 0;
  lambda_ = 0;

  n_x_ = 5;
  n_aug_ = 7;
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  weights_ = VectorXd(2*n_aug_+1);
  
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
  if (! is_initialized_) { InitializeMeasurement(meas_package); return; }
  
  double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;

  Prediction(dt);

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) UpdateRadar(meas_package);
  else UpdateLidar(meas_package);
  
}

void UKF::InitializeMeasurement(MeasurementPackage meas_package){
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      double rho = meas_package.raw_measurements_(0);
      double phi = meas_package.raw_measurements_(1);
      double rhodot = meas_package.raw_measurements_(2);
      double x = rho * cos(phi);
      double y = rho * sin(phi);
      double vx = rhodot * cos(phi);
      double vy = rhodot * sin(phi);
      double v = sqrt(vx * vx + vy * vy);
      x_ << x, 
            y, 
            v, 
            rho, 
            rhodot;
      

      P_ << std_radr_*std_radr_, 0, 0, 0, 0,
            0, std_radr_*std_radr_, 0, 0, 0,
            0, 0, std_radrd_*std_radrd_, 0, 0,
            0, 0, 0, std_radphi_, 0,
            0, 0, 0, 0, std_radphi_;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      x_ << meas_package.raw_measurements_(0), 
            meas_package.raw_measurements_(1), 
            0, 
            0, 
            0.0;
    
      P_ << std_laspx_*std_laspx_, 0, 0, 0, 0,
            0, std_laspy_*std_laspy_, 0, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 0, 1, 0,
            0, 0, 0, 0, 1;
    }
    
    is_initialized_ = true;
    time_us_ = meas_package.timestamp_;
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
  // Computation of original sigma points, each sigma point is a column in Xsig.
  lambda_ = 3 - n_x_;
  MatrixXd Xsig_ = MatrixXd(n_x_, 2 * n_x_ + 1);
  MatrixXd A_ = P_.llt().matrixL();
  
  Xsig_.col(0) = x_;
  for(int i = 0; i < n_x_; i++) {
    Xsig_.col(i+1) = x_ + std::sqrt(lambda_+n_x_)*A_.col(i);
    Xsig_.col(i+1+n_x_) = x_ - std::sqrt(lambda_+n_x_)*A_.col(i);
  }
  
  // Augmentation Process  
  lambda_ = 3 - n_aug_;
  VectorXd x_aug_ = VectorXd(7);    // Augmented Mean Vector
  MatrixXd P_aug_ = MatrixXd(7, 7); // Augmented State Covariance
  
  MatrixXd Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  
  x_aug_.head(5) = x_;
  x_aug_(5) = 0;
  x_aug_(6) = 0;
  
  MatrixXd Q = MatrixXd(2,2);
  Q << std_a_*std_a_, 0,
        0, std_yawdd_*std_yawdd_;
  P_aug_.fill(0.0);
  P_aug_.topLeftCorner(5, 5) = P_;
  P_aug_.bottomRightCorner(2, 2) = Q;
  
  MatrixXd A_aug = P_aug_.llt().matrixL();
  
  Xsig_aug_.col(0) = x_aug_;
  for(int i = 0; i < n_aug_; i++) {
    Xsig_aug_.col(i+1) = x_aug_ + std::sqrt(lambda_+n_aug_)*A_aug.col(i);
    Xsig_aug_.col(i+1+n_aug_) = x_aug_ - std::sqrt(lambda_+n_aug_)*A_aug.col(i);
  }
  
  VectorXd vec1 = VectorXd(5);
  VectorXd vec2 = VectorXd(5);
  for(int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd calc_col = Xsig_aug_.col(i);
    double px = calc_col(0);
    double py = calc_col(1);
    double v = calc_col(2);
    double yaw = calc_col(3);
    double yawd = calc_col(4);
    double v_aug = calc_col(5);
    double v_yawdd = calc_col(6);
    
    VectorXd orig = calc_col.head(5);
    
    if(yawd > .001) {
      vec1 << (v/yawd)*(sin(yaw+yawd*delta_t) - sin(yaw)),
              (v/yawd)*(-cos(yaw+yawd*delta_t) + cos(yaw)),
              0,
              yawd * delta_t,
              0;
    } else {
      vec1 << v*cos(yaw)*delta_t,
              v*sin(yaw)*delta_t,
              0,
              yawd*delta_t,
              0;
    }
    
    vec2 << .5*delta_t*delta_t*cos(yaw)*v_aug,
            .5*delta_t*delta_t*sin(yaw)*v_aug,
            delta_t*v_aug,
            .5*delta_t*delta_t*v_yawdd,
            delta_t*v_yawdd;
    
    Xsig_pred_.col(i) << orig + vec1 + vec2;
  }
  
  VectorXd x_pred = VectorXd(n_x_);
  MatrixXd P_pred = MatrixXd(n_x_, n_x_);
  
  x_pred.fill(0.0);
  P_pred.fill(0.0);
  
  for(int i = 0; i < 2 * n_aug_ + 1; i++) {
    
    if (i == 0) weights_(i) = lambda_ / (lambda_ + n_aug_);
    else weights_(i) = .5 / (lambda_ + n_aug_);    
    x_pred += weights_(i) * Xsig_pred_.col(i);
  }
  
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    
    VectorXd x_diff = Xsig_pred_.col(i) - x_pred;
    while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;
    
    P_pred += weights_(i) * x_diff * x_diff.transpose();
  }
  
  x_ = x_pred;
  P_ = P_pred;
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
  int n_z = 2;
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1); Zsig.fill(0.0);
  VectorXd z_pred = VectorXd(n_z); z_pred.fill(0.0);
  MatrixXd S = MatrixXd(n_z,n_z); S.fill(0.0);
  
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd state_vec = Xsig_pred_.col(i);
    double px = state_vec(0);
    double py = state_vec(1);
    
    Zsig.col(i) << px,
                   py;
    
    z_pred += weights_(i) * Zsig.col(i);
  }
  
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    S += weights_(i) * z_diff * z_diff.transpose();
  }
  
  MatrixXd R = MatrixXd(2,2);
  R << std_laspx_*std_laspx_, 0,
       0, std_laspy_*std_laspy_;
  S += R;
  
  VectorXd z = VectorXd(n_z);
  
  double meas_px = meas_package.raw_measurements_(0);
  double meas_py = meas_package.raw_measurements_(1);
  
  z << meas_px,
       meas_py;
  
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);
  
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    
    while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;
    
    VectorXd z_diff = Zsig.col(i) - z_pred;
    Tc += weights_(i) * x_diff * z_diff.transpose();

  }
  
  VectorXd z_diff = z - z_pred;

  MatrixXd K = Tc * S.inverse();
  x_ += K*z_diff;
  P_ -= K*S*K.transpose();
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
  int n_z = 3;
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1); Zsig.fill(0.0);
  VectorXd z_pred = VectorXd(n_z); z_pred.fill(0.0);
  MatrixXd S = MatrixXd(n_z,n_z); S.fill(0.0);

  double rho = 0;
  double phi = 0;
  double rho_d = 0;
  
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd state_vec = Xsig_pred_.col(i);
    double px = state_vec(0);
    double py = state_vec(1);
    double v = state_vec(2);
    double yaw = state_vec(3);
    double yaw_d = state_vec(4);
    
    rho = sqrt(px*px+py*py);
    phi = atan2(py,px);
    rho_d = (px*cos(yaw)*v+py*sin(yaw)*v) / rho;
    
    Zsig.col(i) << rho,
                   phi,
                   rho_d;
    
    z_pred += weights_(i) * Zsig.col(i);
  }
  
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < - M_PI) z_diff(1) += 2. * M_PI;
    
    S += weights_(i) * z_diff * z_diff.transpose();
  }
  
  MatrixXd R = MatrixXd(3,3);
  R << std_radr_*std_radr_, 0, 0,
       0, std_radphi_*std_radphi_, 0,
       0, 0, std_radrd_*std_radrd_;
  S += R;
  
  VectorXd z = VectorXd(n_z);
  
  double meas_rho = meas_package.raw_measurements_(0);
  double meas_phi = meas_package.raw_measurements_(1);
  double meas_rhod = meas_package.raw_measurements_(2);
  
  z << meas_rho,
       meas_phi,
       meas_rhod;
  
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);
  
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;
    
    VectorXd z_diff = Zsig.col(i) - z_pred;
    while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;
  
    Tc += weights_(i) * x_diff * z_diff.transpose();
  }
  
  VectorXd z_diff = z - z_pred;
  while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
  while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;
  
  MatrixXd K = Tc * S.inverse();
  x_ += K*z_diff;
  P_ -= K*S*K.transpose();
}