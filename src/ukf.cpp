#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
#include <cmath>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = M_PI / 5;
  
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

  R_l_ = MatrixXd(2, 2);
  R_l_ << std_laspx_*std_laspx_, 0,
          0, std_laspy_*std_laspy_;;

  R_r_ = MatrixXd(3, 3);
  R_r_ << std_radr_ * std_radr_, 0, 0,
              0, std_radphi_ * std_radphi_, 0,
              0, 0, std_radrd_ * std_radrd_;


  // Will be set to true after the first measurement.
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
  
  // Initialize Matrices if it is the first measuement
  if (! is_initialized_) { InitializeMeasurement(meas_package); return; }

  double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;

  Prediction(dt);

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  } else {
    UpdateLidar(meas_package);
  }

}

void UKF::InitializeMeasurement(MeasurementPackage meas_package){
  if (meas_package.sensor_type_ == MeasurementPackage::LASER){
      
      x_ << meas_package.raw_measurements_(0), 
            meas_package.raw_measurements_(1), 
            0.0, 
            0.0, 
            0.0;
      
      P_ << std_laspx_*std_laspx_, 0, 0, 0, 0,
            0, std_laspy_*std_laspy_, 0, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 0, 1, 0,
            0, 0, 0, 0, 1;

    } else if(meas_package.sensor_type_ == MeasurementPackage::RADAR){
      
      double ro = meas_package.raw_measurements_(0);
      double phi = meas_package.raw_measurements_(1);
      double ro_dot = meas_package.raw_measurements_(2);
      
      double x = ro * cos(phi);
      double y = ro * sin(phi);
      double vx = ro_dot * cos(phi);
      double vy = ro_dot * sin(phi);
      double v = sqrt(vx * vx + vy * vy);
      
      x_ << x, 
            y, 
            v, 
            ro, 
            ro_dot;
      
      //state covariance matrix
      //***** values can be tuned *****
      P_ << std_radr_*std_radr_, 0, 0, 0, 0,
            0, std_radr_*std_radr_, 0, 0, 0,
            0, 0, std_radrd_*std_radrd_, 0, 0,
            0, 0, 0, std_radphi_, 0,
            0, 0, 0, 0, std_radphi_;
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
  VectorXd x_aug_ = VectorXd(7);    // Augmented State
  MatrixXd P_aug_ = MatrixXd(7, 7); // Augmented State Covariance
  
  MatrixXd Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1); // Sigma points for the new augmented space
  
  // Augmented State Mean 
  x_aug_.head(5) = x_;
  x_aug_(5) = 0;
  x_aug_(6) = 0;
  
  // Augmented Satte Covariance Matrix
  MatrixXd Q = MatrixXd(2,2);
  Q << std_a_*std_a_, 0,
        0, std_yawdd_*std_yawdd_;
  P_aug_.fill(0.0);
  P_aug_.topLeftCorner(5, 5) = P_;
  P_aug_.bottomRightCorner(2, 2) = Q;

  // Square root of Augmented State Cov Matrix
  MatrixXd A_aug = P_aug_.llt().matrixL();
  
  // Augmented Sigma Points
  Xsig_aug_.col(0) = x_aug_;
  for(int i = 0; i < n_aug_; i++) {
    Xsig_aug_.col(i+1) = x_aug_ + std::sqrt(lambda_+n_aug_)*A_aug.col(i);
    Xsig_aug_.col(i+1+n_aug_) = x_aug_ - std::sqrt(lambda_+n_aug_)*A_aug.col(i);
  }
  
  // Perform Prediction on Augmented Sigma Points, v1, v2 vectors for each added part (one changes for yaw = 0 other not). 
  VectorXd v1 = VectorXd(5);
  VectorXd v2 = VectorXd(5);
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
      v1 << (v/yawd)*(sin(yaw+yawd*delta_t) - sin(yaw)),
            (v/yawd)*(-cos(yaw+yawd*delta_t) + cos(yaw)),
            0,
            yawd * delta_t,
            0;
    } else {
      v1 << v*cos(yaw)*delta_t,
            v*sin(yaw)*delta_t,
            0,
            yawd*delta_t,
            0;
    }
    
    v2 << .5*delta_t*delta_t*cos(yaw)*v_aug,
            .5*delta_t*delta_t*sin(yaw)*v_aug,
            delta_t*v_aug,
            .5*delta_t*delta_t*v_yawdd,
            delta_t*v_yawdd;
    
    Xsig_pred_.col(i) << orig + v1 + v2;
  }

  // Compute the predicted state vector and covariance matrix by merging predicted sigma points with their corresponding weights

  VectorXd x_pred = VectorXd(n_x_); x_pred.fill(0.0);     // Predicted state vector
  MatrixXd P_pred = MatrixXd(n_x_, n_x_); P_pred.fill(0.0); // Predicted state covariance matrix

  for(int i = 0; i < 2 * n_aug_ + 1; i++) {
    weights_(i) = (i==0)?lambda_:.5 / (lambda_ + n_aug_);
    x_pred += weights_(i) * Xsig_pred_.col(i);
  }
  
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    
    VectorXd x_diff = Xsig_pred_.col(i) - x_pred;
    while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;
    
    P_pred += weights_(i) * x_diff * x_diff.transpose();

  }
  
  // Update Variables with final prediction result
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

  
  int n_z = 2; // Laser Measures two values px, py
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1); Zsig.fill(0.0); // Sigma Points in Measurement Space
  
  VectorXd z_pred = VectorXd(n_z); z_pred.fill(0.0); // Updated State After Measurement
  MatrixXd S = MatrixXd(n_z,n_z); S.fill(0.0); // Updated Cov. Matrix

  // Transform sigma points into the measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    Zsig.col(i) << Xsig_pred_.col(i)(0),
                   Xsig_pred_.col(i)(1);
    z_pred += weights_(i) * Zsig.col(i);
  }

  // Computing Measurement Cov. Matrix
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    S += weights_(i) * z_diff * z_diff.transpose();
  }
  
  S += R_l_;
  
  VectorXd z = VectorXd(n_z);
  z << meas_package.raw_measurements_(0),
       meas_package.raw_measurements_(1);
  
  MatrixXd Tc = MatrixXd(n_x_, n_z); Tc.fill(0.0);
  
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
}