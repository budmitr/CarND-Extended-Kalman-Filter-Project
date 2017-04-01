#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0,      0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0,      0,
              0,    0.0009, 0,
              0,    0,      0.09;

  noise_ax = 9;
  noise_ay = 9;

}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    // initialize only, no need to predict or update
    Initialize(measurement_pack);
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/
  Predict(measurement_pack);

  /*****************************************************************************
   *  Update
   ****************************************************************************/
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(measurement_pack);
  } else {
    UpdateLaser(measurement_pack);
  }
}


void FusionEKF::Initialize(const MeasurementPackage &measurement_pack) {
  cout << "Initializing KF..." << endl;

  // Initialize state
  ekf_.x_ = VectorXd(4);
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    cout << "Initializing position by Radar..." << endl;

    double range = measurement_pack.raw_measurements_[0];
    double bearing = measurement_pack.raw_measurements_[1];

    ekf_.x_ << range * cos(bearing),  // px
               range * sin(bearing),  // py
               0,                     // vx
               0;                     // vy
  }
  else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
    cout << "Initializing position by Laser..." << endl;

    ekf_.x_ << measurement_pack.raw_measurements_[0],  // px
               measurement_pack.raw_measurements_[1],  // py
               0,                                      // vx
               0;                                      // vy
  }

  // Initialize state covariance matrix (with big variance for vx/vy)
  ekf_.P_ = MatrixXd(4, 4);
  ekf_.P_ << 1, 0, 0,   0,
             0, 1, 0,   0,
             0, 0, 999, 0,
             0, 0, 0,   999;

  // Initialize state transition matrix
  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_ << 1, 0, 0, 0,  // add deltaT on prediction
             0, 1, 0, 0,  // add deltaT on prediction
             0, 0, 1, 0,
             0, 0, 0, 1;

  // Initialize timestamp
  previous_timestamp_ = measurement_pack.timestamp_;

  // Initialize measurement matrix for laser
  H_laser_ = MatrixXd(2, 4);
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;

  // Initialize measurement Jacobian for radar
  Hj_ = MatrixXd(3, 4);

  // Initialize process covariance matrix
  ekf_.Q_ = MatrixXd(4, 4);

  is_initialized_ = true;
}


void FusionEKF::Predict(const MeasurementPackage &measurement_pack) {
  double dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  double dt_2 = dt   * dt;
  double dt_3 = dt_2 * dt;
  double dt_4 = dt_3 * dt;
  previous_timestamp_ = measurement_pack.timestamp_;

  // Update state transition matrix -- use deltaT
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;

  //set the process covariance matrix Q
  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ <<  dt_4/4*noise_ax, 0,               dt_3/2*noise_ax, 0,
              0,               dt_4/4*noise_ay, 0,               dt_3/2*noise_ay,
              dt_3/2*noise_ax, 0,               dt_2*noise_ax,   0,
              0,               dt_3/2*noise_ay, 0,               dt_2*noise_ay;

  ekf_.Predict();
}


void FusionEKF::UpdateLaser(const MeasurementPackage &measurement_pack) {
  ekf_.H_ = H_laser_;
  ekf_.R_ = R_laser_;
  ekf_.Update(measurement_pack.raw_measurements_);
}


void FusionEKF::UpdateRadar(const MeasurementPackage &measurement_pack) {
  Hj_ = tools.CalculateJacobian(ekf_.x_);
  ekf_.H_ = Hj_;
  ekf_.R_ = R_radar_;
  ekf_.UpdateEKF(measurement_pack.raw_measurements_);
}
