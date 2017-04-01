#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using std::pow;
using std::sqrt;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  // check the validity of the following inputs:
  //  * the estimation vector size should not be zero
  //  * the estimation vector size should equal ground truth vector size
  if(estimations.size() != ground_truth.size()
     || estimations.size() == 0){
    std::cout << "Invalid estimation or ground_truth data" << std::endl;
    return rmse;
  }

  //accumulate squared residuals
  for(unsigned int i=0; i < estimations.size(); ++i){
    VectorXd residual = estimations[i] - ground_truth[i];

    //coefficient-wise multiplication
    residual = residual.array() * residual.array();
    rmse += residual;
  }

  //calculate the mean
  rmse = rmse / estimations.size();

  //calculate the squared root
  rmse = rmse.array().sqrt();

  //return the result
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  MatrixXd Hj(3,4);

  //recover state parameters
  double px = x_state(0);
  double py = x_state(1);
  double vx = x_state(2);
  double vy = x_state(3);

  //check division by zero
  if (px == 0 && py == 0) {
    std::cout << "Warning - Division by zero" << std::endl;
    px = py = 0.01;  // use small values to avoid zero devisions
  }

  //compute the Jacobian matrix
  double px2_py2 = pow(px, 2) + pow(py, 2);

  Hj << px / sqrt(px2_py2), py / sqrt(px2_py2), 0, 0,
        -py / px2_py2, px / px2_py2, 0, 0,
        py * (vx * py - vy * px) / pow(px2_py2, 1.5),
        px * (vy * px - vx * py) / pow(px2_py2, 1.5),
        px / sqrt(px2_py2), py / sqrt(px2_py2);

  return Hj;
}
