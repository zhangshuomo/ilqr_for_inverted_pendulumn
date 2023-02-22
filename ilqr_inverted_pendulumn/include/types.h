#ifndef TYPES
#define TYPES
#include<Eigen/Core>

typedef Eigen::Matrix<double,4,1> State;
typedef Eigen::Matrix<double,1,1> Control;
typedef Eigen::Matrix<double,4,4> StateMat;
typedef Eigen::Matrix<double,4,1> ControlMat;
#endif