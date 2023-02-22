#include<Eigen/Dense>
#include"types.h"

// l(x,u)=(x-x_ref).T*Q*(x-x_ref)+u.T*R*u
// final_l(x)=(x-x_ref).T*Q_final*(x-x_ref)
class LossFunction{
public:
    LossFunction(int state_dim, int control_dim, double whole_time, int time_steps, 
                 State* state_traj, 
                 const Eigen::MatrixXd &Q,
                 const Eigen::MatrixXd &R,
                 const Eigen::MatrixXd &Q_final,
                 const Eigen::MatrixXd &R_final);
    
    // time_step in range [0,time_steps-1]
    double intermidiate_loss(State state, Control control, int time_step);  
    Eigen::MatrixXd hessian_cost_to_state(State current_state, Control control, int time_step);
    Eigen::MatrixXd hessian_cost_to_control(State current_state, Control control, int time_step);
    Eigen::VectorXd partial_dev_cost_to_state(State current_state, Control control, int time_step);
    Eigen::VectorXd partial_dev_cost_to_control(State current_state, Control control, int time_step);

    double final_loss(State final_state);
    Eigen::MatrixXd hessian_final_cost_to_state(State final_state);
    Eigen::VectorXd dev_final_cost_to_state(State final_state);
private:
    int time_steps;
    int state_dim;
    int control_dim;
    double dT;
    State* state_traj;

    Eigen::MatrixXd Q;
    Eigen::MatrixXd R;

    Eigen::MatrixXd Q_final;
    Eigen::MatrixXd R_final;
};


