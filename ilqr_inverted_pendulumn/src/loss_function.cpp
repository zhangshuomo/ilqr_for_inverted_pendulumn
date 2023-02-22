#include"loss_function.h"

LossFunction::LossFunction(int state_dim, int control_dim, double whole_time, int time_steps,
                           State* state_traj, 
                            const Eigen::MatrixXd &Q,
                            const Eigen::MatrixXd &R,
                            const Eigen::MatrixXd &Q_final,
                            const Eigen::MatrixXd &R_final){
    this->state_dim=state_dim;
    this->control_dim=control_dim;
    this->time_steps=time_steps;
    this->dT=whole_time/double(time_steps);

    this->state_traj=state_traj;
    this->Q=Q;
    this->R=R;
    this->Q_final=Q_final;
    this->R_final=R_final;
}

double LossFunction::intermidiate_loss(State state, Control control, int time_step){
    auto state_diff=state-state_traj[time_step];
    auto intermidiate=state_diff.transpose()*Q*dT*state_diff+control.transpose()*R*dT*control;
    return intermidiate(0,0);
}

Eigen::MatrixXd LossFunction::hessian_cost_to_state(State current_state, Control control, int time_step){
    return 2*Q*dT;
}

Eigen::MatrixXd LossFunction::hessian_cost_to_control(State current_state, Control control, int time_step){
    return 2*R*dT;
}

Eigen::VectorXd LossFunction::partial_dev_cost_to_state(State current_state, Control control, int time_step){
    return 2*Q*dT*(current_state-state_traj[time_step]);
}

Eigen::VectorXd LossFunction::partial_dev_cost_to_control(State current_state, Control control, int time_step){
    return 2*R*dT*control;
}

double LossFunction::final_loss(State final_state){
    auto state_diff=final_state-state_traj[time_steps];
    auto final_loss_mat=(state_diff.transpose()*Q_final*dT*state_diff);
    return final_loss_mat(0,0);
}

Eigen::MatrixXd LossFunction::hessian_final_cost_to_state(State final_state){
    return 2*Q_final*dT;
}

Eigen::VectorXd LossFunction::dev_final_cost_to_state(State final_state){
    return 2*Q_final*dT*(final_state-state_traj[time_steps]);
}
