#include<iostream>
#include<Eigen/Core>
#include<vector>
#include"cartpole_dyn.h"
#include"loss_function.h"
#include<matplotlibcpp.h>
#include<fstream>
#include<iomanip>
namespace plt=matplotlibcpp;

const double total_time=10;
const double step_size=0.01;
const int time_steps = total_time / step_size;

LossFunction loss_function_init(){
    int state_dim=4;
    int control_dim=1;

    State* state_traj=new State[time_steps+1];
    for(int i=0;i<time_steps;++i){
        state_traj[i]<<0.0,
                       M_PI,
                       0.0,
                       0.0;
    }
    state_traj[time_steps]<<0.0,
                            M_PI,
                            0.0,
                            0.0;

    Eigen::Matrix<double,4,4> Q, Q_final;
    Q.setZero();
    Q.diagonal()<<0.0,
                  7.0,
                  1.0,
                  0.0;
    Q_final.setZero();
    Q_final.diagonal()<<0.0,
                        10.0,
                        30.0,
                        10.0;

    Eigen::Matrix<double,1,1> R, R_final;
    R.setZero();
    R_final.setZero();
    R.diagonal()<<2e-4;
    R_final.diagonal()<<2e-4;

    return LossFunction(state_dim, control_dim, total_time, time_steps, state_traj, Q,R,Q_final,R_final);
}

void SLQ(const State &initial_state, const std::vector<Control> &initial_control, 
        std::vector<Eigen::MatrixXd> &feedback_res_gains, std::vector<Control> &feedforward_res_controls){
    CartpoleDyn cartpole_dyn;
    LossFunction loss_function=loss_function_init();
    // int time_steps=500;
    const double alpha_decay=0.8;
    const double alpha_ending_condition=1e-4;
    const int max_total_iter=500;
    double last_loss=0;

    // std::vector<double> loss_record;
    // std::vector<double> step_size_record;

    std::vector<Control> control_traj;
    std::vector<State> state_traj;

    std::vector<Eigen::MatrixXd> state_dev_mat, control_dev_mat;
    std::vector<Eigen::VectorXd> c_vec;
    std::vector<Eigen::MatrixXd> R_mat, Q_mat;
    std::vector<Eigen::VectorXd> q_vec, s_vec;

    std::vector<Eigen::MatrixXd> feedback_gains;
    std::vector<Eigen::VectorXd> feedforward_controls;
    // control_traj should be initialized
    control_traj=initial_control;
    assert(initial_control.size()==time_steps);

    // simulate the system to initialize state_traj 
    state_traj.push_back(initial_state);
    State current_state=initial_state;
    for(int i=0;i<time_steps;++i){
        auto next_state=cartpole_dyn.sys_dyn(current_state, control_traj[i]);
        last_loss+=loss_function.intermidiate_loss(current_state, control_traj[i], i);
        state_traj.push_back(next_state);
        current_state=next_state;
    }
    last_loss+=loss_function.final_loss(current_state);
    int total_iter=0;
    while(true){
        // forward iteration
        for(int i=0;i<time_steps;++i){
            auto Ak=cartpole_dyn.sensitivityForState(state_traj[i], control_traj[i]);
            auto Bk=cartpole_dyn.sensitivityForControl(state_traj[i], control_traj[i]);
            auto ck=state_traj[i+1]-Ak*state_traj[i]-Bk*control_traj[i];

            auto hess_to_state=loss_function.hessian_cost_to_state(state_traj[i], control_traj[i], i);
            auto hess_to_control=loss_function.hessian_cost_to_control(state_traj[i], control_traj[i], i);
            auto grad_to_state=loss_function.partial_dev_cost_to_state(state_traj[i], control_traj[i],i);
            auto grad_to_control=loss_function.partial_dev_cost_to_control(state_traj[i], control_traj[i],i);

            auto Qk=hess_to_state / 2;
            auto Rk=hess_to_control / 2;
            auto qk=(grad_to_state-hess_to_state*state_traj[i]) / 2;
            auto sk=(grad_to_control-hess_to_control*control_traj[i]) / 2;

            // record the state trajectory and partial derivative matrices and vectors
            state_dev_mat.push_back(Ak);
            control_dev_mat.push_back(Bk);
            c_vec.push_back(ck);

            Q_mat.push_back(Qk);
            R_mat.push_back(Rk);
            q_vec.push_back(qk);
            s_vec.push_back(sk);
        }
        // backward iteration
        auto hess_to_state=loss_function.hessian_final_cost_to_state(state_traj[time_steps]);
        auto grad_to_state=loss_function.dev_final_cost_to_state(state_traj[time_steps]);
        Eigen::MatrixXd Pk_=hess_to_state / 2;
        Eigen::VectorXd pk_=(grad_to_state-hess_to_state*state_traj[time_steps]) / 2;
        for(int i=time_steps-1;i>=0;--i){
            auto Ak=state_dev_mat[i];
            auto Bk=control_dev_mat[i];
            auto ck=c_vec[i];

            auto Qk=Q_mat[i];
            auto Rk=R_mat[i];
            auto qk=q_vec[i];
            auto sk=s_vec[i];

            auto Kk=(Rk+Bk.transpose()*Pk_*Bk).inverse()*Bk.transpose()*Pk_*Ak;
            auto kk=(Rk+Bk.transpose()*Pk_*Bk).inverse()*(sk+Bk.transpose()*pk_+Bk.transpose()*Pk_*ck);
            feedback_gains.push_back(Kk);
            feedforward_controls.push_back(kk);

            auto Pk=Qk+Ak.transpose()*Pk_*Ak-(Ak.transpose()*Pk_*Bk)*Kk;
            auto pk=qk+Ak.transpose()*pk_+Ak.transpose()*Pk_*ck-(Ak.transpose()*Pk_*Bk)*kk;
            Pk_=Pk;
            pk_=pk;
        }
        std::reverse(feedback_gains.begin(), feedback_gains.end());
        std::reverse(feedforward_controls.begin(), feedforward_controls.end());
        // alpha line search to get the new control for this SLQ iteration
        std::vector<State> new_state_traj;
        std::vector<Control> new_control_traj;
        double total_loss=0;
        double alpha=1;
        int inner_loop_iter=0;
        const int max_inner_loop_iter=50;
        
        while(true){
            current_state=initial_state;
            new_state_traj.push_back(current_state);
            for(int i=0;i<time_steps;++i){
                /* Control current_control=control_traj[i]+(-feedback_gains[i])*(current_state-state_traj[i])+
                                         alpha*(-control_traj[i]+(-feedback_gains[i])*state_traj[i]-feedforward_controls[i]); */
                Control feedforward=(1-alpha)*(control_traj[i]+feedback_gains[i]*state_traj[i])-alpha*feedforward_controls[i];
                Eigen::MatrixXd feedback_gain = -feedback_gains[i];
                Control current_control = feedforward + feedback_gain * current_state;

                State next_state=cartpole_dyn.sys_dyn(current_state, current_control);

                total_loss+=loss_function.intermidiate_loss(current_state, current_control, i);

                feedback_res_gains.push_back(feedback_gain);
                feedforward_res_controls.push_back(feedforward);
                new_control_traj.push_back(current_control);
                new_state_traj.push_back(next_state);

                if(std::isnan(total_loss)){
                    throw(std::runtime_error("nan encountered!"));
                }  
                current_state=next_state; 
            }
            total_loss+=loss_function.final_loss(current_state);

            if(total_loss<last_loss||inner_loop_iter==max_inner_loop_iter-1){
                state_traj=new_state_traj;
                control_traj=new_control_traj;
                last_loss=total_loss;
                if(inner_loop_iter==max_inner_loop_iter-1)
                    std::cout<<"Maximum iteration has reached, break out!"<<std::endl;
                break;
            }
            feedback_res_gains.clear();
            feedforward_res_controls.clear();
            new_state_traj.clear();
            new_control_traj.clear();
            total_loss=0;
            alpha*=alpha_decay;
            inner_loop_iter++;
        }
        
        // clear the memories
        state_dev_mat.clear();
        control_dev_mat.clear();
        c_vec.clear();
        R_mat.clear();
        Q_mat.clear();
        q_vec.clear();
        s_vec.clear();
        feedback_gains.clear();
        feedforward_controls.clear();

        std::cout<<"iteration: "<<total_iter<<"\tloss function: "<<last_loss<<"\tstep size: "<<alpha<<std::endl;
        if(alpha<alpha_ending_condition||total_iter>=max_total_iter)    break;
        
        total_iter++;
    }
    
    // figure plot
    std::vector<double> x_record, phi_record, x_dot_record, phi_dot_record;
    std::vector<double> us;
    for(int i=0;i<time_steps;++i){
        x_record.push_back(state_traj[i][0]);
        phi_record.push_back(state_traj[i][1]);
        x_dot_record.push_back(state_traj[i][2]);
        phi_dot_record.push_back(state_traj[i][3]);

        us.push_back(control_traj[i][0]);
    }

    // plot control and state trajectory
    plt::figure();
    plt::named_plot("pos", x_record);
    plt::named_plot("pos_dot",x_dot_record);
    plt::grid(true);
    plt::legend();

    plt::figure();
    plt::named_plot("phi", phi_record);
    plt::plot(std::vector<double>(phi_record.size(),M_PI),"r--");
    plt::named_plot("phi_dot",phi_dot_record);
    plt::grid(true);
    plt::legend();

    plt::figure();
    plt::named_plot("u",us);
    plt::grid(true);
    plt::legend();

    // plt::figure();
    // plt::title("loss function");
    // plt::named_plot("loss", loss_record);
    // plt::legend();
    // plt::grid(true);
    //
    // plt::figure();
    // plt::title("step size");
    // plt::named_plot("step size",step_size_record,"r--");
    // plt::legend();
    // plt::grid(true);
    plt::show();
}

int main(int argc, char *argv[])
{
    State initial_state;
    initial_state<<0,0,0,0;
    std::vector<Control> initial_control;
    for(int i=0;i<time_steps;++i){
        Control control;
        control<<1.0;
        initial_control.push_back(control);
    }

    std::vector<Control> feedforward_control;
    std::vector<Eigen::MatrixXd> feedback_gains;
    SLQ(initial_state, initial_control, feedback_gains, feedforward_control);

    const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
    
	std::ofstream file("controller_parameter", std::ios::out);
	if (file.is_open())
	{
        for(int i=0;i<time_steps;++i){
            file << feedback_gains[i].format(CSVFormat);
            file << "\n";
            file << feedforward_control[i].format(CSVFormat);
            file << "\n";
    
        }
		file.close();
	}

    return 0;
}
