#ifndef CARTPOLE_DYN
#define CARTPOLE_DYN
#include"pinocchio/parsers/urdf.hpp"
#include"pinocchio/algorithm/joint-configuration.hpp"
#include"pinocchio/algorithm/kinematics.hpp"
#include"pinocchio/algorithm/frames.hpp"
#include"pinocchio/algorithm/jacobian.hpp"
#include"pinocchio/algorithm/rnea.hpp"
#include"pinocchio/parsers/sample-models.hpp"
#include"pinocchio/algorithm/aba.hpp"
#include"pinocchio/algorithm/crba.hpp"
#include"pinocchio/algorithm/contact-dynamics.hpp"
#include"pinocchio/algorithm/centroidal.hpp"
#include"pinocchio/algorithm/energy.hpp"
#include"pinocchio/autodiff/casadi.hpp"
#include<Eigen/Core>
#include<casadi/casadi.hpp>
#include"types.h"

namespace cas=casadi;
namespace pin=pinocchio;

Eigen::Matrix<cas::SX,-1, 1> cas_to_eig(const cas::SX& cas)
{
    Eigen::Matrix<cas::SX,-1,1> eig(cas.size1());
    for(int i = 0; i < eig.size(); i++)
    {
        eig(i) = cas(i);
    }
    return eig;
}

cas::SX eig_to_cas(const Eigen::Matrix<cas::SX,-1,1>& eig)
{
    auto sx = cas::SX(cas::Sparsity::dense(eig.size()));
    for(int i = 0; i < eig.size(); i++)
    {
        sx(i) = eig(i);
    }
    return sx;

}

template<typename T>
Eigen::Matrix<T,4,1> system_conti_dyn(const pin::ModelTpl<T>& model, 
                                      pin::DataTpl<T>& data, 
                                      const Eigen::Matrix<T,4,1>& state, 
                                      const Eigen::Matrix<T,1,1>& input){
    auto q=state.head(2);
    auto v=state.tail(2);
    pin::forwardKinematics(model, data, q, v);
    pin::framesForwardKinematics(model, data, q);
    Eigen::Matrix<T,2,1> tau; tau.setZero();
    tau[0]=input[0];
    
    Eigen::Matrix<T,4,1> dstate;
    dstate.head(2)=v;
    dstate.tail(2)=pin::aba(model,data,q,v,tau);
    
    return dstate;
}



class CartpoleDyn{
public:
    CartpoleDyn(double time_interval=0.01);
    State sys_dyn(State current_state, Control control);
    StateMat sensitivityForState(State cur_state, Control cur_control);
    ControlMat sensitivityForControl(State cur_state, Control cur_control);
private:
    double time_interval;
    cas::Function forward_dyn;
    cas::Function A_sens;
    cas::Function B_sens;
};


CartpoleDyn::CartpoleDyn(double time_interval):time_interval(time_interval){ 
    pin::Model model_;
    pin::urdf::buildModel("/home/zhangduo/test_ws/src/ilqr_inverted_pendulumn/urdf/cartpole.urdf", model_);
    pin::ModelTpl<cas::SX> model=model_.cast<cas::SX>();
    pin::DataTpl<cas::SX> data=pin::DataTpl<cas::SX>(model);

    cas::SX cur_state=cas::SX::sym("cur_state", 4);
    cas::SX cur_control=cas::SX::sym("cur_control");

    // continuous dynamics and derivatives for state and control
    auto dstate_=system_conti_dyn<cas::SX>(model, data, cas_to_eig(cur_state),cas_to_eig(cur_control));
    cas::SX dstate=eig_to_cas(dstate_);
    cas::SX Amat_=jacobian(dstate, cur_state);
    cas::SX Bmat_=jacobian(dstate, cur_control);
    cas::Function dyn("dyn",{cur_state, cur_control},{dstate});
    cas::Function devA("devA",{cur_state, cur_control},{Amat_});
    cas::Function devB("devB",{cur_state, cur_control},{Bmat_});
   
    cas::SX state=cas::SX::sym("state", 4);
    cas::SX input=cas::SX::sym("input");
    
    // forward simulation using RK4
    auto k1=dyn(cas::SXVector{state, input})[0];
    auto k2=dyn(cas::SXIList{state+0.5*time_interval*k1,input})[0];
    auto k3=dyn(cas::SXIList{state+0.5*time_interval*k2,input})[0];
    auto k4=dyn(cas::SXIList{state+time_interval*k3,input})[0];
    auto next_state = state+1.0/6*time_interval*(k1+2*k2+2*k3+k4);
    
    // calculate sensitivity using RK4
    auto kG1=devA(cas::SXIList{state,input})[0];
    auto kG2=mtimes(devA(cas::SXIList{state+0.5*time_interval*k1,input})[0],cas::SX::eye(4)+0.5*kG1*time_interval);
    auto kG3=mtimes(devA(cas::SXIList{state+0.5*time_interval*k2,input})[0],cas::SX::eye(4)+0.5*kG2*time_interval);
    auto kG4=mtimes(devA(cas::SXIList{state+time_interval*k3,input})[0],cas::SX::eye(4)+kG3*time_interval);
    auto G=cas::SX::eye(4)+1.0/6*time_interval*(kG1+2*kG2+2*kG3+kG4);

    auto kH1=devB(cas::SXIList{state,input})[0];
    auto kH2=mtimes(devA(cas::SXIList{state+0.5*time_interval*k1,input})[0],0.5*time_interval*kH1)+devB(cas::SXIList{state+0.5*time_interval*k1,input})[0];
    auto kH3=mtimes(devA(cas::SXIList{state+0.5*time_interval*k2,input})[0],0.5*time_interval*kH2)+devB(cas::SXIList{state+0.5*time_interval*k2,input})[0];
    auto kH4=mtimes(devA(cas::SXIList{state+time_interval*k3,input})[0],time_interval*kH3)+devB(cas::SXIList{state+time_interval*k3,input})[0];
    auto H=1.0/6*time_interval*(kH1+2*kH2+2*kH3+kH4);
    
    // define cas::function obj
    forward_dyn=cas::Function("forward_dyn",{state,input},{next_state});
    A_sens=cas::Function("A_sens",{state,input},{G});
    B_sens=cas::Function("B_sens",{state,input},{H});
}


State CartpoleDyn::sys_dyn(State current_state, Control control){ 
    State res;
    cas::DM state_dm(4,1), control_dm(1,1);
    for(int i=0;i<4;++i)
        state_dm(i,0)=current_state[i];
    for(int i=0;i<1;++i)
        control_dm(i,0)=control[i];
    cas::DMVector args={state_dm, control_dm};
    auto res_vec=forward_dyn(args)[0].get_elements();
    
    for(int i=0;i<4;++i)
        res(i,0)=res_vec[i];
    return res;
}

StateMat CartpoleDyn::sensitivityForState(State cur_state, Control cur_control){
    StateMat res;
    cas::DM state_dm(4,1), control_dm(1,1);
    for(int i=0;i<4;++i)
        state_dm(i,0)=cur_state[i];
    for(int i=0;i<1;++i)
        control_dm(i,0)=cur_control[i];
    cas::DMVector args={state_dm, control_dm};
    auto res_vec=A_sens(args)[0].get_elements();
    for(int j=0;j<4;++j)
        for(int i=0;i<4;++i)
            res(i,j)=res_vec[4*j+i];
    return res;
}

ControlMat CartpoleDyn::sensitivityForControl(State cur_state, Control cur_control){
    ControlMat res;
    cas::DM state_dm(4,1), control_dm(1,1);
    for(int i=0;i<4;++i)
        state_dm(i,0)=cur_state[i];
    for(int i=0;i<1;++i)
        control_dm(i,0)=cur_control[i];
    cas::DMVector args={state_dm, control_dm};
    auto res_vec=B_sens(args)[0].get_elements();
    for(int j=0;j<1;++j)
        for(int i=0;i<4;++i)
            res(i,j)=res_vec[4*j+i];
    return res;
}
#endif
