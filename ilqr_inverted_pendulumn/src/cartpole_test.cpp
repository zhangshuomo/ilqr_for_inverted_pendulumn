#include<ros/ros.h>
#include<fstream>
#include<iostream>
#include<std_msgs/Float64.h>
#include<Eigen/Core>
#include<sensor_msgs/JointState.h>
#include<matplotlibcpp.h>
#include"types.h"

namespace plt=matplotlibcpp;

State state;    // global variable get changed in the callback function
double time_step=0.01;

void get_state(const sensor_msgs::JointState::ConstPtr& joint_state){
    double x=joint_state->position[1];
    double phi=joint_state->position[0];
    double x_dot=joint_state->velocity[1];
    double phi_dot=joint_state->velocity[0];

    state[0]=x;
    state[1]=phi;
    state[2]=x_dot;
    state[3]=phi_dot;
}

void get_controller_params(std::vector<Eigen::MatrixXd> &feedback_gains,
                           std::vector<Eigen::VectorXd> &feedforward_controls){
    std::ifstream matrixDataFile("/home/zhangduo/test_ws/controller_parameter");

    std::string matrixRowString;
    std::string matrixEntry;
    int counter=0;

    while (getline(matrixDataFile, matrixRowString)) // here we read a row by row of matrixDataFile and store every line into the string variable matrixRowString
	{
		std::stringstream matrixRowStringStream(matrixRowString); //convert matrixRowString that is a string to a stream variable.    
        std::vector<double> matrixEntries;
		while (getline(matrixRowStringStream, matrixEntry, ',')) // here we read pieces of the stream matrixRowStringStream until every comma, and store the resulting character into the matrixEntry
		{
			matrixEntries.push_back(stod(matrixEntry));   //here we convert the string to double and fill in the row vector storing all the matrix entries
		}
        Eigen::MatrixXd res=Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(matrixEntries.data(), 1, matrixEntries.size());
        if(counter%2==0)
            feedback_gains.push_back(res);
        else
            feedforward_controls.push_back(res);
        counter++;
	}
}

int main(int argc, char *argv[])
{
    srand(time(0));

    ros::init(argc, argv, "cartpole_controller");
    ros::NodeHandle nh;

    ros::Publisher pub1 = nh.advertise<std_msgs::Float64>("/cartpole/joint1_effort_controller/command",1);
    ros::Publisher pub2 = nh.advertise<std_msgs::Float64>("/cartpole/joint2_effort_controller/command",1);
    ros::Subscriber sub = nh.subscribe<sensor_msgs::JointState>("/cartpole/joint_states", 1, get_state);
    
    ros::Rate r(100);

    std::vector<Eigen::MatrixXd> feedback_gains;
    std::vector<Eigen::VectorXd> feedforward_controls;
    get_controller_params(feedback_gains, feedforward_controls);
    
    std::vector<double> true_pos;
    std::vector<double> true_dpos;
    std::vector<double> true_phi;
    std::vector<double> true_dphi;

    int counter=0;
    while (ros::ok())
    {
        if(counter>=feedback_gains.size()) break;
        
        auto data=feedforward_controls[counter]+feedback_gains[counter]*state;

        std_msgs::Float64 force1, force2;
        force1.data=data[0];
        force2.data=0.0;
        
        pub1.publish(force1);
        pub2.publish(force2);

        ros::spinOnce();
        r.sleep();
        
        true_pos.push_back(state[0]);
        true_phi.push_back(state[1]);
        true_dpos.push_back(state[2]);
        true_dphi.push_back(state[3]);

        counter++;
    }
    std::cout<<"control end"<<std::endl;

    plt::figure();
    plt::title("position");
    plt::named_plot("true pos",true_pos);
    plt::named_plot("true pos_dot",true_dpos);
    plt::grid(true);
    plt::legend();

    plt::figure();
    plt::title("angle");
    plt::named_plot("true phi",true_phi);
    plt::plot(std::vector<double>(true_phi.size(),M_PI),"r--");
    plt::named_plot("true phi_dot", true_dphi);
    plt::grid(true);
    plt::legend();

    plt::show();

    return 0;
}
