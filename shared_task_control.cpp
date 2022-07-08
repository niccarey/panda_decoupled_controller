// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#include <array>
#include <atomic>
#include <cmath>
#include <queue>
#include <functional>
#include <iostream>
#include <fstream>
#include <string>
#include <iterator>
#include <mutex>
#include <thread>
#include <ncurses.h>
#include <unistd.h>
#include <chrono>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "LowPassFilter.hpp"
//#include "panda_inertia_lib.hpp"
#include "ThreadsafeQueue.hpp"
#include "CircularBuffer.hpp"
#include "Dynamic_Utilities.hpp"

#include <librealsense2/rs.hpp>

#include <franka/duration.h>
#include <franka/exception.h>
#include <franka/model.h>
#include <franka/rate_limiting.h>
#include <franka/robot.h>
#include <franka/gripper.h>

#include "examples_common.h"

#include <Poco/DateTimeFormatter.h>
#include <Poco/File.h>
#include <Poco/Path.h>

/**
 * @example generate_joint_velocity_motion.cpp
 * An example showing how to generate a joint velocity motion.
 *
 * @warning Before executing this example, make sure there is enough space in front of the robot.
 */

//std::atomic_bool estimator_initiated{false};
//std::atomic_bool run_estimator{true};
//std::atomic_bool run_dynamic_control{false};

// set up file writer
std::ofstream output_stream;
// make that two file writers
std::ofstream force_stream;

namespace {
template <class T, size_t N>
std::ostream& operator<<(std::ostream& ostream, const std::array<T, N>& array) {
  ostream << "[";
  std::copy(array.cbegin(), array.cend() - 1, std::ostream_iterator<T>(ostream, ","));
  std::copy(array.cend() - 1, array.cend(), std::ostream_iterator<T>(ostream));
  ostream << "]";
  return ostream;
  }
}  // anonymous namespace

// Replace all threadsafeQueue instances with boost queues to split classes?



// Shared dynamic data queues
// Might need a new namespace here in order to access these queues from different classes? OR
// pass the queues in as class initialisers?
ThreadsafeQueue<Eigen::VectorXd> WrenchQ;
ThreadsafeQueue<Eigen::Affine3d> TransformQ;
ThreadsafeQueue<Eigen::Matrix<double, 6, 4>> DataMatrixQ;

ThreadsafeQueue<Eigen::VectorXd> subsampleWrench;
ThreadsafeQueue<Eigen::Matrix<double, 6, 4>> subsampleDynMatrix;

ThreadsafeQueue<franka::RobotState> DecoupledControlStates;

int keyhit(void){
    int ch = getch();
    if (ch != ERR) { return 1;}
    else { return 0; }
}

// declare external sensor data structures
// we should pass pointers instead
// but pipeline callback functions seem to be very rigid in structure?
namespace extSensors{
    Eigen::Vector3d raw_accel_data;
    Eigen::Vector3d raw_gyro_data;
    Eigen::Quaterniond estimated_pose;
    Eigen::Vector3d euler_pose;
    Eigen::Vector3d ang_accel;
}


Eigen::Vector3d grav_in_EE(Eigen::Affine3d ee_tf)
{
  Eigen::Vector3d gravity_base(0, 0, 9.81);
  return(ee_tf.linear()*gravity_base);
}

Eigen::Vector3d omega_ee_est(Eigen::Affine3d rel_transform, Eigen::Vector3d omega_prev, double ang_vel){
  Eigen::Vector3d z(0,0,1);

  // return updated omega
  return(rel_transform.linear()*omega_prev + z*ang_vel);
}


// This might be better in the dynamic utilities class? But can be outside for now
void update_link_transforms(franka::Model& model, franka::RobotState state, std::vector<Eigen::Affine3d>& current_link_transforms){
  int iterator = 0;
  for (franka::Frame frame = franka::Frame::kJoint1; frame <= franka::Frame::kJoint7; frame++) {
    // Get current link transform:
    Eigen::Affine3d link_transform(Eigen::Matrix4d::Map(model.pose(frame, state).data()));
    current_link_transforms[iterator] = link_transform;
    iterator++;
  }

}



bool check_imu_is_supported()
{
    bool found_gyro = false;
    bool found_accel = false;
    rs2::context ctx;
    for (auto dev : ctx.query_devices())
    {
        // The same device should support gyro and accel
        found_gyro = false;
        found_accel = false;
        for (auto sensor : dev.query_sensors())
        {
            for (auto profile : sensor.get_stream_profiles())
            {
                if (profile.stream_type() == RS2_STREAM_GYRO)
                    found_gyro = true;

                if (profile.stream_type() == RS2_STREAM_ACCEL)
                    found_accel = true;
            }
        }
        if (found_gyro && found_accel)
            break;
    }
    return found_gyro && found_accel;
}




class rotation_estimator{

  Eigen::Vector3d theta;
  rs2_vector last_w;
  std::mutex theta_mtx;
  float alpha = 0.98f;
  bool firstGyro = true;
  bool firstAccel = true;
  double last_ts_gyro = 0;
  double last_ts_accel = 0;

      // most of this isn't very useful

public:

  Eigen::Vector3d gyro_diff(rs2_vector gyro_data, double ts)
  {
    if (firstAccel) {
      firstAccel = false;
      last_w = gyro_data;
      last_ts_accel = ts;
      return(Eigen::Vector3d(0,0,0));
    }

    double dt_gyro = (ts - last_ts_accel)/1000.0;
    last_ts_gyro = ts;
    Eigen::Vector3d gyro_diff(gyro_data.x - last_w.x, gyro_data.y - last_w.y, gyro_data.z - last_w.z);
    last_w = gyro_data;
    return(gyro_diff/dt_gyro);
  }

  void process_gyro(rs2_vector gyro_data, double ts)
  {
    if (firstGyro) {
      firstGyro = false;
      last_ts_gyro = ts;
      return;
    }

    Eigen::Vector3d gyro_angle;
    gyro_angle[0] = gyro_data.x;
    gyro_angle[1] = gyro_data.y;
    gyro_angle[2] = gyro_data.z;

    double dt_gyro = (ts - last_ts_gyro)/1000.0;
    last_ts_gyro = ts;

    gyro_angle = gyro_angle*static_cast<float>(dt_gyro);

    // apply to the shared theta variable:
    std::lock_guard<std::mutex> lock(theta_mtx);
    Eigen::Vector3d gyro_update(-gyro_angle[2], -gyro_angle[1], gyro_angle[0]);
    theta += gyro_update;

    update_pose();

  }

  void process_accel(rs2_vector accel_data)
  {
    Eigen::Vector3d accel_angle;
    accel_angle[2] = atan2(accel_data.y, accel_data.z);
    accel_angle[1] = atan2(accel_data.x, sqrt(accel_data.y *accel_data.y + accel_data.z*accel_data.z));
    if (firstAccel)
        {
            firstAccel = false;
            theta = accel_angle;
            // Since we can't infer the angle around Y axis using accelerometer data, we'll use PI as a convetion for the initial pose
            theta[1] = M_PI;
        }

    else{
      // apply complementary filter

            theta[0] = theta[0] * alpha + accel_angle[0] * (1 - alpha);
            theta[2] = theta[2] * alpha + accel_angle[2] * (1 - alpha);
    }
    update_pose();
  }

  Eigen::Vector3d get_theta()
  {
    std::lock_guard<std::mutex> lock(theta_mtx);
    return theta;
  }

  void update_pose(){
    // updates externally shared pose reference
    // Convert theta to wrist sensor frame: rotate 90 degrees around local Z
    Eigen::AngleAxisd correct(M_PI_2, Eigen::Vector3d::UnitZ());

    Eigen::Quaterniond q;
    q = Eigen::AngleAxisd(theta[0], Eigen::Vector3d::UnitX())
        * Eigen::AngleAxisd(theta[1], Eigen::Vector3d::UnitY())
        * Eigen::AngleAxisd(theta[2], Eigen::Vector3d::UnitZ());

    extSensors::estimated_pose = correct*q;

    Eigen::Matrix3d m;
    m = (correct*q).normalized().toRotationMatrix();

    extSensors::euler_pose = m.eulerAngles(0,1,2);
  }

};


int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <robot-hostname>" << std::endl;
    return -1;
  }

/*
  // check we  have a realsense with IMU support plugged in
  if (!check_imu_is_supported()){
    std::cerr << "Device supporting IMU not found";
    return EXIT_FAILURE;
  }

  // Declare realsense pipeline
  rs2::pipeline pipe;
  // Create configuration
  rs2::config cfg;
  cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F);
  cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F); // recommended to use joint angles
  // for angular acceleration sensing rather than sensor data. Check these against each other.

  // declare our calibration class
  rotation_estimator angle_calib;

  // start the realsense pipeline


  auto rs_profile = pipe.start(cfg, [&](rs2::frame frame){
      auto motion = frame.as<rs2::motion_frame>();
      if (motion && motion.get_profile().stream_type()==RS2_STREAM_GYRO &&
      motion.get_profile().format() == RS2_FORMAT_MOTION_XYZ32F)
      {
          double ts = motion.get_timestamp();
          rs2_vector gyro_data = motion.get_motion_data();

          // Angle extraction is probably not needed as we can use the robot's internal estimate
          // angle_calib.process_gyro(gyro_data,ts);

          // convert to Eigen for external access
          extSensors::raw_gyro_data[0] = gyro_data.x;
          extSensors::raw_gyro_data[1] = gyro_data.y;
          extSensors::raw_gyro_data[2] = gyro_data.z;

          Eigen::Vector3d ang_accel_est(angle_calib.gyro_diff(gyro_data, ts));
          extSensors::ang_accel[0] = ang_accel_est[0];
          extSensors::ang_accel[1] = ang_accel_est[1];
          extSensors::ang_accel[2] = ang_accel_est[2];
      }

      if (motion && motion.get_profile().stream_type() == RS2_STREAM_ACCEL &&
          motion.get_profile().format() == RS2_FORMAT_MOTION_XYZ32F)
      {
          rs2_vector accel_data = motion.get_motion_data();
          // angle_calib.process_accel(accel_data);

          // convert to Eigen for external access
          extSensors::raw_accel_data[0] = accel_data.x;
          extSensors::raw_accel_data[1] = accel_data.y;
          extSensors::raw_accel_data[2] = accel_data.z;
        }

  });
  */

  // Setting low collision tolerance for initial motion_frame

  // rs_background.detach();
  const double simple_damping = 20.0;
  const double simple_stiffness = 15.0;
  Eigen::MatrixXd stiffness(6, 6), damping(6, 6);

  damping.setZero();
  damping.topLeftCorner(3,3) << 2.0 * sqrt(simple_damping)*Eigen::MatrixXd::Identity(3,3);
  damping.bottomRightCorner(3,3) << 2.0 * sqrt(simple_damping)*Eigen::MatrixXd::Identity(3,3);

  stiffness.setZero();

  stiffness.topLeftCorner(3,3) << simple_stiffness*Eigen::MatrixXd::Identity(3,3);
  stiffness.bottomRightCorner(3,3) << simple_stiffness*Eigen::MatrixXd::Identity(3,3);

  // dropping this too low causes dubious behaviour, possibly instability.
  // Unclear how the near zero-stiffness compliance in guided mode is achieved.
  const double task_stiffness = 10;
  const double task_damping = 2;

  const double null_stiffness = 250.0;
  const double null_damping = 35.0;

  Eigen::VectorXd task_bin_vector(6);
  task_bin_vector.setZero();
  task_bin_vector[0] = 1;
  task_bin_vector[1] = 1;
  task_bin_vector[5] = 1;


  try {
    franka::Robot robot(argv[1]);
    setDefaultBehavior(robot);

    // load model: how do we use this with control?
    franka::Model model = robot.loadModel();

    // First move the robot to a suitable joint configuration
    std::array<double, 7> q_goal = {{0, -M_PI_4, 0, -3 * M_PI_4, 0, M_PI_2, M_PI_4}};

    MotionGenerator motion_generator(0.5, q_goal);
    std::cout << "WARNING: This example will move the robot! "
              << "Please make sure to have the user stop button at hand!" << std::endl
              << "Press Enter to continue..." << std::endl;
    std::cin.ignore();
    robot.control(motion_generator);
    std::cout << "Finished moving to initial joint configuration." << std::endl;

    // add sensor and mount weight to end-effector
    std::array<double,3> realsense_com = {0,0,0};
    std::array<double,9> realsense_inertia = {0.001, 0 , 0 , 0, 0.001, 0, 0, 0, 0.001};
    double realsense_mass = 0.1445;

    robot.setLoad(realsense_mass, realsense_com, realsense_inertia);

    // inertial estimation thread loop

    // set collision behaviour to 'loose' and add a grasp homing
    robot.setCollisionBehavior(
                {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0}}, {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0}},
                {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0}}, {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0}},
                {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0}}, {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0}},
                {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0}}, {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0}});

    std::cout << "Setting collision behaviour to high tolerance." << std::endl;

    franka::RobotState initial_state = robot.readOnce();

    // equilibrium point is the initial position
    Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(initial_state.O_T_EE.data()));
    Eigen::Vector3d position_d(initial_transform.translation());
    Eigen::Quaterniond orientation_d(initial_transform.linear());

    Eigen::Vector3d position_home(position_d);
    Eigen::Quaterniond orientation_home(orientation_d);

    // Use initial transform to calculate a lift phase transition so we don't get weird
    // variation in positions. Also, no longer need to do inertial calcs during lift phase
    // so this could actually be a point to point motion instead of a velocity driven motion?

    // PROBLEM: two control lambdas referencing the same outer loop variable.
    // Seems to cause compile issues. My options are: use a shared pointer captured by value to manage the lambda lifetime
    // Or just use a different goal position for each controller.

    // Set up a list of state transforms we can populate as needed:
    std::vector<Eigen::Affine3d> state_transform_record;
    for (franka::Frame frame = franka::Frame::kJoint1; frame <= franka::Frame::kJoint7; frame++) {
      // Populate state transform vector
      Eigen::Affine3d link_transform(Eigen::Matrix4d::Map(model.pose(frame, initial_state).data()));
      state_transform_record.push_back(link_transform);
    }

    // Set up force control functions:

    // Guided functionality: arrange pick up point
    std::function<franka::Torques(const franka::RobotState&, franka::Duration)>
     guided_controller = [&](const franka::RobotState& robot_state,
                                         franka::Duration /*duration*/) -> franka::Torques
        {
          // get state variables
          std::array<double, 7> coriolis_array = model.coriolis(robot_state);
          std::array<double, 42> jacobian_array =
              model.zeroJacobian(franka::Frame::kEndEffector, robot_state);

          // convert to Eigen
          Eigen::Map<const Eigen::Matrix<double, 7, 1> > coriolis(coriolis_array.data());
          Eigen::Map<const Eigen::Matrix<double, 6, 7> > jacobian(jacobian_array.data());
          Eigen::Map<const Eigen::Matrix<double, 7, 1> > q(robot_state.q.data());
          Eigen::Map<const Eigen::Matrix<double, 7, 1> > dq(robot_state.dq.data());
          Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
          Eigen::Vector3d position(transform.translation());
          Eigen::Quaterniond orientation(transform.linear());


          // compute error to desired equilibrium pose
          Eigen::Matrix<double, 6, 1> error;
          error.head(3) << position - position_d;

          // orientation error
          // "difference" quaternion
          if (orientation_d.coeffs().dot(orientation.coeffs()) < 0.0) {
                orientation.coeffs() << -orientation.coeffs();
          }
          // "difference" quaternion
          Eigen::Quaterniond error_quaternion(orientation.inverse() * orientation_d);
          error.tail(3) << error_quaternion.x(), error_quaternion.y(), error_quaternion.z();
          // Transform to base frame
          error.tail(3) << -transform.linear() * error.tail(3);

         // Calculate applicable torque:

         Eigen::VectorXd tau_task(7), tau_d(7);
         // Spring damper system with damping ratio=1
         tau_task << jacobian.transpose() * (-stiffness * error - damping * (jacobian * dq));
         tau_d << tau_task + coriolis;

         std::array<double, 7> tau_d_array{};
         Eigen::VectorXd::Map(&tau_d_array[0], 7) = tau_d;

         // Update goal positions to ensure robot doesn't fight operator
         position_d = position;
         orientation_d = orientation;


         // check for keyboard press:

         if (keyhit()){
             std::cout << std::endl << "... pickup location set. Joint states: " << "\r" << std::endl;
             std::cout << robot_state.q << std::endl;
             endwin();
             franka::Torques zero_torques{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
             return franka::MotionFinished(zero_torques);
        }
        return tau_d_array;

      };



      // Upon end, trigger gripper close subroutine
      std::this_thread::sleep_for(std::chrono::seconds(1));

      franka::Gripper gripper(argv[1]);
      //double grasping_width = std::stod(argv[2]);
      std::cout << "Starting gripper subprocess" << "\r" << std::endl;


    // do a homing routine
    // later: move homing earlier, change this to just a close grasp method.
    gripper.homing();


    initscr();
    std::cout << "Robot entering guided state. Press any key when at goal. \r " << std::endl;

    cbreak();
    noecho();
    nodelay(stdscr,TRUE);

    robot.control(guided_controller);

    using namespace std::chrono;
    using namespace std::this_thread;
    std::this_thread::sleep_for(std::chrono::seconds(3));

    // Start grasp
    bool grasp_succeeded = gripper.grasp(0.001, 10, 100);

    std::this_thread::sleep_for(std::chrono::seconds(2));
    // Gripper closed, object grasped

    // should already be at pickup point.

    franka::RobotState grasp_state = robot.readOnce();
    Eigen::Affine3d grasp_tf(Eigen::Matrix4d::Map(grasp_state.O_T_EE.data()));
    Eigen::Vector3d position_g(grasp_tf.translation());

    // Now, switch to velocity control:
    std::cout << "Executing motion." << std::endl;

    try {

      double lift_time = 6.0;
      double time_max = 17.0;
      // use initial and current state to establish the omega scaling factor(s)
      Eigen::Vector3d delta_grasp = position_home - position_g;

      //double omega_max = 0.5;
      double time = 0.0;
      int j_idx = 0;

      // Instantiate a dynamic utilities structure and populate or initialise requisite variables
      // Call the inertial estimation loop as a threaded function.


      // Start calculating requisite dynamic components for control
      Dynamic_Utilities panda_dynamics = Dynamic_Utilities();

      // Call dynamic estimation functions
      Eigen::VectorXd inertial_params(4);
      inertial_params.setZero();

      // Start the inertial estimation thread
      panda_dynamics.run_estimator = true;
      std::thread inertial_est(&Dynamic_Utilities::inertial_estimation_iterator, panda_dynamics, std::ref(inertial_params));

      std::cout << "Dynamic structures initialised, starting parameter estimation" << std::endl;

      // velocity control callback:
      auto velocity_control_callback =
          [&time, &time_max, &lift_time, &delta_grasp, &model,  &inertial_params, &panda_dynamics](const franka::RobotState& robot_state, franka::Duration period) -> franka::JointVelocities{

              time += period.toSec();

              Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
              Eigen::Vector3d position(transform.translation());
              double jvels[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

              // populate link state transforms and update coriolis matrix

              // validate with current coriolis estimation:
              std::array<double, 7> coriolis_array = model.coriolis(robot_state);

              Eigen::Map<const Eigen::Matrix<double, 7, 1> > coriolis(coriolis_array.data());
              Eigen::Map<const Eigen::Matrix<double, 7, 1> > dq(robot_state.dq.data());

              if (time < lift_time) {

              // 1. TRACK POSITION: when we've reached a certain height/lateral position, (a) update mass
              // and (b) send zero velocities, then switch to a full joint rotation

              // actually, too much junk data in the param estimate - move this into well conditioned motion only

                    double kx_v = (delta_grasp[0]*M_PI_2/lift_time) * ( std::sin(2.0*time*M_PI_2/lift_time) ); // hits max at peak
                    double kz_v = (delta_grasp[2]*M_PI_2/lift_time) * ( std::cos(time*M_PI_2/lift_time) ); // hits zero at peak

                    // stiffness frame velocity vector:
                    // calculate pseudo inverse for jacobian:
                    std::array<double, 42> jacobian_array =
                        model.zeroJacobian(franka::Frame::kEndEffector, robot_state);
                    Eigen::Map<const Eigen::Matrix<double, 6, 7> > jacobian(jacobian_array.data());
                    Eigen::Matrix<double, 7, 6> J_e = jacobian.transpose()*(jacobian*jacobian.transpose()).inverse();
                    // simple but should be enough unless/until we want to exploit the null space.

                    Eigen::VectorXd task_velocities(6);
                    task_velocities.setZero();
                    task_velocities(0) = kx_v;
                    task_velocities(2) = kz_v;

                    Eigen::VectorXd joint_velocities = J_e*task_velocities;
                    Eigen::VectorXd::Map(&jvels[0], joint_velocities.size()) = joint_velocities;

              }
              // probably need to do something sophisticated here to avoid discontinuities
              else {

                if (time < time_max) {

                // 2. TRACK TIME: after 12s (or after all mass/inertial elements have stabilized)
                // (a) update inertial parameters and (b) move back to (compensated) force control.

                double cycle = std::floor(std::pow(-1.0, ((time-lift_time) - std::fmod((time-lift_time), time_max)) / time_max));
                double omega = cycle * 0.5 / 3.0 * (1.0 - std::cos(2.0 * M_PI / time_max * (time-lift_time)));

                double amplitudes[] = {2.61/8.0, -2.61/8.0, 1.8/8.0, -2.4/8.0, -3.55/8.0, 2.4/8.0, -2.9/8.0};
                double frequencies[] = {3.68, 2.04, 2.98, 1.75, 4.43, 2.749, 1.4};

                for (int j_i = 0; j_i<7; j_i++){ jvels[j_i] = amplitudes[j_i] * sin( 2.0 * M_PI * (time-lift_time)/frequencies[j_i]); }

                // here is the only place we push data to the stack (for now)
                Eigen::VectorXd raw_wrench = Eigen::VectorXd::Map(robot_state.O_F_ext_hat_K.data(), robot_state.O_F_ext_hat_K.size());
                Eigen::Quaterniond orientation(transform.linear());

                      // Push wrench and transform data onto stack
                WrenchQ.push(raw_wrench);
                TransformQ.push(transform);

                Eigen::Matrix<double, 6, 4> single_data_matrix;
                panda_dynamics.data_matrix_fill(robot_state, model, single_data_matrix);
                DataMatrixQ.push(single_data_matrix);


                }

                if (time >=  time_max) {
                    //  Need to do a trajectory speed wind-down before we kill the motion
                    // Take last joint velocities and reduce them to zero? Need to store for the loop, which is annoying
                    // Try using desired velocities instead
                    Eigen::VectorXd current_velocities = Eigen::VectorXd::Map(robot_state.dq_d.data(), robot_state.dq_d.size());
                    current_velocities *= 0.7;

                    Eigen::VectorXd current_abs = current_velocities.array().abs();

                    if (current_abs.maxCoeff() < 0.001) {

                      std::cout << std::endl << "Finished motion, moving to guided state" << std::endl;

                      franka::JointVelocities zero_vel{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
                      return franka::MotionFinished(zero_vel);
                    }

                    else { Eigen::VectorXd::Map(&jvels[0], current_velocities.size()) = current_velocities;}
                }
              }

              franka::JointVelocities velocities = {{jvels[0], jvels[1], jvels[2], jvels[3], jvels[4], jvels[5], jvels[6]}};

              //std::cout<< inertial_params.transpose()<<std::endl;
              return velocities;
            };

            panda_dynamics.estimator_initiated = true; // does atomic bool have a notification function?

            robot.control(velocity_control_callback);

            std::array<double,3> load_CoM = {inertial_params[1],inertial_params[2], inertial_params[3]};
            double load_mass = inertial_params[0];

            std::cout << "Check load estimate" << std::endl;

            std::cout << load_mass << std::endl;

            // something in here jams up if load registers as negative
            if (load_mass > 0){
            robot.setLoad(load_mass, load_CoM, realsense_inertia); // we can update this with real inertia if necessary
            Eigen::Vector3d cast_load_com;
            cast_load_com << inertial_params[1],inertial_params[2], inertial_params[3];
            panda_dynamics.add_load_mass(load_mass, cast_load_com);
          }
            // need to extend the estimator to 10 parameters though

            // Do a short force control call to slowly return to the initial position before moving into guided state:
            // I should put all the basic error derivations into a subfunction, I call them for almost every controller

            std::this_thread::sleep_for(std::chrono::seconds(1));

            std::function<franka::Torques(const franka::RobotState&, franka::Duration)>
                return_to_origin= [&](const franka::RobotState& robot_state, franka::Duration )  -> franka::Torques
                {
                  std::array<double, 7> coriolis_array = model.coriolis(robot_state);
                  std::array<double, 42> jacobian_array =
                      model.zeroJacobian(franka::Frame::kEndEffector, robot_state);

                  // convert to Eigen
                  Eigen::Map<const Eigen::Matrix<double, 7, 1> > coriolis(coriolis_array.data());
                  Eigen::Map<const Eigen::Matrix<double, 6, 7> > jacobian(jacobian_array.data());
                  Eigen::Map<const Eigen::Matrix<double, 7, 1> > q(robot_state.q.data());
                  Eigen::Map<const Eigen::Matrix<double, 7, 1> > dq(robot_state.dq.data());

                  Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
                  Eigen::Vector3d position(transform.translation());
                  Eigen::Quaterniond orientation(transform.linear());
                  // compute error to desired equilibrium pose
                  Eigen::Matrix<double, 6, 1> error;
                  error.head(3) << position - position_home;

                  // orientation error
                  // "difference" quaternion
                  if (orientation_home.coeffs().dot(orientation.coeffs()) < 0.0) {
                        orientation.coeffs() << -orientation.coeffs();
                  }
                  // "difference" quaternion
                  Eigen::Quaterniond error_quaternion(orientation.inverse() * orientation_home);
                  error.tail(3) << error_quaternion.x(), error_quaternion.y(), error_quaternion.z();
                  // Transform to base frame
                  error.tail(3) << -transform.linear() * error.tail(3);

                  Eigen::VectorXd tau_task(7), tau_d(7);
                  Eigen::MatrixXd ctrl_stiffness(80.0*Eigen::MatrixXd::Identity(6,6));
                  Eigen::MatrixXd ctrl_damping(18.0*Eigen::MatrixXd::Identity(6,6));

                  tau_task << jacobian.transpose() * (-ctrl_stiffness * error - ctrl_damping * (jacobian * dq));
                  tau_d << tau_task + coriolis;

                  std::array<double, 7> tau_d_array{};
                  Eigen::VectorXd::Map(&tau_d_array[0], 7) = tau_d;

                  if (error.norm() < 0.05){
                    std::cout << "Ready to command" <<std::endl;
                    franka::Torques zero_torques{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
                    return franka::MotionFinished(zero_torques);
                  }

                  return tau_d_array;
                };
            robot.control(return_to_origin);

            // TODO: update EE data to include load and CoM offset. Does it make sense to include this
            // into the panda inertia namespace??

            // Devolved impedance controller: for now, we use a simple no-gravity impedance controller
            // as a baseline test, and check control torque values.

            std::cout << "Initialising devolved state controller" << std::endl;
            // Create a decoupled controller:
            // Task space is FORCE FOLLOWING, null space is POSITION FOLLOWING
            Eigen::VectorXi decoupled_task_vector(6);
            decoupled_task_vector.setZero();
            decoupled_task_vector(0) = 1;
            decoupled_task_vector(1) = 1;
            decoupled_task_vector(5) = 1;

            // Set up responsive controller functions
            DecoupledController responsive_controller = DecoupledController();
            //responsive_controller.set_task_gains(5.0, 20.0);
            //responsive_controller.set_null_damping(0.4);


            std::cout << "Starting coriolis matrix calculations " << std::endl;
            panda_dynamics.run_dynamic_control = true; // flag for running coriolis calculations, should maybe change this name.
            // Start calculating the dynamic compoenents needed for decoupled control torque
            std::thread coriolis_calc(&Dynamic_Utilities::populate_dynamic_coefficients, panda_dynamics, std::ref(state_transform_record));


            franka::RobotState ff_state = robot.readOnce();
            franka::RobotState current_state = ff_state;

            Eigen::Affine3d ff_transform(Eigen::Matrix4d::Map(ff_state.O_T_EE.data()));
            Eigen::Vector3d position_ff(ff_transform.translation());
            Eigen::Quaterniond orientation_ff(ff_transform.linear());

            Eigen::Map<const Eigen::Matrix<double, 7, 1> > q_raw(ff_state.q.data());
            Eigen::VectorXd ff_q(q_raw);

            Eigen::VectorXd ff_task_torque(7), ff_null_torque(7);
            responsive_controller.cntr_vec = decoupled_task_vector;


            responsive_controller.state_init_flag = false;

            // std::cout << "Starting torque parameter calculation thread " << std::endl;
            // Can probably move these components into the main loop (and then we can update the vector ... think about how to handle this at a class level )
            // for now, assume a static vector

            std::thread control_torque_updater(&DecoupledController::run, responsive_controller, std::ref(model),  std::ref(panda_dynamics.full_coriolis_matrix), std::ref(ff_transform), std::ref(current_state),std::ref(ff_task_torque), std::ref(ff_null_torque));

            // Inside the thread, we need to update the coriolis matrix, queue the state
            // and if we change gains, goal states, etc, update those as well.


            //std::cout << "Setting joint-space null stiffness gains " << std::endl;
            std::function<franka::Torques(const franka::RobotState&, franka::Duration)>
                devolved_controller= [&](const franka::RobotState& robot_state, franka::Duration duration)  -> franka::Torques
                {
                  // get state variables
                  std::array<double, 7> coriolis_array = model.coriolis(robot_state);
                  std::array<double, 42> jacobian_array =
                      model.zeroJacobian(franka::Frame::kEndEffector, robot_state);

                  // get control input
                  Eigen::VectorXd raw_wrench = Eigen::VectorXd::Map(robot_state.O_F_ext_hat_K.data(), robot_state.O_F_ext_hat_K.size());

                  // convert to Eigen
                  Eigen::Map<const Eigen::Matrix<double, 7, 1> > coriolis(coriolis_array.data());
                  Eigen::Map<const Eigen::Matrix<double, 6, 7> > jacobian(jacobian_array.data());
                  Eigen::Map<const Eigen::Matrix<double, 7, 1> > q(robot_state.q.data());
                  Eigen::Map<const Eigen::Matrix<double, 7, 1> > dq(robot_state.dq.data());

                  // Stack state onto queue

                  update_link_transforms(model, robot_state, state_transform_record); // state transform needs updating so ChrCo populating thread is kept up to date
                  panda_dynamics.update_coriolis(robot_state); // this updates the full coriolis matrix based on current robot state, and whatever we pull from ChrCo

                  // Now update the robot state used to calculate the controller variables
                  //current_state = robot_state;
                  DecoupledControlStates.push(robot_state);


                  Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
                  Eigen::Vector3d position(transform.translation());
                  Eigen::Quaterniond orientation(transform.linear());

                  // compute error to desired equilibrium pose
                  // Might need to input error as a 6-vector in order to update it appropriately
                  Eigen::VectorXd error(6);
                  error.setZero();
                  error.head(3) << position - position_ff;

                  // orientation error
                  // "difference" quaternion (seems like a more arduous approach?)
                  if (orientation_ff.coeffs().dot(orientation.coeffs()) < 0.0) {
                        orientation.coeffs() << -orientation.coeffs();
                  }
                  // "difference" quaternion
                  Eigen::Quaterniond error_quaternion(orientation.inverse() * orientation_ff);

                  error.tail(3) << error_quaternion.x(), error_quaternion.y(), error_quaternion.z();
                  // Transform to base frame
                  error.tail(3) << -transform.linear() * error.tail(3);

                  //std::cout << error << std::endl;


                 Eigen::VectorXd tau_task(7), tau_d(7);

                 // Naive approach
                 //tau_task << jacobian.transpose() * (-stiffness * error - damping * (jacobian * dq));
                 //tau_d << tau_task + coriolis;

                 // Check against control torque:
                 // Think this is too slow, we get crashes every time it's included. Try going back to a parallel threading.
                 //responsive_controller.decoupled_control_loop(robot_state, model, panda_dynamics.full_coriolis_matrix, ff_transform, ff_q, ff_task_torque, ff_null_torque, lpf_task_torque, lpf_null_torque);
                 //std::cout << "Task control torque "<< ff_task_torque << std::endl;
                //                  std::cout << "Null control torque "<< ff_null_torque << std::endl;

                 // todo: identify significant terms
                 // identify any sign/scaling errors
                 // Ensure task and null functions are aligned with state error directions and dimensions


                 if (robot_state.time.toSec()>0.01) {
                                   tau_d << ff_task_torque + ff_null_torque + coriolis;
                 } else {tau_d << tau_task + coriolis;}


                 std::array<double, 7> tau_d_array{};
                 Eigen::VectorXd::Map(&tau_d_array[0], 7) = tau_d;
                 // check for keyboard press:

                 if (keyhit()){
                     std::cout << std::endl << "... shutting down control " << "\r" << std::endl;
                     endwin();
                     franka::Torques zero_torques{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
                     responsive_controller.task_flag = false;
                     return franka::MotionFinished(zero_torques);
                }


                return tau_d_array;

              };

            std::cout << "run controller" << std::endl;

            robot.control(devolved_controller);
          }



    catch (const franka::ControlException& e) {
      std::cout << e.what() << std::endl;
      std::cout << "Running error recovery..." << std::endl;
      robot.automaticErrorRecovery();
    }

  } catch (const franka::Exception& e) {
    std::cout << e.what() << std::endl;
    return -1;
  }

  return 0;
}
