
#ifndef _Dynamic_Utilities_hpp_
#define _Dynamic_Utilities_hpp_

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

#include <franka/duration.h>
#include <franka/exception.h>
#include <franka/model.h>
#include <franka/rate_limiting.h>
#include <franka/robot.h>
#include <franka/gripper.h>


class Dynamic_Utilities{
  public:

    bool run_dynamic_control;
    bool run_estimator;
    bool estimator_initiated; // Do we need to load non atomic bools? Actually, should these be atomic bools anyway?

    Eigen::TensorFixedSize<double, Eigen::Sizes<7,7,7>> ChrCo;
    Eigen::Matrix<double, 7,7> full_coriolis_matrix;
    Eigen::Matrix<double, 6,7> jac_dot_;

    Eigen::TensorFixedSize<double, Eigen::Sizes<3,7,7,7>> MArray;
    Eigen::TensorFixedSize<double, Eigen::Sizes<3,7,7,7>> FArray;

    Eigen::Matrix<double, 7,7> MacSum1;
    Eigen::Matrix<double, 7,7> MacSum2;
    Eigen::Matrix<double, 7,7> MacSum3;
    Eigen::Matrix<double, 7,7> FacSum1;
    Eigen::Matrix<double, 7,7> FacSum2;
    Eigen::Matrix<double, 7,7> FacSum3;

    // panda inertial elements:
    std::vector<LowPassFilter> lpf_wrench;
    float inertial_cutoff;
    float inertial_dt;

    std::vector<double> link_masses {3.06, 2.34, 2.36, 2.38, 2.43, 3.5, 1.47, 0.45};

        Eigen::Matrix3d link1 = (Eigen::Matrix3d() << 7.0337e-01, -1.3900e-04, 6.7720e-03, -1.3900e-04, 7.0661e-01, 1.9169e-02, 6.7720e-03, 1.9169e-02, 9.1170e-03).finished();
        Eigen::Matrix3d link2 = (Eigen::Matrix3d() << 7.9620e-03, -3.9250e-03, 1.0254e-02, -3.9250e-03, 2.8110e-02, 7.0400e-04, 1.0254e-02, 7.0400e-04, 2.5995e-02).finished();
        Eigen::Matrix3d link3 = (Eigen::Matrix3d() << 3.7242e-02, -4.7610e-03, -1.1396e-02, -4.7610e-03, 3.6155e-02, -1.2805e-02, -1.1396e-02, -1.2805e-02, 1.0830e-02).finished();
        Eigen::Matrix3d link4 = (Eigen::Matrix3d() << 2.5853e-02, 7.7960e-03, -1.3320e-03, 7.7960e-03, 1.9552e-02, 8.6410e-03,-1.3320e-03, 8.6410e-03, 2.8323e-02).finished();
        Eigen::Matrix3d link5 = (Eigen::Matrix3d() << 3.5549e-02, -2.1170e-03, -4.0370e-03, -2.1170e-03, 2.9474e-02, 2.2900e-04,-4.0370e-03, 2.2900e-04, 8.6270e-03).finished();
        Eigen::Matrix3d link6 = (Eigen::Matrix3d() << 1.9640e-03, 1.0900e-04, -1.1580e-03, 1.0900e-04, 4.3540e-03, 3.4100e-04, -1.1580e-03, 3.4100e-04, 5.4330e-03).finished();
        Eigen::Matrix3d link7 = (Eigen::Matrix3d() << 1.2516e-02, -4.2800e-04, -1.1960e-03, -4.2800e-04, 1.0027e-02, -7.4100e-04, -1.1960e-03, -7.4100e-04, 4.8150e-03).finished();

    std::vector<Eigen::Matrix3d> link_inertias{link1, link2, link3, link4, link5, link6, link7};

    Eigen::Vector3d link1_com = (Eigen::Vector3d() << -5.689946E-06, -0.03122697, 0.1225765).finished();
    Eigen::Vector3d link2_com = (Eigen::Vector3d() << -7.598723E-06, -0.07036345, 0.0311796).finished();
    Eigen::Vector3d link3_com = (Eigen::Vector3d() << -0.04440198, -0.03811803, -0.0249679).finished();
    Eigen::Vector3d link4_com = (Eigen::Vector3d() << 0.03853792, 0.0247187, -0.03953181).finished();
    Eigen::Vector3d link5_com = (Eigen::Vector3d() << 7.227068E-05, -0.1099062, -0.03844703).finished();
    Eigen::Vector3d link6_com = (Eigen::Vector3d() << -0.05101816, 0.006172824, -0.006939902).finished();
    Eigen::Vector3d link7_com = (Eigen::Vector3d() << -0.01561021, 0.07659241, -0.01413517).finished();

    std::vector<Eigen::Vector3d> link_coms{link1_com, link2_com, link3_com, link4_com, link5_com, link6_com, link7_com};

    // TODO: add hand mass (and additional CoM updates for link 7) TO HERE?
    double gripper_mass = 0.73;
    Eigen::Vector3d gripper_com = (Eigen::Vector3d() << -0.01, 0, 0.03).finished();
    Eigen::Matrix3d gripper_inertia = (Eigen::Matrix3d() << 0.001, 0, 0, 0, 0.0025, 0, 0, 0, 0.0017).finished();

    double load_mass_estimate = 0.0;
    Eigen::Vector3d load_com_est; // can I update variables declared in this name space as long as they aren't static?

  Dynamic_Utilities(){
    // Initialise components to zero
    run_dynamic_control = false;
    estimator_initiated = false;
    run_estimator = false;


    // should this take any arguments?
    ChrCo.setZero();
    full_coriolis_matrix.setZero();
    jac_dot_.setZero();

    MacSum1.setZero();
    MacSum2.setZero();
    MacSum3.setZero();

    FacSum1.setZero();
    FacSum2.setZero();
    FacSum3.setZero();


    // Need to run a function to update the last link mass to include gripper
    // We could probably abstract this to an "update mass" function which takes eg. the link index and arbitrary variables.
    // this would make it easy to add new mass estimates (also we need to be able to 'reset' load in the case of a variable load)
    // TODO LIST

    this->add_gripper_mass();

    // Need to instantiate an LP filter for every dimension we want to filter:
    inertial_cutoff = 1;
    inertial_dt = 0.001;
    for (int i = 0; i < 7 ; i++ ){
      lpf_wrench.push_back(LowPassFilter(inertial_cutoff, inertial_dt));
    }

  }

  void add_load_mass(double new_load_mass, Eigen::Vector3d new_com){
    load_mass_estimate = new_load_mass;
    load_com_est = new_com;

    int final_link = 6;

    Eigen::Vector3d offsetVector = load_com_est - link_coms[final_link];
    Eigen::Matrix3d offsetOP = offsetVector * offsetVector.transpose();

    Eigen::Matrix3d pAxis = (load_mass_estimate * offsetVector.dot(offsetVector)) * Eigen::MatrixXd::Identity(3,3) - offsetOP;
    // now we add hand inertia (incorporating parallel axis) to the fixed link inertia
    link_inertias[final_link] += pAxis;
    // And add cumulative inertia from load, when available (assuming a fixed grip)

    // Also need to adjust the centre of mass
    link_coms[final_link] = (1.0/(link_masses[final_link] + load_mass_estimate))*((link_masses[final_link])*link_coms[final_link] + (load_mass_estimate) * load_com_est);
    link_masses[final_link] += load_mass_estimate;

  }

  void add_gripper_mass(){
    int final_link = 6;

    // Add cumulative inertia from manipulator (ignoring rotation around the axis):
    Eigen::Vector3d offsetVector = gripper_com - link_coms[final_link];
    Eigen::Matrix3d offsetOP = offsetVector * offsetVector.transpose();

    Eigen::Matrix3d pAxis = (gripper_mass * offsetVector.dot(offsetVector)) * Eigen::MatrixXd::Identity(3,3) - offsetOP;
    // now we add hand inertia (incorporating parallel axis) to the fixed link inertia
    link_inertias[final_link] += pAxis;
    // And add cumulative inertia from load, when available (assuming a fixed grip)

    // Also need to adjust the centre of mass
    link_coms[final_link] = (1.0/(link_masses[final_link] + gripper_mass))*((link_masses[final_link])*link_coms[final_link] + (gripper_mass) * gripper_com);
    link_masses[final_link] += gripper_mass;

  }

   void populate_dynamic_coefficients(std::vector<Eigen::Affine3d>& current_link_transforms){
    std::vector<Eigen::Matrix3d> half_Li;

    franka::Frame iter_limit;
    iter_limit = franka::Frame::kJoint7; // stop at joint 7: assume everything beyond this is fixed

    std::vector<franka::Frame> state_pose_vector;
    for (franka::Frame frame = franka::Frame::kJoint1; frame <= iter_limit; frame++) {
        state_pose_vector.push_back(frame);
        half_Li.push_back(Eigen::MatrixXd::Identity(3,3));
      }

    // move this into an inertial calculation subfunction:
    Eigen::Matrix3d fixed_inertia;
    double link_mass;
    Eigen::Vector3d local_com;

    int final_link = 6;

    // Loop until we've got a good enough estimation.
    while (run_dynamic_control) {

      MArray.setZero();
      FArray.setZero();

      MacSum1.setZero();
      MacSum2.setZero();
      MacSum3.setZero();
      FacSum1.setZero();
      FacSum2.setZero();
      FacSum3.setZero();

      for (int i = 0; i < 7; i++){
        // local link inertia
        fixed_inertia = link_inertias[i];

        // extract local transform
        Eigen::Affine3d link_transform = current_link_transforms[i];
        Eigen::Quaterniond link_rotation(link_transform.linear());

        // fill dynamic inertia matrix
        half_Li[i] = 0.5 * (link_rotation * (fixed_inertia.trace()*Eigen::MatrixXd::Identity(3,3))) - 2*fixed_inertia*link_transform.rotation().transpose();
        }

      // nested loops over links and joints (move moment array interator into the main loop)

      for (int i = 0; i < 7; i++){

        for (int j = 1; j < i+1; j++ ){
          // we are operating on link i, according to all the links and joints below
          Eigen::Affine3d link_transform = current_link_transforms[j];
          Eigen::Matrix3d link_rr = link_transform.rotation();
          Eigen::Vector3d Tj_cart = link_transform.translation();

          double t1 = half_Li[i](0,0) * link_rr(0,2) + half_Li[i](0,1) * link_rr(1,2) + half_Li[i](0,2) * link_rr(2,2);
          double t2 = half_Li[i](1,0) * link_rr(0,2) + half_Li[i](1,1) * link_rr(1,2) + half_Li[i](1,2) * link_rr(2,2);
          double t3 = half_Li[i](2,0) * link_rr(0,2) + half_Li[i](2,1) * link_rr(1,2) + half_Li[i](2,2) * link_rr(2,2);

          for (int m = j; m < i+1; m++){

            // Get frame transform:
            Eigen::Affine3d m_tf = current_link_transforms[m];
            Eigen::Matrix3d m_rr = m_tf.rotation();

            // moment array calcs
            MArray(0, m, j, i) = t2 * m_rr(2,2) - t3 * m_rr(1, 2);
            MArray(1, m, j, i) = -t1 * m_rr(2, 2) + t3 * m_rr(0, 2);
            MArray(2, m, j, i) = t1 * m_rr(1, 2) - t2 * m_rr(0, 2);
          }
        }
      }


      Eigen::Vector3d pci;
      Eigen::Vector3d pcim;
      Eigen::Vector3d tmp;

      for (int i= 0; i < 7; i++){
        Eigen::Affine3d link_transform = current_link_transforms[i];
        link_mass = link_masses[i];
        local_com = link_coms[i];

        pci(0) = link_transform(0,3) + link_transform(0,0)*local_com(0) + link_transform(0,1)*local_com(1) + link_transform(0,2)*local_com(2);
        pci(1) = link_transform(1,3) + link_transform(1,0)*local_com(0) + link_transform(1,1)*local_com(1) + link_transform(1,2)*local_com(2);
        pci(2) = link_transform(2,3) + link_transform(2,0)*local_com(0) + link_transform(2,1)*local_com(1) + link_transform(2,2)*local_com(2);

        for (int j = 0; j< i+1; j++){
          Eigen::Affine3d j_tf = current_link_transforms[j];
          tmp(0) = link_mass * j_tf(0,2);
          tmp(1) = link_mass * j_tf(1,2);
          tmp(2) = link_mass * j_tf(2,2);

          for (int m = j; m < i+1; m++){
            Eigen::Affine3d m_tf = current_link_transforms[m];
            pcim(0) = pci(0) - m_tf(0,3);
            pcim(1) = pci(1) - m_tf(1,3);
            pcim(2) = pci(2) - m_tf(2,3);

            double t1 = m_tf(1,2) * pcim(2) - m_tf(2,2)*pcim(1);
            double t2 = -m_tf(0,2) * pcim(2) + m_tf(2,2)*pcim(0);
            double t3 = m_tf(0,2) * pcim(1) - m_tf(1,2)*pcim(0);

            FArray(0,m,j,i) = tmp(1) * t3 - tmp(2) * t2;
            FArray(1,m,j,i) = -tmp(0) * t3 + tmp(2) * t1;
            FArray(2,m,j,i) = tmp(0) *t2 - tmp(1) * t1;

          }
        }
      }



      // Cycle in reverse through each link

      Eigen::Vector3d L(0,0,0);
      Eigen::Vector3d sup_d(0,0,0);

      for (int i=6; i > -1; i--){
        Eigen::Affine3d i_tf = current_link_transforms[i];
        Eigen::Matrix3d i_rr = i_tf.rotation();
        Eigen::Vector3d i_cart = i_tf.translation();
        Eigen::Vector3d i_com = link_coms[i];

        Eigen::Vector3d pc_ii = i_rr * i_com;

        if (i<6){
            Eigen::Affine3d sup_tf = current_link_transforms[i+1];
            sup_d = sup_tf.translation() - i_tf.translation();
            }

        for (int j = 0; j < 7; j++){

            for (int m = j; m < 7; m++){

                MacSum1(m, j) += MArray(0, m, j, i) + sup_d(1) * FacSum3(m, j) - sup_d(2) * FacSum2(m, j) + pc_ii(1) * FArray(2, m, j, i) - pc_ii(2) * FArray(1, m, j, i);
                MacSum2(m, j) += MArray(1, m, j, i) - sup_d(0) * FacSum3(m, j) + sup_d(2) * FacSum1(m, j) - pc_ii(0) * FArray(2, m, j, i) + pc_ii(2) * FArray(0, m, j, i);
                MacSum3(m, j) += MArray(2, m, j, i) + sup_d(0) * FacSum2(m, j) - sup_d(1) * FacSum1(m, j) + pc_ii(0) * FArray(1, m, j, i) - pc_ii(1) + FArray(0, m, j, i);

                FacSum1(m,j) += FArray(0, m, j, i);
                FacSum2(m,j) += FArray(1, m, j, i);
                FacSum3(m,j) += FArray(2, m, j, i);

                ChrCo(m, j, i) = i_rr(0, 2) * MacSum1(m, j) + i_rr(1, 2) * MacSum2(m, j) + i_rr(2, 2) * MacSum3(m, j); // should all be scalars
                ChrCo(j, m, i) = ChrCo(m, j, i);
          }
        }
      }
    }

  }


  void update_coriolis(franka::RobotState state){

    // Extract joint velocities
    Eigen::Map<const Eigen::Matrix<double, 7, 1> > dq(state.dq.data());

    // Update coefficients according to current robot state
    // Within the control loop, we call a coriolis matrix updater which accesses the current
    // chr-coefficient matrix and quickly updates the full coriolis function, which can then be used to divide state
    // priority control, etc.

    // Depending on calculation load, we may subsume this last into a control updater which also runs in parallel
    // and just updates task/null torque commands when data is available.

    // Calculate new coriolis matrix:
    int nJoints = 7; // match panda joint/frame count
    for (int i = 0; i < nJoints; i++)
      {
        for (int j = 0; j < nJoints; j++)
          {
            Eigen::VectorXd ChrCoSlice(nJoints); // create a vectorized slice for each link frame

            // Cycle through link frames and grab the relevant coefficient
            for (int k = 0; k < nJoints; k++) {ChrCoSlice(k) = ChrCo(i, j, k); }
            full_coriolis_matrix(i,j) = ChrCoSlice.dot(dq);
          }
      }
    }


  // Need to neaten this up and rewrite the bits in the control function that use this
  void inertial_estimation_iterator(Eigen::VectorXd& PV_out){

    // put some of these variables in the initialisation function

      // Smoothed data sample
    // instead of a vector, use a queue for dynamic matrix sample and wrench sample
    std::vector<Eigen::VectorXd> smoothed_sample;

    // Initialise components:

    bool filter_init = false;
    int sample_cnt = 0;

    while ( estimator_initiated ){ std::this_thread::sleep_for(std::chrono::milliseconds(5));  }


  // sleep for a moment to ensure the robot is moving
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  // start solver thread - need a function instance, I think?
  // does this need to be threaded?
  //
  std::thread svd_solve_thread(&Dynamic_Utilities::svd_solver, this, std::ref(PV_out));

  // large iterative version
  while (run_estimator){

    // Initialise a cutoff-frequency based low pass filter before we run the estimator.
    try {

      // pop dynamic matrix (discard if we don't use)
      Eigen::Matrix<double, 6, 4> single_data_matrix = DataMatrixQ.pop().value();

      // pop wrench and filter it
      Eigen::VectorXd wrench_input = WrenchQ.pop().value();
      Eigen::Affine3d transform = TransformQ.pop().value();
      Eigen::VectorXd wrench_tf(6), wrench_tf_sm(6);

      //Eigen::VectorXd print_data(12);

      wrench_tf.head(3) = transform.linear().transpose()*wrench_input.head(3);
      wrench_tf.tail(3) = transform.linear().transpose()*wrench_input.tail(3);

      for (int w_i = 0; w_i < 6; w_i++) {
        wrench_tf_sm(w_i) = lpf_wrench[w_i].update(wrench_tf(w_i));
      }

      //print_data.head(6) = wrench_tf;
      //print_data.tail(6) = wrench_tf_sm;
      //print_to_file(print_data);

      if (sample_cnt > 2) {
        // push every nth sample onto the stack, deal with it in a different thread
        subsampleWrench.push(wrench_tf_sm);
        subsampleDynMatrix.push(single_data_matrix);

        sample_cnt = 0;
      }

      sample_cnt++;
    }

    catch (std::bad_optional_access const& exception){
        // probably nothing to pop
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }

   }
 }


const Eigen::Matrix<double, 6, 1> getPartialDerivative(Eigen::Matrix<double, 6,7> *bs_jac_, const unsigned int& joint_idx, const unsigned int& column_idx)
{
  int j = joint_idx;
  int i = column_idx;
  Eigen::VectorXd jac_j_ = bs_jac_->col(j);
  Eigen::VectorXd jac_i_ = bs_jac_->col(i);
  Eigen::VectorXd t_dj_dq(6);
  t_dj_dq.setZero();

  // this is dumb, there should be a twist class that handles 6-vec cross products more elegantly
  Eigen::Vector3d jac_j_vel = jac_j_.head(3);
  Eigen::Vector3d jac_j_rot = jac_j_.tail(3);

  Eigen::Vector3d jac_i_vel = jac_i_.head(3);
  Eigen::Vector3d jac_i_rot = jac_i_.tail(3);

  if (j<i){
    t_dj_dq.head(3) = jac_j_rot.cross(jac_i_vel);
    t_dj_dq.tail(3) = jac_j_rot.cross(jac_i_rot);
  } else if (j>i){
    t_dj_dq.head(3) = -jac_j_vel.cross(jac_i_rot);
    t_dj_dq.tail(3).setZero();
  } else if (j==i){
    t_dj_dq.head(3) = jac_i_rot.cross(jac_i_vel);
    t_dj_dq.tail(3).setZero();
  }
  return t_dj_dq;
}

void jntToJacDot(franka::Model& model, franka::RobotState state, Eigen::VectorXd dq_, franka::Frame tgt_fr, Eigen::Matrix<double, 6,7> *jac_dot_){
  // Should initialise jac_dot_ to zero in case it was passed in uninitialised
  jac_dot_->setZero(); // but note this might be a bit slow

  // Don't think there is a differentiation between joint link and fixed link in the libfranka iterator.
  // So there is no pre-existing mapping between Model and State - you need to know where the joints stop and the flange/end-effector begin
  // hence, we can't do an in-loop check for joint iteration. So this is not generalisable to (random robot).

  // We need to check where the target frame is and if it's beyond the joint limit, we treat the remaining frame TFs differently
  // Since the transform from joint 7 to the end effector is just a translation, do we need to do anything?

  franka::Frame iter_limit;
  if (tgt_fr > franka::Frame::kJoint7){
    iter_limit = franka::Frame::kJoint7;  }
  else { iter_limit = tgt_fr;}

  // Compute jacobian to target frame:
  std::array<double, 42> jac_array = model.zeroJacobian(tgt_fr, state);
  Eigen::Matrix<double, 6, 7> jac_(jac_array.data());


  // Compute jacobian derivative columns:
  Eigen::VectorXd jac_dot_k_(6);
  jac_dot_k_.setZero();

// to do: make the frame input variable below kJoint7, otherwise capped
  int k = 0;
  for (franka::Frame frame = franka::Frame::kJoint1; frame <= iter_limit; frame++){
    for(unsigned int j=0; j < dq_.rows(); j++){
       jac_dot_k_ += this->getPartialDerivative(&jac_, j, k)*dq_[j];
    }
    // set column k of jac_dot_ to jac_dot_k_
    jac_dot_->col(k) = jac_dot_k_;
    k++;
    // reset column twist
    jac_dot_k_.setZero();
  }
}


void dynamic_variable_updater(franka::Model& model, franka::RobotState state, Eigen::VectorXd *twist_out, Eigen::VectorXd *accel_out) {

  twist_out->setZero();
  accel_out->setZero();
  franka::Frame goal_frame = franka::Frame::kEndEffector;

  std::array<double, 42> jacobian_array = model.zeroJacobian(goal_frame, state);
  Eigen::Map<const Eigen::Matrix<double, 6, 7> > jacobian(jacobian_array.data());
  // Eigen::Map<const Eigen::Matrix<double, 7, 1> > q(robot_state.q.data());
  Eigen::Map<const Eigen::Matrix<double, 7, 1> > dq(state.dq.data());
  Eigen::Map<const Eigen::Matrix<double, 7, 1> > ddq(state.ddq_d.data());

  Eigen::Affine3d transform(Eigen::Matrix4d::Map(state.O_T_EE.data()));

   Eigen::VectorXd base_twist = jacobian*dq; // velocity (linear and angular) in base frame

  // perform frame transform
  twist_out->head(3) = transform.linear().transpose()*Eigen::Vector3d(base_twist.head(3));
  twist_out->tail(3) = transform.linear().transpose()*Eigen::Vector3d(base_twist.tail(3));

  Eigen::Matrix<double, 6, 7> jac_dot;
  jac_dot.setZero();

  this->jntToJacDot(model, state, dq, goal_frame, &jac_dot);

  // Calculate end-effector acceleration in base frame
  Eigen::VectorXd base_acc =  jac_dot*dq + jacobian*ddq;

  // Convert to ee frame
  accel_out->head(3) = transform.linear().transpose()*base_acc.head(3);
  accel_out->tail(3) = transform.linear().transpose()*base_acc.tail(3);

  //print_to_file(*accel_out);

}

//Iterative parameter estimator - the functional bits

void data_matrix_fill(franka::RobotState robot_state, franka::Model& model, Eigen::Matrix<double, 6, 4>& data_matrix){

  data_matrix.setZero();
  Eigen::VectorXd xd(6), xdd(6);
  Eigen::Vector3d xd_om, xdd_om, xd_lin, xdd_lin;

  this->dynamic_variable_updater(model, robot_state, &xd, &xdd);

  Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
  Eigen::VectorXd grav_comp = transform.linear().transpose()*Eigen::Vector3d(0,0,-9.81);

  xd_lin = xd.head(3);
  xd_om = xd.tail(3);
  xdd_lin = xdd.head(3) - grav_comp; // should grav_comp be negative
  xdd_om = xdd.tail(3);

  // get angular velocity estimates from robot state or IMU - pretty much equivalent

  std::array<double, 42> jacobian_array =
  model.zeroJacobian(franka::Frame::kEndEffector, robot_state);

  Eigen::Map<const Eigen::Matrix<double, 6, 7> > jacobian(jacobian_array.data());
  Eigen::Map<const Eigen::Matrix<double, 7, 1> > q(robot_state.q.data());
  Eigen::Map<const Eigen::Matrix<double, 7, 1> > dq(robot_state.dq.data());

  // assemble data matrix elements

  // short version:
  data_matrix.block<3,1>(0,0) = xdd_lin; // is this already normalised to subtract R(g)?

  // upper half

  data_matrix(0,1) = -xd_om[1]*xd_om[1] - xd_om[0]*xd_om[0];
  data_matrix(1,1) = xd_om[0]*xd_om[1] + xdd_om[0];
  data_matrix(2,1) = xd_om[0]*xd_om[2] - xdd_om[1];

  data_matrix(0,2) = xd_om[0]*xd_om[1] - xdd_om[2];
  data_matrix(1,2) = -xd_om[0]*xd_om[0] - xd_om[2]*xd_om[2];
  data_matrix(2,2) = xd_om[1]*xd_om[2] + xdd_om[0];

  data_matrix(0,3) = xd_om[0]*xd_om[2] + xdd_om[1];
  data_matrix(1,3) = xd_om[1]*xd_om[2] - xdd_om[0];
  data_matrix(2,3) = -xd_om[1]*xd_om[1] - xd_om[0]*xd_om[0];

  // lower half

  data_matrix(3,1) = 0;
  data_matrix(4,1) = -data_matrix(2,0);
  data_matrix(5,1) = data_matrix(1,0);
  data_matrix(3,2) = data_matrix(2,0);
  data_matrix(4,2) = 0;
  data_matrix(5,2) = -data_matrix(0,0);
  data_matrix(3,3) = -data_matrix(1,0);
  data_matrix(4,3) = data_matrix(0,0);
  data_matrix(5,3) = 0;

  /*
  // convert this into block matrix entries, single run is too hard to keep track of
  data_matrix.block<3,1>(0,0) = xdd_lin;
  data_matrix(0,1) = -xd_om[1]*xd_om[1] - xd_om[0]*xd_om[0];
  data_matrix(1,1) = xd_om[0]*xd_om[1] + xdd_om[0];
  data_matrix(2,1) = xd_om[0]*xd_om[2] - xdd_om[1];
  data_matrix(0,2) = xd_om[0]*xd_om[1] - xdd_om[2];
  data_matrix(1,2) = -xd_om[0]*xd_om[0] - xd_om[2]*xd_om[2];
  data_matrix(2,2) = xd_om[1]*xd_om[2] + xdd_om[0];
  data_matrix(0,3) = xd_om[0]*xd_om[2] + xdd_om[1];
  data_matrix(1,3) = xd_om[1]*xd_om[2] - xdd_om[0];
  data_matrix(2,3) = -xd_om[1]*xd_om[1] - xd_om[0]*xd_om[0];

  data_matrix.block<3,6>(0,4) *= 0;

  data_matrix.block<3,1>(3,0) *= 0;

  data_matrix(3,1) = 0;
  data_matrix(4,1) = -data_matrix(2,0);
  data_matrix(5,1) = data_matrix(1,0);
  data_matrix(3,2) = data_matrix(2,0);
  data_matrix(4,2) = 0;
  data_matrix(5,2) = -data_matrix(0,0);
  data_matrix(3,3) = -data_matrix(1,0);
  data_matrix(4,3) = data_matrix(0,0);
  data_matrix(5,3) = 0;

  data_matrix(3,4) = xdd_om[0];
  data_matrix(4,4) = xd_om[0]*xd_om[2];
  data_matrix(5,4) = -xd_om[0]*xd_om[1];
  data_matrix(3,5) = -data_matrix(2,1);
  data_matrix(4,5) = data_matrix(2,2);
  data_matrix(5,5) = xd_om[0]*xd_om[0] - xd_om[1]*xd_om[1];
  data_matrix(3,6) = data_matrix(1,1);
  data_matrix(4,6) = xd_om[2]*xd_om[2] - xd_om[0]*xd_om[0];
  data_matrix(5,6) = data_matrix(1,3);

  data_matrix(3,7) = -xd_om[1]*xd_om[2];
  data_matrix(4,7) = xdd_om[1];
  data_matrix(5,7) =  xd_om[0]*xd_om[1];
  data_matrix(3,8) = xd_om[1]*xd_om[1] - xd_om[2]*xd_om[2];
  data_matrix(4,8) = data_matrix(0,2);
  data_matrix(5,8) = data_matrix(0,3);
  data_matrix(3,9) = xd_om[1]*xd_om[2];
  data_matrix(4,9) =  -xd_om[0]*xd_om[2];
  data_matrix(5,9) = xdd_om[2]; */


}


void svd_solver(Eigen::VectorXd& param_out){
  // start this as a NEW THREAD when we call the operator on the estimator

  // Change this back to an iterative updater that uses the reference frame error instead:

  // (a) preliminary phase, do an SVD on a reasonable sample set to get an initial value.
  // (b) iterative phase: update the parameters, use the current model to get an error estimate, iterate.

  // Declare large sample matrix and input vector
  int s_i = 0;
  // Initialise flag for moving from preliminary to iterative phase
  bool param_init = false;

  const int num_samples = 9;
  Eigen::Matrix<double, 6*num_samples, 4> combined_data_matrix;
  Eigen::VectorXd sample_wrench_vector(6*num_samples);

  Eigen::MatrixXd svd_U;
  Eigen::MatrixXd svd_V;

  // iterative elements
  Eigen::MatrixXd P_o, P_k;
  Eigen::MatrixXd K_k;

  //iterative weighting
  const std::chrono::time_point<std::chrono::steady_clock> est_start =
        std::chrono::steady_clock::now();
  float delta = 0.94;

  // pop data from global queues, when available, and fill dynamic matrix and sample wrench
  while (run_estimator) {
    try{
      Eigen::VectorXd wrench_input = subsampleWrench.pop().value(); // wrench input is in local ee frame
      Eigen::Matrix<double, 6,4> dynamic_input = subsampleDynMatrix.pop().value();

      if (!param_init){
          if (s_i < num_samples-1){
            combined_data_matrix.block<6,4>(s_i*6,0) = dynamic_input;
            sample_wrench_vector.segment(6*s_i,6) = wrench_input;
          }
          else {
            combined_data_matrix.block<6,4>(36, 0) = dynamic_input;
            sample_wrench_vector.segment(36,6) = wrench_input;

            // parameter_estimator(combined_data_matrix, sample_wrench_vector);
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(combined_data_matrix, Eigen::ComputeFullU | Eigen::ComputeFullV );
            svd_U = svd.matrixU();
            svd_V = svd.matrixV();

            Eigen::MatrixXd svd_S(svd_U.rows(), svd_V.rows());
            svd_S.setZero();
            svd_S.block<4,4>(0,0) = svd.singularValues().asDiagonal();  // might need to pad this.
            Eigen::MatrixXd temp_S_inv = svd_S.transpose();

            // can't invert a non-square matrix, so:
            for (int n = 0; n < svd_S.cols(); n++){
              if (svd_S(n,n) < 10e-6) { temp_S_inv(n,n) = 0; }
              else {temp_S_inv(n,n) = 1/svd_S(n,n);}
            }

            param_out = svd_V*temp_S_inv* (svd_U.transpose()*sample_wrench_vector);
            //Eigen::VectorXd print_input_data(Eigen::Map<Eigen::VectorXd>(dynamic_input.data(), dynamic_input.cols()*dynamic_input.rows()));
            // print_to_file(param_out);

            // Create a pseudo-inverse error matrix:
            P_o = (dynamic_input.transpose()*dynamic_input).inverse();

            // uncertainty is pretty high so maybe we should start with a different guess
            //std::cout << P_o << std::endl;

            param_init = true;
            //std::cout << "dynamic parameters initialised" << std::endl;
          }
          s_i++;
        }
      else {
        // initial parameter guess is in param_out
        // current dynamic matrix is in dynamic_input
        // current sensor data is in wrench_input
        // previous P matrix is in P_o

        // Don't estimate non-mass parameters until after lift mode is finished - but what does this mean for P_o?
        // actually, overall mass estimation isn't THAT good during lift phase. Can I start param est after lift?

        if (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - est_start).count() > 2000.0){delta = 0.999;}
        // Calculate expected wrench given current load estimation and current state:
        Eigen::VectorXd u_k_hat = dynamic_input*param_out;
        // Calculate error between expected and actual wrench:
        Eigen::VectorXd e_k = u_k_hat - wrench_input;
        K_k = P_o*(dynamic_input.transpose()) * (delta*Eigen::MatrixXd::Identity(6,6) - dynamic_input*P_o*(dynamic_input.transpose())).inverse();

        // P_k will always be larger than P_o due to delta factor ??
        P_k = (1/delta)*(Eigen::MatrixXd::Identity(4,4) + K_k*dynamic_input) * P_o;

        param_out = param_out + K_k*e_k;

        Eigen::VectorXd data_out(param_out.size()+1);
        data_out << (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - est_start).count()), param_out;

        //print_to_file(data_out);
        P_o = P_k;

        }
    }
    catch (std::bad_optional_access const& e){
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }


 }

};


#endif //_Dynamic_Utilities_hpp_
