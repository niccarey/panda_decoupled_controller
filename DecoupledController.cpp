
#ifndef _DecoupledController_hpp_
#define _DecoupledController_hpp_

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

#include <franka/duration.h>
#include <franka/exception.h>
#include <franka/model.h>
#include <franka/rate_limiting.h>
#include <franka/robot.h>
#include <franka/gripper.h>



// Controller class:
// - has an associated boolean we can switch off to shut down calculation threads.
// - all the necessary state matrices etc can be variables instatiated upon initialisation


class DecoupledController{

  // The assumption is that the function owner is running parallel control threads and is responsible for
  // choosing the correct value wrt to the dimension vector state
  // We do this to (1) avoid resizing dynamic elements, and (2) to maintain dynamic continuity across changes in the task dimension state

  // This might not be the best method but optimality can wait.

    public:
      bool task_flag; // should be set to true on initialisation unless otherwise required
      bool state_init_flag;


      franka::Duration duration;
      Eigen::Matrix<double, 6,7> full_jac_prev;
      Eigen::Matrix<double, 7,7> mass_matrix_prev;

      Eigen::VectorXi cntr_vec;
      Eigen::VectorXd goal_ee_position;


  DecoupledController(){
    task_flag = true;
    // If we want to make this dimension agnostic, then we need to get rid of a lot of these pre-allocations
    // and use local variables only to create core calcs.

  }

  void set_task_gains(float d_gain, float k_gain,Eigen::Ref< Eigen::MatrixXd>task_stiffness_gain,  Eigen::Ref<Eigen::MatrixXd> task_damp_gain, int task_index){
    task_damp_gain = d_gain*Eigen::MatrixXd::Identity(task_index, task_index);
    task_stiffness_gain = k_gain*Eigen::MatrixXd::Identity(task_index, task_index);
  }


  void set_null_damping(float bv_gain,  Eigen::Ref<Eigen::MatrixXd> null_damp_gain, int task_index){
    null_damp_gain = bv_gain*Eigen::MatrixXd::Identity(7-task_index, 7-task_index);
  }

  void set_null_stiffness(Eigen::VectorXd kd_null_gain, Eigen::Ref<Eigen::MatrixXd> null_stiffness_gain){
    null_stiffness_gain = kd_null_gain.asDiagonal();
  }

  void populate_subJacobian(Eigen::Ref<Eigen::MatrixXd>t_J, Eigen::Matrix<double, 6,7>& full_jacobian){
      // Can extract task dimensions straight from full jacobian
      //for (auto dim_index : cntr_vec){
      int task_index = 0;
      t_J.setZero(); // probably unecessary

      for (int di =0; di < cntr_vec.size(); di ++){
        if (cntr_vec(di) > 0) {
          t_J.row(task_index) = full_jacobian.row(di);
          task_index++;
        }
      }
    }

  void populate_nullJacobian(Eigen::Ref<Eigen::MatrixXd>n_Z, Eigen::Ref<Eigen::MatrixXd>t_J){

    int task_index = 0;
    for (int di =0; di < cntr_vec.size(); di ++){ if (cntr_vec(di) > 0) {task_index++;} }
    int null_index = 7 - task_index;

    // It may be necessary (!) to reorder the columns of t_J to extract an invertable J_m? Unclear how this would affect subsequent operations
    // Might come out in the wash, since we perform internal operations which might result in col-independence.

    // IDEALLY: Check chosen Jm for invertability before doing the following:
    // YEP we get a problem in certain dimensions! How do we fix this?
    Eigen::MatrixXd J_m(task_index, task_index);
    Eigen::MatrixXd J_r(task_index, null_index);

    J_m = t_J.block(0, 0 , task_index, task_index);
    J_r = t_J.block(0 ,task_index, task_index, null_index);

    // Stable pseudo inverse construction:
    Eigen::MatrixXd JmPinv = J_m.fullPivHouseholderQr().inverse();

    Eigen::MatrixXd nullProjection = -J_r.transpose()*(JmPinv).transpose();

    n_Z.block(0,0, task_index, null_index) = nullProjection.transpose();
    n_Z.block(null_index-1, 0, null_index, null_index) =  Eigen::MatrixXd::Identity(null_index,null_index);
  }

  void task_jac_inv(Eigen::Ref<Eigen::MatrixXd>t_J, Eigen::Ref<Eigen::MatrixXd>t_J_Inv, Eigen::Matrix<double, 7,7>& mass_matrix){
        t_J_Inv = mass_matrix.inverse()*t_J.transpose()*(t_J*mass_matrix.inverse()*t_J.transpose()).inverse();
  }

  void null_jac_inv(Eigen::Ref<Eigen::MatrixXd>n_Z, Eigen::Ref<Eigen::MatrixXd>n_Z_Inv, Eigen::Matrix<double, 7,7>& mass_matrix){
    n_Z_Inv = (n_Z.transpose()*mass_matrix*n_Z).inverse()*n_Z.transpose()*mass_matrix;
  }

  void lp_filter_array_init(int cntr, std::vector<LowPassFilter>& output_filter_array, float th_hz, float dt){
    int i = 0;
    while (i < cntr){
      output_filter_array.push_back(LowPassFilter(th_hz,dt));
      i++;
    }
  }

  float sig_filt(LowPassFilter filter, float signal){
   return(filter.update(signal));
  }


  Eigen::VectorXd lp_torque_filter(Eigen::Ref<Eigen::VectorXd> input_vector, std::vector<LowPassFilter>& lpf_array){
    if (input_vector.size() != lpf_array.size()){
      // log error
      std::cout << "Warning: low pass filter has wrong dimensions" << std::endl;
      return input_vector;
    }

    Eigen::VectorXd updated_torque(input_vector.size());

    for (int i = 0; i< input_vector.size(); i++){
      updated_torque(i) = sig_filt(lpf_array[i], input_vector(i));
    }
    return updated_torque;
  }

//  float num_derivative(float s_prev, float s_current, float dt){
 //   return((s_current-s_prev)*(1.0/dt));
 // }

  void matrix_deriv(float dt, Eigen::Ref<Eigen::MatrixXd> mat_current, Eigen::Ref<Eigen::MatrixXd> mat_prev, Eigen::Ref<Eigen::MatrixXd> mat_dot, int mat_size){
    // for (auto mat_item : mat_current){
    // catch divide by zero errorsf
    mat_dot = mat_current - mat_prev;
    mat_dot /= dt;
    //for (int j = 0; j < mat_size; j++){ mat_dot(j) = this->num_derivative(mat_prev(j), mat_current(j), dt);}
  }

  void calc_task_mass_matrix(Eigen::Ref<Eigen::MatrixXd> t_J, Eigen::Ref<Eigen::MatrixXd> task_mass_matrix, Eigen::Matrix<double, 7, 7>& full_mass_matrix){
    task_mass_matrix = (t_J*full_mass_matrix.inverse()*t_J.transpose()).inverse();
  }

  void calc_null_mass_matrix(Eigen::Ref<Eigen::MatrixXd> n_Z, Eigen::Ref<Eigen::MatrixXd> null_mass_matrix, Eigen::Matrix<double, 7, 7>& full_mass_matrix){
    null_mass_matrix = n_Z.transpose()*full_mass_matrix*n_Z;
  }

  void transform_to_state(Eigen::Affine3d& state_transform, Eigen::VectorXd& full_state_vector){
    // Take current end-effector pose and transform to a state vector (x,y,z, r, p, y)
    // When computing state error vector, it may be necessary to convert angle resultants to sinusoids and scale
    // in order to avoid discontinuities. Or convert all angular states back to full quaternions and recompute dimensional error vectors (this might be better)

    full_state_vector.head(3) = state_transform.translation();
    full_state_vector.tail(3) = state_transform.rotation().eulerAngles(0,1,2);

  }

  void populate_task_state(Eigen::Ref<Eigen::VectorXd>task_state, Eigen::VectorXd& full_state_vector){
    // Qu: is there a way of generating the task state using the task Jacobian? (would be a linearized approximation, might be dodgy
    // k is updating weird?
    int k = 0;
    for(int i=0; i< cntr_vec.size(); i++){
      if (cntr_vec(i) > 0){
        //td::cout << k << std::endl;
        task_state(k) = full_state_vector(i); ++k;}
    }
  }

  void populate_task_state_velocity(Eigen::Ref<Eigen::MatrixXd>t_J, Eigen::Ref<Eigen::VectorXd> task_state_vel, Eigen::VectorXd& qd){
    task_state_vel = t_J*qd;
  }

  void populate_null_state(Eigen::Ref<Eigen::MatrixXd> n_Z_Inv, Eigen::Ref<Eigen::VectorXd> null_state, Eigen::VectorXd& qd){
    null_state = n_Z_Inv*qd;
  }

  Eigen::VectorXd calculate_state_error(Eigen::Affine3d& goal_transform, Eigen::Affine3d& current_transform, int task_index){

    Eigen::VectorXd error_state_vector(6);
    // If angular errors are near zero, we can encounter singularities in axis/angle representation.
    // To handle this, we convert angular errors to a scaled sinusoid (could also do a filtered unwrap).

    error_state_vector.head(3) = goal_transform.translation() - current_transform.translation();
    Eigen::Quaterniond rotation_error(current_transform.linear()*(goal_transform.linear().inverse()));

    Eigen::AngleAxisd unpack_rotation(rotation_error);

    Eigen::Vector3d rescale_rotation;
    rescale_rotation = unpack_rotation.axis()*(std::sin(unpack_rotation.angle()));

    error_state_vector.tail(3) = rescale_rotation;

    Eigen::VectorXd task_error(task_index);

    // extract task components
    int k = 0;
    for(int i=0; i< cntr_vec.size(); i++){
      if (cntr_vec(i) > 0){task_error(k) = error_state_vector(i); k++;}
    }

    return(task_error);

  }

  Eigen::VectorXd calculate_null_error(Eigen::Affine3d& goal_transform, Eigen::Affine3d& current_transform){

    Eigen::VectorXd error_state_vector(6);
    // If angular errors are near zero, we can encounter singularities in axis/angle representation.
    // To handle this, we convert angular errors to a scaled sinusoid (could also do a filtered unwrap).

    error_state_vector.head(3) = goal_transform.translation() - current_transform.translation();
    Eigen::Quaterniond rotation_error(current_transform.linear()*(goal_transform.linear().inverse()));

    Eigen::AngleAxisd unpack_rotation(rotation_error);

    Eigen::Vector3d rescale_rotation;
    rescale_rotation = unpack_rotation.axis()*(std::sin(unpack_rotation.angle()));

    error_state_vector.tail(3) = rescale_rotation;

    Eigen::VectorXd null_error(6);
    null_error.setZero();

    // Set null components but leave full error vector size
    for(int i=0; i< cntr_vec.size(); i++){
      if (cntr_vec(i) < 1){null_error(i) = error_state_vector(i);}
    }

    return(null_error);

  }

  Eigen::VectorXd calculate_joint_error(Eigen::VectorXd& q, Eigen::VectorXd& goal_joint_state){
    // todo: unwrapping checks?
    return(goal_joint_state - q);
  }

  void update_task_coriolis(Eigen::Ref<Eigen::MatrixXd> t_J_Inv, Eigen::Ref<Eigen::MatrixXd> t_J_dot, Eigen::Ref<Eigen::MatrixXd>task_mass_matrix, Eigen::Ref<Eigen::MatrixXd> task_coriolis, Eigen::Matrix<double, 7,7>& full_coriolis_matrix){
    task_coriolis = (t_J_Inv.transpose()*full_coriolis_matrix - task_mass_matrix*t_J_dot)*t_J_Inv;
  }

  void update_null_coriolis(Eigen::Ref<Eigen::MatrixXd> n_Z, Eigen::Ref<Eigen::MatrixXd> n_Z_Inv_dot, Eigen::Ref<Eigen::MatrixXd>null_mass_matrix, Eigen::Ref<Eigen::MatrixXd> null_coriolis, Eigen::Matrix<double, 7,7>& full_coriolis_matrix){
    null_coriolis = (n_Z.transpose()* full_coriolis_matrix - null_mass_matrix*n_Z_Inv_dot)*n_Z;
  }


  void run(franka::Model& model, Eigen::Matrix<double,7,7>& full_coriolis_matrix, Eigen::Affine3d& goal_transform, franka::RobotState& init_state,
    Eigen::VectorXd& task_torque, Eigen::VectorXd& null_torque){

    // Note: provided these elements are excecuted swiftly, we could move the control loop
    // into the main thread. Try it once it's working.

    // IF GAINS ARE ADAPTIVE they also need to be passed as a shared reference

    // If the torque calculations are fast enough, we could simply return the necessary control torque each loop.
    // Check timing on decoupled_control_loop, and for now, try updating parallel shared arrays.

    // Testing controller: Set actual command torques to 'balance' mode and simply print control torques to file.

    state_init_flag = false;

    std::vector<LowPassFilter> lpf_task_torque;
    std::vector<LowPassFilter> lpf_null_torque;

    // other constants
    float cutoff = 50.0;
    float dt_est = 0.001;

    // Initialise low pass filters: these will always be 7x1
    this->lp_filter_array_init(7, lpf_task_torque, 30.0, dt_est);
    this->lp_filter_array_init(7, lpf_null_torque, 50.0, dt_est);

    //Eigen::VectorXd previous_input_wrench(6); // we always know how big this is
    //previous_input_wrench.setZero(); // and it's fine to start at zero

    // Note: Since this is not event-driven, we have no check on running the same data twice (leading to jumps in first order functions)
    // so moved to stack-based control, though this runs the risk of mem buffer overflow on long sequences. Need to deal with this at some point

    std::cout << "starting parallel control loop" << std::endl;
    // need to wait to ensure coriolis matrix is populated?

    std::this_thread::sleep_for(std::chrono::milliseconds(500));


    output_stream.open("task_dim_error.csv");  //, std::ofstream::out);
    force_stream.open("task_torque.csv");

    Eigen::Map<const Eigen::Matrix<double, 7, 1> > q_map(init_state.q.data());
    Eigen::VectorXd goal_joint_state(q_map);

    std::vector<CircularBuffer> torque_buffer;
    for (int i = 0; i < goal_joint_state.size(); i++){
      torque_buffer.push_back(CircularBuffer(100));
    }

    while (task_flag){
      decoupled_control_loop(model, full_coriolis_matrix, goal_transform, goal_joint_state, task_torque, null_torque, lpf_task_torque, lpf_null_torque, torque_buffer);
    }

    // Can also low pass filter the torque command outside the control loop (probably).
    output_stream.close();
    force_stream.close();

  }

  void decoupled_control_loop(franka::Model& model, Eigen::Matrix<double, 7,7>& full_coriolis_matrix, Eigen::Affine3d& goal_transform, Eigen::VectorXd& goal_joint_state,
    Eigen::VectorXd& task_torque, Eigen::VectorXd& null_torque, std::vector<LowPassFilter>& lpf_task_torque, std::vector<LowPassFilter>& lpf_null_torque, std::vector<CircularBuffer>& torque_buffer){

    // Outputs (shared as predefined inputs)
    //    task space torque vector
    //    null space torque vector

    // Local variables:

    float dyn_task_stiffness = 20.0;
    float dyn_task_damping = 5.0;
    float dyn_null_damping = 1.0;

    try{
        franka::RobotState robot_state = DecoupledControlStates.pop().value();

        Eigen::VectorXd goal_null_stiffness(7);
        Eigen::VectorXd vec1(3);
        vec1 << 1, 0.5 ,0.5;
        Eigen::VectorXd vec2(4);
        vec2 << 0.5, 0.5, 0.5, 0.5;
        goal_null_stiffness << vec1, vec2;
        goal_null_stiffness *= 10.0;

        Eigen::VectorXd full_state_vector(6);

        std::array<double, 42> jacobian_array =
        model.zeroJacobian(franka::Frame::kEndEffector, robot_state);
        std::array<double, 49> mass_array = model.mass(robot_state);

        Eigen::Matrix<double, 6, 7> jacobian(Eigen::Matrix<double, 6,7>::Map(jacobian_array.data()));
        Eigen::Matrix<double, 7, 7> mass_matrix(Eigen::Matrix<double, 7,7>::Map(mass_array.data()));

        // unpack task and joint states from the state vector
        Eigen::Map<const Eigen::Matrix<double, 7, 1> > est_ext_torque(robot_state.tau_ext_hat_filtered.data());
        Eigen::Map<const Eigen::Matrix<double, 7, 1> > q_map(robot_state.q.data());
        Eigen::Map<const Eigen::Matrix<double, 7, 1> > dq_map(robot_state.dq.data());
        Eigen::Map<const Eigen::Matrix<double, 6, 1> > wrench_map(robot_state.O_F_ext_hat_K.data());

        Eigen::VectorXd q(q_map);
        Eigen::VectorXd dq(dq_map);
        Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
        Eigen::VectorXd external_wrench(wrench_map);

        // should probably LP filt the wrench also?


        // RECONFIGURE CONTROL CALCULATIONS: Take back to a raw loop using directly calculated local variables.
        if (!state_init_flag) {
          // populate previous jacobian.
          full_jac_prev = jacobian;
          mass_matrix_prev = mass_matrix;
          state_init_flag = true;
          duration = robot_state.time;
        }

        else {

          // good to go
          // Calculate the state vector based on current task domain inputs:
          // Note: task domain vector can be a globally accessible variable which is set by the primary robot driver loop
          // so leave this as cntrl_vec

          float dt = (robot_state.time - duration).toMSec()/1000.0;
          duration = robot_state.time;
          //std::cout << dt << std::endl;

          this-> transform_to_state(transform, full_state_vector); // always the same size, can be pre-allocated

          // Initialise task states, subjacobians, inversions, etc based on current task domain status:
          int task_index = 0;
          // auto iterator only works for Eigen 3.4 and above: todo, upgrade Eigen
          //for (auto dim_index : cntr_vec){
          for (int di =0; di < cntr_vec.size(); di ++){ if (cntr_vec(di) > 0) {task_index++;} }

          // Preallocating memory (locally accessible variables!)
          Eigen::MatrixXd t_J = Eigen::MatrixXd::Zero(task_index, 7);
          Eigen::MatrixXd n_Z = Eigen::MatrixXd::Zero(7, 7-task_index);

          Eigen::MatrixXd t_J_Inv = Eigen::MatrixXd::Zero(t_J.cols(), t_J.rows());
          Eigen::MatrixXd n_Z_Inv = Eigen::MatrixXd::Zero(n_Z.cols(), n_Z.rows());

          Eigen::MatrixXd t_J_prev = Eigen::MatrixXd::Zero(t_J.rows(), t_J.cols());
          Eigen::MatrixXd n_Z_prev = Eigen::MatrixXd::Zero(7, 7-task_index);

          Eigen::MatrixXd n_Z_Inv_prev = Eigen::MatrixXd::Zero(n_Z.cols(), n_Z.rows());

          Eigen::MatrixXd t_J_dot = Eigen::MatrixXd::Zero(t_J.rows(), t_J.cols());
          Eigen::MatrixXd n_Z_Inv_dot = Eigen::MatrixXd::Zero(n_Z_Inv.rows(), n_Z_Inv.cols());

          Eigen::VectorXd task_state = Eigen::VectorXd(task_index);
          Eigen::VectorXd null_state = Eigen::VectorXd(7-task_index); // check variable sizing
          Eigen::VectorXd task_state_vel = Eigen::VectorXd(task_index);

          Eigen::MatrixXd task_mass_matrix = Eigen::MatrixXd(task_index, task_index);
          Eigen::MatrixXd null_mass_matrix = Eigen::MatrixXd(7-task_index, 7-task_index);

          Eigen::MatrixXd task_coriolis = Eigen::MatrixXd(task_index, task_index);
          Eigen::MatrixXd null_coriolis = Eigen::MatrixXd(7-task_index, 7-task_index);

          Eigen::MatrixXd task_damp_gain = Eigen::MatrixXd::Identity(task_index, task_index);
          Eigen::MatrixXd task_stiffness_gain = Eigen::MatrixXd::Identity(task_index, task_index);

          Eigen::MatrixXd null_damp_gain = Eigen::MatrixXd::Identity(7-task_index, 7-task_index);
          Eigen::MatrixXd null_stiffness_gain = Eigen::MatrixXd::Identity(7,7);

          // Populate control elements:

          this->populate_task_state(task_state, full_state_vector);
          this->populate_subJacobian(t_J, jacobian);
          this->populate_nullJacobian(n_Z, t_J); // will have to pass by reference

          this->task_jac_inv(t_J, t_J_Inv, mass_matrix);
          this->null_jac_inv(n_Z, n_Z_Inv, mass_matrix);

          this->populate_null_state(n_Z_Inv, null_state, dq);
          this->populate_task_state_velocity(t_J, task_state_vel, dq);

          this->populate_subJacobian(t_J_prev,full_jac_prev);
          this->populate_nullJacobian(n_Z_prev, t_J_prev);

          this-> null_jac_inv(n_Z_prev, n_Z_Inv_prev, mass_matrix_prev);
          this-> matrix_deriv(dt, t_J, t_J_prev, t_J_dot, t_J.size()); // will need to pass by reference
          this-> matrix_deriv(dt, n_Z_Inv, n_Z_Inv_prev, n_Z_Inv_dot, n_Z_Inv.size()); // inverted null Jacobian derivative

          this-> calc_task_mass_matrix(t_J, task_mass_matrix, mass_matrix);
          this-> calc_null_mass_matrix(n_Z, null_mass_matrix, mass_matrix);

          this-> update_task_coriolis(t_J_Inv, t_J_dot, task_mass_matrix, task_coriolis, full_coriolis_matrix);
          this-> update_null_coriolis(n_Z, n_Z_Inv_dot, null_mass_matrix, null_coriolis, full_coriolis_matrix);

          Eigen::VectorXd xe = this-> calculate_state_error(goal_transform, transform, task_index); // may need a scaling factor on (some) state error elements
          Eigen::VectorXd ne = this-> calculate_null_error(goal_transform, transform);
          Eigen::VectorXd qe = this-> calculate_joint_error(q, goal_joint_state);

          // Could set the task gains internally, if they're based on error functions?

          // TODO: Function, update task gains

          this->set_task_gains(dyn_task_damping, dyn_task_stiffness, task_stiffness_gain,  task_damp_gain,task_index);
          this->set_null_damping(dyn_null_damping, null_damp_gain, task_index);
          this->set_null_stiffness(goal_null_stiffness, null_stiffness_gain);

          // Get task-dimension-based stiffness
          Eigen::VectorXd task_domain_wrench = Eigen::VectorXd(task_index);
          Eigen::VectorXd null_domain_wrench = Eigen::VectorXd(6); // full task space vector


          Eigen::VectorXd torque_input_scaling(6);
          torque_input_scaling.head(3) << 1.0, 1.0, 1.0;
          torque_input_scaling.tail(3) << 0.1, 0.1, 0.1;
          Eigen::MatrixXd gamma_task_scaling = torque_input_scaling.asDiagonal();


          /*
          for (int i=0; i< external_wrench.size(); i++){
            external_wrench(i) = franka::lowpassFilter(dt, external_wrench(i), prev_wrench(i), 20.0);
          }
          prev_wrench = external_wrench; */

          external_wrench = gamma_task_scaling*external_wrench;



          int k = 0;
          for (int di =0; di < cntr_vec.size(); di ++){
            if (cntr_vec(di) > 0) {
              task_domain_wrench(k) = external_wrench(di);
              k++;
            }
          }

          // Not sure we need this
          null_domain_wrench.setZero();
          for (int di =0; di < cntr_vec.size(); di ++){
            if (cntr_vec(di) < 1) {
              null_domain_wrench(di) = external_wrench(di);
            }
          }

          task_torque.setZero();
          null_torque.setZero();

          task_torque += -t_J.transpose()*task_mass_matrix*(t_J_dot*dq);
          task_torque += -t_J.transpose()*task_mass_matrix*(task_damp_gain * task_state_vel);
          // low gain while we tune null behaviour
          task_torque += 0.5*t_J.transpose()*task_mass_matrix*(-2.0*task_domain_wrench); // current state: following external forces

          // Task torque is still VERY noisy. LPFilt isn't doing much.
          // could run a very wide moving average filter over the data?

          task_torque = lp_torque_filter(task_torque, lpf_task_torque);
          
          for (int i = 0; i < task_torque.size(); i++){
            torque_buffer[i].pushVal(task_torque(i));
            if (torque_buffer[i].isfull()){
              task_torque(i) = torque_buffer[i].meanBuffer();
            }
          }

          /* do a nan check:
          for (int i=0; i < task_torque.size(); i++){
              if (std::isnan(task_torque(i))) {task_torque(i) = 0;}
              if (std::isinf(task_torque(i))) {task_torque(i) = 0;}
            }*/


          // (1) need LP filter on output torque(s)
          // (2) need to tune task torque gains.

          // Error ratios are probably off, actually
          ne = gamma_task_scaling*ne;
          ne(2) *= -1.0; // not sure if this is an issue in all cartesian dimensions or just vertical
          ne.head(3) *= 2.5;
          ne.tail(3) *= 1.5;
          //ne.tail(3) *= 0.2; // bring the angular gain right down
          null_torque += -16.0*n_Z_Inv.transpose()*(n_Z.transpose()*null_stiffness_gain*(jacobian.transpose()*ne));
          // position error correction - TODO update to a well-conditioned jacobian inverse??


          Eigen::MatrixXd null_unity_check = n_Z_Inv.transpose()*n_Z.transpose();
          Eigen::VectorXd null_unity_list(Eigen::Map<Eigen::VectorXd>(null_unity_check.data(), null_unity_check.size()));
          //Eigen::VectorXd jacobian_list(Eigen::Map<Eigen::VectorXd>(jacobian.data(), jacobian.size()));

          Eigen::VectorXd null_torque_inertials(7);

          null_torque_inertials = -n_Z_Inv.transpose()*null_mass_matrix*n_Z_Inv_dot*dq;
          for (int i = 0; i < null_torque_inertials.size(); i++){
            if (null_torque_inertials(i) > 2){null_torque_inertials(i) = 2;}
            else if(null_torque_inertials(i) < -2) {null_torque_inertials(i) = -2;}
          }

          null_torque += 0.1*null_torque_inertials;
          null_torque += -0.1*n_Z_Inv.transpose()*((null_coriolis + null_damp_gain)*null_state); // damping and compensation torques


          /*
          for (int i=0; i < null_torque.size(); i++){
              if (std::isnan(null_torque(i))){null_torque(i) = 0;}
              if (std::isinf(null_torque(i))) {task_torque(i) = 0;}
          }*/
          null_torque = lp_torque_filter(null_torque, lpf_null_torque);

          //Eigen::VectorXd task_torque_projected = jacobian*task_torque; // This is an approximation -  not sure there's an easy way to get the actual projection
          //Eigen::VectorXd null_torque_projected = jacobian*null_torque; // This is an approximation -  not sure there's an easy way to get the actual projection

          // Publish control elements to file:
          std::thread printing_state_error(&DecoupledController::print_to_file, this, std::ref(task_domain_wrench));
          printing_state_error.join();

          std::thread printing_task_command(&DecoupledController::print_to_force_file, this, std::ref(task_torque));
          printing_task_command.join();


          // TODO: ADD LOW PASS FILTER HERE

          full_jac_prev = jacobian;
          mass_matrix_prev = mass_matrix;
          //goal_joint_state = q; // for position following, update qe
        }

      }
      catch (std::bad_optional_access const& exception){
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                // probably nothing to pop
              }
      catch (const franka::Exception& e){std::cout << "error in decoupling loop: " <<  e.what()<< std::endl; }
    }

    // set up printing function
    void print_to_file(Eigen::VectorXd& print_data) {
      for(size_t i = 0, size = print_data.size(); i < size; i++) {
        output_stream << print_data.data()[i] << ", "; }
        output_stream << std::endl;
      }

      void print_to_force_file(Eigen::VectorXd& print_data) {
        for(size_t i = 0, size = print_data.size(); i < size; i++) {
          force_stream << print_data.data()[i] << ", "; }
          force_stream << std::endl;
        }
};

#endif //_DecoupledController_hpp_