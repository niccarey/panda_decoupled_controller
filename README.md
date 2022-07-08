# franka_panda_shared_control

Code base for decoupled task-based cooperative control of the franka panda

Necessary sub elements for (a) inertial estimation and (b) separating control algorithms based on task-space dimensions are found in Dynamic_Utilities.hpp and DecoupledController.hpp respectively

Current status: Controller decoupling successful, but input forces estimated through the joint torque sensors are still very noisy, leading to frequent lockups on the hardware side due to control torque jumps. Urgent to-do: improve smoothing filter on input/task-based torque. Potential long-term fix: switch to wrist force/torque sensor and see if that improves the signal.

Other potential issues: to calculate necessary control torques in the task/null framework, DecoupledController runs as a parallel asynchronous thread (because the processor overhead is too high for this to loop within the real time kernel control signal time envelope). Since its output torques are based on the current robot state, it draws timestamped RobotState data from the hardware control loop, via a shared queue. As the decoupled control loop is frequently slightly slower than the real time hardware loop, there is the possibility of (a) controller lag and (b) buffer overflow, especially in the case of long activity. Improving the efficiency of the decoupled control calculations and bringing the core elements into the main realtime control loop would fix both issues. 
