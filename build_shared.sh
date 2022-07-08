c++ -std=c++17 \
-I '/lib/' \
-I '/usr/local/lib' \
-I '/usr/lib/x86_64-linux-gnu' \
-isystem '/usr/include/eigen3' \
shared_task_control.cpp examples_common.cpp LowPassFilter.cpp -o test \
-lpthread -lfranka -lncurses -lrealsense2
