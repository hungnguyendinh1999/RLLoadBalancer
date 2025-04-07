cmake_minimum_required(VERSION 3.25.1)
project(RLLoadBalancer)

find_package(Torch REQUIRED)
find_package(Matplot++ REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")

set(CMAKE_CXX_STANDARD 20)

add_executable(RLLoadBalancer main.cpp
        Environment/Environment.h
        Environment/Task.h
        Environment/Task.cpp
        Environment/Environment.cpp
        Environment/Processor.cpp
        Environment/Processor.h
        Agent/BaseAgent.h
        Agent/DQNAgent.h
        Agent/DQNAgent.cpp
        Agent/QLAgent.cpp
        Agent/QLAgent.h
        Agent/DPAgent.cpp
        Agent/DPAgent.h
        Agent/FFN.cpp
        Agent/FFN.h
        Agent/ReplayBuffer.cpp
        Agent/ReplayBuffer.h
        Utils/ProgressBar.cpp
        Utils/ProgressBar.h
        DecayScheduler/DecayScheduler.h
        DecayScheduler/LinearDecayScheduler.cpp
        DecayScheduler/LinearDecayScheduler.h
        DecayScheduler/ExponentialDecayScheduler.cpp
        DecayScheduler/ExponentialDecayScheduler.h
        DecayScheduler/DecayScheduler.cpp
        Utils/Plot.cpp
        Utils/Plot.h
)
target_link_libraries(RLLoadBalancer "${TORCH_LIBRARIES}" Matplot++::matplot)

# add_executable(test_replay_buffer test_replay_buffer.cpp
#         Agent/ReplayBuffer.cpp
#         Agent/ReplayBuffer.h
#         Environment/Environment.h
#         Environment/Task.h
#         Environment/Task.cpp
#         Environment/Environment.cpp
#         Environment/Processor.cpp
#         Environment/Processor.h)
# target_link_libraries(test_replay_buffer ${TORCH_LIBRARIES})

# add_executable(test_ffn test_ffn.cpp
#         Agent/FFN.cpp
#         Agent/FFN.h
#         Environment/Environment.h
#         Environment/Task.h
#         Environment/Task.cpp
#         Environment/Environment.cpp
#         Environment/Processor.cpp
#         Environment/Processor.h)
# target_link_libraries(test_ffn ${TORCH_LIBRARIES})

add_executable(test_dqn_agent test_dqn_agent.cpp
        Agent/DQNAgent.h
        Agent/DQNAgent.cpp
        Agent/ReplayBuffer.cpp
        Agent/ReplayBuffer.h
        Agent/FFN.cpp
        Agent/FFN.h
        Environment/Environment.h
        Environment/Task.h
        Environment/Task.cpp
        Environment/Environment.cpp
        Environment/Processor.cpp
        Environment/Processor.h
        Utils/ProgressBar.cpp
        Utils/ProgressBar.h
        DecayScheduler/DecayScheduler.cpp
        DecayScheduler/DecayScheduler.h
        DecayScheduler/ExponentialDecayScheduler.cpp
        DecayScheduler/ExponentialDecayScheduler.h
        DecayScheduler/LinearDecayScheduler.cpp
        DecayScheduler/LinearDecayScheduler.h
)
target_link_libraries(test_dqn_agent ${TORCH_LIBRARIES})

