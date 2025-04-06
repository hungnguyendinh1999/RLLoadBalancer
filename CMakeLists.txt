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
        Utils/ProgressBar.cpp
        Utils/ProgressBar.h
        DecayScheduler/DecayScheduler.h
        DecayScheduler/LinearDecayScheduler.cpp
        DecayScheduler/LinearDecayScheduler.h
        DecayScheduler/DecayScheduler.cpp
        Utils/Plot.cpp
        Utils/Plot.h
)
target_link_libraries(RLLoadBalancer "${TORCH_LIBRARIES}" Matplot++::matplot)