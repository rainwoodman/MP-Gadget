cmake_minimum_required(VERSION 2.8)

project(bigfile)

# Finding optional dependencies
find_package(MPI)
find_package(GSL)

# Add library subdirectoy
add_subdirectory(src)

# Add executables subdirectory
add_subdirectory(utils)
