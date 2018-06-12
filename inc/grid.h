#ifndef GRID_H_
#define GRID_H_

#include <Eigen/Dense>

#include "constant.h"

class Grid {
  public:
    Eigen::Vector3i idx;
    float mass;
    Eigen::Vector3f force, velocity, velocity_star;

    __host__ __device__ Grid(): Grid(Eigen::Vector3i::Zero()) {}
    __host__ __device__ Grid(Eigen::Vector3i _idx): idx(_idx), mass(0.0f), force(0.0f, 0.0f, 0.0f), velocity(0.0f, 0.0f, 0.0f), velocity_star(0.0f, 0.0f, 0.0f) {}
    __host__ __device__ virtual ~Grid() {}

    __device__ void updateVelocity();
    __device__ void applyBoundaryCollision();
};

#endif  // GRID_H_
