#ifndef GRID_H_
#define GRID_H_

#include <Eigen/Dense>

#include "constant.h"

class Grid {
  public:
    Eigen::Vector3i idx;
    Eigen::Vector3f force, velocity, velocity_star;
    float mass;

    __host__ __device__ Grid() {}
    __host__ __device__ Grid(const Eigen::Vector3i _idx): idx(_idx), mass(0.0f), force(0.0f, 0.0f, 0.0f), velocity(0.0f, 0.0f, 0.0f), velocity_star(0.0f, 0.0f, 0.0f) {}
    __host__ __device__ virtual ~Grid() {}

    __host__ __device__ Grid& operator=(const Grid&);

    __host__ __device__ void reset();
    __host__ __device__ void updateVelocity();
    __host__ __device__ void applyBoundaryCollision();
};

#endif  // GRID_H_
