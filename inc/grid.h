#ifndef GRID_H_
#define GRID_H_

#include <Eigen/Dense>

#include "constant.h"

class Grid {
  public:
    Eigen::Vector3i idx;
    Eigen::Vector3f force, velocity, velocity_star;
    float mass;
#if ENABLE_IMPLICIT
    Eigen::Vector3f v, r, p, delta_force, ar, ap;
    float rar_tmp;
#endif

    __host__ __device__ Grid() {}
    __host__ __device__ Grid(const Eigen::Vector3i _idx)
        : idx(_idx), mass(0.0f), force(0.0f, 0.0f, 0.0f), velocity(0.0f, 0.0f, 0.0f), velocity_star(0.0f, 0.0f, 0.0f)
#if ENABLE_IMPLICIT
        , v(0.0f, 0.0f, 0.0f), r(0.0f, 0.0f, 0.0f), p(0.0f, 0.0f, 0.0f), delta_force(0.0f, 0.0f, 0.0f),
          ar(0.0f, 0.0f, 0.0f), ap(0.0f, 0.0f, 0.0f), rar_tmp(0.0f)
#endif
    {}
    __host__ __device__ virtual ~Grid() {}

    __host__ __device__ void reset();
    __host__ __device__ void updateVelocity();
    __host__ __device__ void applyBoundaryCollision();
};

#endif  // GRID_H_
