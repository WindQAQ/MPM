#include "constant.h"
#include "grid.h"

__host__ __device__ void Grid::reset() {
    mass = 0.0f;
    force.setZero();
    velocity.setZero();
    velocity_star.setZero();
}

__host__ __device__ void Grid::updateVelocity() {
    if (mass > 0.0f) {
        float inv_mass = 1.0f / mass;
        force += (mass * GRAVITY);
        velocity *= inv_mass;
        velocity_star = velocity + TIMESTEP * inv_mass * force;
    }
}
