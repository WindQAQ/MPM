#include "constant.h"
#include "grid.h"

__host__ __device__ Grid& Grid::operator=(const Grid& other) {
    idx = other.idx;
    force = other.force;
    velocity = other.velocity;
    velocity_star = other.velocity_star;
    mass = other.mass;

    return *this;
}

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
