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

__host__ __device__ void Grid::applyBoundaryCollision() {
    float vn;
    Eigen::Vector3f vt, normal, pos((idx.cast<float>() * PARTICLE_DIAM) + (TIMESTEP * velocity_star));

    bool collision;

    for (int i = 0; i < 3; i++) {
        collision = false;
        normal.setZero();

        if (pos(i) <= BOX_BOUNDARY_1) {
            collision = true;
            normal(i) = 1.0f;
        }
        else if (pos(i) >= BOX_BOUNDARY_2) {
            collision = true;
            normal(i) = -1.0f;
        }

        if (collision) {
            vn = velocity_star.dot(normal);

            if (vn >= 0.0f) continue;

            for (int j = 0; j < 3; j++) {
                if (j != i) {
                    velocity_star(j) *= STICKY_WALL;
                }
            }

            vt = velocity_star - vn * normal;

            if (vt.norm() <= -FRICTION * vn) {
                velocity_star.setZero();
                return;
            }

            velocity_star = vt + FRICTION * vn *  vt.normalized();
        }
    }
}
