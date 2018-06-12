#include "constant.h"
#include "grid.h"

__device__ void Grid::updateVelocity() {
    float inv_mass = 1.0f / mass;
    force(1) += (mass * GRAVITY);
    velocity *= inv_mass;
    velocity_star = velocity + TIMESTEP * inv_mass * force;
}

__device__ void Grid::applyBoundaryCollision() {
    float vn;
    Eigen::Vector3f vt, normal, pos((idx.cast<float>() * PARTICLE_DIAM * TIMESTEP).cwiseProduct(velocity_star));

    bool collision;

    for (int i = 0; i < 3; i++) {
        collision = false;
        normal = Eigen::Vector3f::Zero();

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

            velocity_star(i) = 0.0f;
            for (int j = 0; j < 3; j++) {
                if (j != i) {
                    velocity_star(j) *= STICKY_WALL;
                }
            }

            vt = velocity_star - vn * normal;

            if (vt.norm() <= -FRICTION * vn) {
                velocity_star = Eigen::Vector3f::Zero();
                return;
            }

            velocity_star = vt + FRICTION * vn *  vt.normalized();
        }
    }
}
