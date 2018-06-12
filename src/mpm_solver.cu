#include <Eigen/Dense>

#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

#include "mpm_solver.h"

__device__ float NX(const float& x) {
    if (x < 1.0f) {
        return 0.5f * (x * x * x) - (x * x) + (2.0f / 3.0f);
    }
    else if (x < 2.0f) {
        return (-1.0f / 6.0f) * (x * x * x) + (x * x) - (2.0f * x) + (4.0f / 3.0f);
    }
    else {
        return 0.0f;
    }
}

__device__ float dNX(const float& x) {
    float abs_x = fabs(x);
    if (abs_x < 1.0f) {
        return (1.5f * abs_x * x) - (2.0f * x);
    }
    else if (abs_x < 2.0f) {
        return -0.5f * (abs_x * x) + (2.0f * x) - (2.0f * x / abs_x);
    }
    else {
        return 0.0f;
    }
}

__device__ float weight(const Eigen::Vector3f& xpgp_diff) {
    return NX(xpgp_diff(0)) * NX(xpgp_diff(1)) * NX(xpgp_diff(2));
}

__device__ Eigen::Vector3f gradientWeight(const Eigen::Vector3f& xpgp_diff) {
    const auto& v = xpgp_diff;
    return (1.0f / PARTICLE_DIAM) * Eigen::Vector3f(dNX(v(0)) * NX(fabs(v(1))) * NX(fabs(v(2))),
                                                    NX(fabs(v(0))) * dNX(v(1)) * NX(fabs(v(2))),
                                                    NX(fabs(v(0))) * NX(fabs(v(1))) * dNX(v(2)));
}

__device__ int getGridIndex(const Eigen::Vector3i& pos) {
    return (pos(2) * GRID_BOUND_Y * GRID_BOUND_X) + (pos(1) * GRID_BOUND_X) + pos(0);
}

__host__ void MPMSolver::transferData() {
    Grid *grid_ptr = thrust::raw_pointer_cast(&grids[0]);

    if (initial_transfer) {
        auto ff = [=] __device__ (Particle& p) {
            float h_inv = 1.0f / PARTICLE_DIAM;
            Eigen::Vector3i pos((p.position * h_inv).cast<int>());

            for (int z = -2; z <= 2; z++) {
                for (int y = -2; y <= 2; y++) {
                    for (int x = -2; x <= 2; x++) {
                        auto _pos = pos + Eigen::Vector3i(x, y, z);
                        Eigen::Vector3f diff = (p.position - (_pos.cast<float>() * PARTICLE_DIAM)) * h_inv;
                        int grid_idx = getGridIndex(_pos);
                        float mi = p.mass * weight(diff.cwiseAbs());
                        atomicAdd(&(grid_ptr[grid_idx].mass), mi);
                    }
                }
            }
        };
        thrust::for_each(thrust::device, particles.begin(), particles.end(), ff);
        initial_transfer = false;
    }
    else {
        auto ff = [=] __device__ (Particle& p) {
            float h_inv = 1.0f / PARTICLE_DIAM;
            Eigen::Vector3i pos((p.position * h_inv).cast<int>());

            for (int z = -2; z <= 2; z++) {
                for (int y = -2; y <= 2; y++) {
                    for (int x = -2; x <= 2; x++) {
                        auto _pos = pos + Eigen::Vector3i(x, y, z);
                        Eigen::Vector3f diff = (p.position - (_pos.cast<float>() * PARTICLE_DIAM)) * h_inv;
                        auto gw = gradientWeight(diff);
                        int grid_idx = getGridIndex(_pos);
                        Eigen::Vector3f f = -1.0f * p.energyDerivative() * gw;
                        float mi = p.mass * weight(diff.cwiseAbs());
                        Eigen::Vector3f vm = p.velocity * mi;
                        atomicAdd(&(grid_ptr[grid_idx].mass), mi);
                        atomicAdd(&(grid_ptr[grid_idx].velocity(0)), vm(0));
                        atomicAdd(&(grid_ptr[grid_idx].velocity(1)), vm(1));
                        atomicAdd(&(grid_ptr[grid_idx].velocity(2)), vm(2));
                        atomicAdd(&(grid_ptr[grid_idx].force(0)), f(0));
                        atomicAdd(&(grid_ptr[grid_idx].force(1)), f(1));
                        atomicAdd(&(grid_ptr[grid_idx].force(2)), f(2));
                    }
                }
            }
        };
        thrust::for_each(thrust::device, particles.begin(), particles.end(), ff);
    }
}

__host__ void MPMSolver::computeVolumes() {
    Grid *grid_ptr = thrust::raw_pointer_cast(&grids[0]);

    auto ff = [=] __device__ (Particle& p) {
        float h_inv = 1.0f / PARTICLE_DIAM;
        Eigen::Vector3i pos((p.position * h_inv).cast<int>());
        float p_density = 0.0f;
        float inv_grid_volume = h_inv * h_inv * h_inv;

        for (int z = -2; z <= 2; z++) {
            for (int y = -2; y <= 2; y++) {
                for (int x = -2; x <= 2; x++) {
                    auto _pos = pos + Eigen::Vector3i(x, y, z);
                    Eigen::Vector3f diff = (p.position - (_pos.cast<float>() * PARTICLE_DIAM)) * h_inv;
                    int grid_idx = getGridIndex(_pos);
                    p_density += grid_ptr[grid_idx].mass * inv_grid_volume * weight(diff.cwiseAbs());
                }
            }
        }

        p.volume = p.mass / p_density;
    };

    thrust::for_each(thrust::device, particles.begin(), particles.end(), ff);
}

__host__ void MPMSolver::updateVelocities() {
    thrust::for_each(
        thrust::device,
        grids.begin(),
        grids.end(),
        [=] __device__ (Grid& g) {
            g.updateVelocity();
        }
    );
}

__host__ void MPMSolver::bodyCollisions() {
    thrust::for_each(
        thrust::device,
        grids.begin(),
        grids.end(),
        [=] __device__ (Grid& g) {
            g.applyBoundaryCollision();
        }
    );
}

__host__ void MPMSolver::updateDeformationGradient() {
    Grid *grid_ptr = thrust::raw_pointer_cast(&grids[0]);

    auto computeVelocityGradient = [=] __device__ (const Particle& p) -> Eigen::Matrix3f {
        float h_inv = 1.0f / PARTICLE_DIAM;
        Eigen::Vector3i pos((p.position * h_inv).cast<int>());
        Eigen::Matrix3f velocity_gradient(Eigen::Matrix3f::Zero());

        for (int z = -2; z <= 2; z++) {
            for (int y = -2; y <= 2; y++) {
                for (int x = -2; x <= 2; x++) {
                    auto _pos = pos + Eigen::Vector3i(x, y, z);
                    Eigen::Vector3f diff = (p.position - (_pos.cast<float>() * PARTICLE_DIAM)) * h_inv;
                    Eigen::Vector3f gw = gradientWeight(diff);
                    int grid_idx = getGridIndex(_pos);

                    velocity_gradient += grid_ptr[grid_idx].velocity_star * gw.transpose();
                }
            }
        }

        return velocity_gradient;
    };

    thrust::for_each(
        thrust::device,
        particles.begin(),
        particles.end(),
        [=] __device__ (Particle& p) {
            auto velocity_gradient = computeVelocityGradient(p);
            p.updateDeformationGradient(velocity_gradient);
        }
    );
}

__host__ void MPMSolver::updateParticleVelocities() {
    Grid *grid_ptr = thrust::raw_pointer_cast(&grids[0]);

    auto computeVelocity = [=] __device__ (const Particle& p) -> thrust::pair<Eigen::Vector3f, Eigen::Vector3f> {
        float h_inv = 1.0f / PARTICLE_DIAM;
        Eigen::Vector3i pos((p.position * h_inv).cast<int>());

        Eigen::Vector3f velocity_pic(Eigen::Vector3f::Zero()),
                        velocity_flip(p.velocity);

        for (int z = -2; z <= 2; z++) {
            for (int y = -2; y <= 2; y++) {
                for (int x = -2; x <= 2; x++) {
                    auto _pos = pos + Eigen::Vector3i(x, y, z);
                    Eigen::Vector3f diff = (p.position - (_pos.cast<float>() * PARTICLE_DIAM)) * h_inv;
                    int grid_idx = getGridIndex(_pos);
                    float w = weight(diff.cwiseAbs());
                    auto grid = grid_ptr[grid_idx];
                    velocity_pic += grid.velocity_star * w;
                    velocity_flip = (grid.velocity_star - grid.velocity) * w;
                }
            }
        }

        return thrust::make_pair(velocity_pic, velocity_flip);
    };

    thrust::for_each(
        thrust::device,
        particles.begin(),
        particles.end(),
        [=] __device__ (Particle& p) {
            auto velocity_result = computeVelocity(p);
            p.updateVelocity(velocity_result.first, velocity_result.second);
        }
    );
}

__host__ void MPMSolver::particleBodyCollisions() {
    thrust::for_each(
        thrust::device,
        particles.begin(),
        particles.end(),
        [=] __device__ (Particle& p) {
            p.applyBoundaryCollision();
        }
    );
}

__host__ void MPMSolver::updateParticlePositions() {
    thrust::for_each(
        thrust::device,
        particles.begin(),
        particles.end(),
        [=] __device__ (Particle& p) {
            p.updatePosition();
        }
    );
}
