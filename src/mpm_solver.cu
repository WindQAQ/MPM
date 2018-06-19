#include <cassert>

#include <fstream>

#include <Eigen/Dense>

#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "mpm_solver.h"

#define IN_GRID(POS) (0 <= POS(0) && POS(0) < GRID_BOUND_X && \
                      0 <= POS(1) && POS(1) < GRID_BOUND_Y && \
                      0 <= POS(2) && POS(2) < GRID_BOUND_Z)

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
    /*
    return Eigen::Vector3f(dNX(v(0)) * NX(fabs(v(1))) * NX(fabs(v(2))),
                           NX(fabs(v(0))) * dNX(v(1)) * NX(fabs(v(2))),
                           NX(fabs(v(0))) * NX(fabs(v(1))) * dNX(v(2)));
    */
}

__device__ int getGridIndex(const Eigen::Vector3i& pos) {
    return (pos(2) * GRID_BOUND_Y * GRID_BOUND_X) + (pos(1) * GRID_BOUND_X) + pos(0);
}

__device__ Eigen::Vector3f applyBoundaryCollision(const Eigen::Vector3f& position, const Eigen::Vector3f& velocity) {
    float vn;
    Eigen::Vector3f vt, normal, ret(velocity);

    bool collision;

    for (int i = 0; i < 3; i++) {
        collision = false;
        normal.setZero();

        if (position(i) <= BOX_BOUNDARY_1) {
            collision = true;
            normal(i) = 1.0f;
        }
        else if (position(i) >= BOX_BOUNDARY_2) {
            collision = true;
            normal(i) = -1.0f;
        }

        if (collision) {
            vn = ret.dot(normal);

            if (vn >= 0.0f) continue;

            for (int j = 0; j < 3; j++) {
                if (j != i) {
                    ret(j) *= STICKY_WALL;
                }
            }

            vt = ret - vn * normal;

            if (vt.norm() <= -FRICTION * vn) {
                ret.setZero();
                return ret;
            }

            ret = vt + FRICTION * vn *  vt.normalized();
        }
    }

    return ret;
}

struct f {
    __host__ __device__
    Grid operator()(const int& idx) {
        return Grid(Eigen::Vector3i(idx % GRID_BOUND_X, idx % (GRID_BOUND_X * GRID_BOUND_Y) / GRID_BOUND_X, idx / (GRID_BOUND_X * GRID_BOUND_Y)));
    }
};

__host__ MPMSolver::MPMSolver(const std::vector<Particle>& _particles) {
    particles.resize(_particles.size());
    thrust::copy(_particles.begin(), _particles.end(), particles.begin());

    grids.resize(GRID_BOUND_X * GRID_BOUND_Y * GRID_BOUND_Z);
    thrust::tabulate(
        thrust::device,
        grids.begin(),
        grids.end(),
        f()
    );
}

__host__ MPMSolver::MPMSolver(const std::vector<Particle>& _particles, const std::vector<Grid>& _grids) {
    particles.resize(_particles.size());
    grids.resize(_grids.size());

    thrust::copy(_particles.begin(), _particles.end(), particles.begin());
    thrust::copy(_grids.begin(), _grids.end(), grids.begin());
}

__host__ void MPMSolver::initialTransfer() {
    Grid *grid_ptr = thrust::raw_pointer_cast(&grids[0]);

    auto ff = [=] __device__ (Particle& p) {
        float h_inv = 1.0f / PARTICLE_DIAM;
        Eigen::Vector3i pos((p.position * h_inv).cast<int>());

        for (int z = -G2P; z <= G2P; z++) {
            for (int y = -G2P; y <= G2P; y++) {
                for (int x = -G2P; x <= G2P; x++) {
                    auto _pos = pos + Eigen::Vector3i(x, y, z);
                    if (!IN_GRID(_pos)) continue;

                    Eigen::Vector3f diff = (p.position - (_pos.cast<float>() * PARTICLE_DIAM)) * h_inv;
                    int grid_idx = getGridIndex(_pos);
                    float mi = p.mass * weight(diff.cwiseAbs());
                    atomicAdd(&(grid_ptr[grid_idx].mass), mi);
                }
            }
        }
    };

    thrust::for_each(thrust::device, particles.begin(), particles.end(), ff);
}

__host__ void MPMSolver::resetGrid() {
    thrust::for_each(
        thrust::device,
        grids.begin(),
        grids.end(),
        [=] __device__ (Grid& g) {
            g.reset();
        }
    );
}

__host__ void MPMSolver::transferData() {
    Grid *grid_ptr = thrust::raw_pointer_cast(&grids[0]);

    auto ff = [=] __device__ (Particle& p) {
        float h_inv = 1.0f / PARTICLE_DIAM;
        Eigen::Vector3i pos((p.position * h_inv).cast<int>());
        Eigen::Matrix3f volume_stress = -1.0f * p.energyDerivative();

        for (int z = -G2P; z <= G2P; z++) {
            for (int y = -G2P; y <= G2P; y++) {
                for (int x = -G2P; x <= G2P; x++) {
                    auto _pos = pos + Eigen::Vector3i(x, y, z);
                    if (!IN_GRID(_pos)) continue;

                    Eigen::Vector3f diff = (p.position - (_pos.cast<float>() * PARTICLE_DIAM)) * h_inv;
                    auto gw = gradientWeight(diff);
                    int grid_idx = getGridIndex(_pos);

                    Eigen::Vector3f f = volume_stress * gw;

                    float mi = p.mass * weight(diff.cwiseAbs());
                    atomicAdd(&(grid_ptr[grid_idx].mass), mi);
                    atomicAdd(&(grid_ptr[grid_idx].velocity(0)), p.velocity(0) * mi);
                    atomicAdd(&(grid_ptr[grid_idx].velocity(1)), p.velocity(1) * mi);
                    atomicAdd(&(grid_ptr[grid_idx].velocity(2)), p.velocity(2) * mi);
                    atomicAdd(&(grid_ptr[grid_idx].force(0)), f(0));
                    atomicAdd(&(grid_ptr[grid_idx].force(1)), f(1));
                    atomicAdd(&(grid_ptr[grid_idx].force(2)), f(2));
                }
            }
        }
    };

    thrust::for_each(thrust::device, particles.begin(), particles.end(), ff);
}

__host__ void MPMSolver::computeVolumes() {
    Grid *grid_ptr = thrust::raw_pointer_cast(&grids[0]);

    auto ff = [=] __device__ (Particle& p) {
        float h_inv = 1.0f / PARTICLE_DIAM;
        Eigen::Vector3i pos((p.position * h_inv).cast<int>());
        float p_density = 0.0f;
        float inv_grid_volume = h_inv * h_inv * h_inv;

        for (int z = -G2P; z <= G2P; z++) {
            for (int y = -G2P; y <= G2P; y++) {
                for (int x = -G2P; x <= G2P; x++) {
                    auto _pos = pos + Eigen::Vector3i(x, y, z);
                    if (!IN_GRID(_pos)) continue;

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
            g.velocity_star = applyBoundaryCollision((g.idx.cast<float>() * PARTICLE_DIAM) + (TIMESTEP * g.velocity_star), g.velocity_star);
        }
    );
}

#if ENABLE_IMPLICIT
__host__ void MPMSolver::computeAr() {
    Grid *grid_ptr = thrust::raw_pointer_cast(&grids[0]);

    auto computeDeltaElastic = [=] __device__ (const Particle& p) -> Eigen::Matrix3f {
        float h_inv = 1.0f / PARTICLE_DIAM;
        Eigen::Vector3i pos((p.position * h_inv).cast<int>());
        Eigen::Matrix3f f(Eigen::Matrix3f::Zero());

        for (int z = -G2P; z <= G2P; z++) {
            for (int y = -G2P; y <= G2P; y++) {
                for (int x = -G2P; x <= G2P; x++) {
                    auto _pos = pos + Eigen::Vector3i(x, y, z);
                    if (!IN_GRID(_pos)) continue;

                    Eigen::Vector3f diff = (p.position - (_pos.cast<float>() * PARTICLE_DIAM)) * h_inv;
                    Eigen::Vector3f gw = gradientWeight(diff);
                    int grid_idx = getGridIndex(_pos);

                    f += TIMESTEP * grid_ptr[grid_idx].velocity_star * gw.transpose();
                }
            }
        }

        return f * p.def_elastic;
    };

    auto updateGridDeltaForce = [=] __device__ (const Particle& p, const Eigen::Matrix3f& delta_force) {
        float h_inv = 1.0f / PARTICLE_DIAM;
        Eigen::Vector3i pos((p.position * h_inv).cast<int>());

        for (int z = -G2P; z <= G2P; z++) {
            for (int y = -G2P; y <= G2P; y++) {
                for (int x = -G2P; x <= G2P; x++) {
                    auto _pos = pos + Eigen::Vector3i(x, y, z);
                    if (!IN_GRID(_pos)) continue;

                    Eigen::Vector3f diff = (p.position - (_pos.cast<float>() * PARTICLE_DIAM)) * h_inv;
                    Eigen::Vector3f gw = gradientWeight(diff);
                    int grid_idx = getGridIndex(_pos);

                    auto fw = delta_force * gw;

                    atomicAdd(&(grid_ptr[grid_idx].delta_force(0)), fw(0));
                    atomicAdd(&(grid_ptr[grid_idx].delta_force(1)), fw(1));
                    atomicAdd(&(grid_ptr[grid_idx].delta_force(2)), fw(2));
                }
            }
        }
    };

    thrust::for_each(
        thrust::device,
        particles.begin(),
        particles.end(),
        [=] __device__ (Particle& p) {
            auto delta_elastic = computeDeltaElastic(p);
            auto delta_force = p.computeDeltaForce(delta_elastic);
            updateGridDeltaForce(p, delta_force);
        }
    );

    thrust::for_each(
        thrust::device,
        grids.begin(),
        grids.end(),
        [=] __device__ (Grid& g) {
            g.ar = g.r - BETA * TIMESTEP * g.delta_force / g.mass;
        }
    );
}

__host__ void MPMSolver::integrateImplicit() {
    // http://alexey.stomakhin.com/research/siggraph2013_tech_report.pdf
    // https://nccastaff.bournemouth.ac.uk/jmacey/MastersProjects/MSc15/05Esther/thesisEMdeJong.pdf
    // initialize some variables

    thrust::for_each(
        thrust::device,
        grids.begin(),
        grids.end(),
        [=] __device__ (Grid& g) {
            g.v = g.velocity_star;
            g.r = g.v;
        }
    );

    computeAr();

    thrust::for_each(
        thrust::device,
        grids.begin(),
        grids.end(),
        [=] __device__ (Grid& g) {
            g.r = g.v - g.ar;
            g.p = g.r;
        }
    );

    computeAr();

    thrust::for_each(
        thrust::device,
        grids.begin(),
        grids.end(),
        [=] __device__ (Grid& g) {
            g.ap = g.ar;
        }
    );

    // run conjugate residual method
    for (int i = 0; i < SOLVE_MAX_ITERATIONS; i++) {
        thrust::for_each(
            thrust::device,
            grids.begin(),
            grids.end(),
            [=] __device__ (Grid& g) {
                float alpha = (g.r).cwiseProduct(g.ar).sum() / (g.ap).cwiseProduct(g.ap).sum();
                g.v += alpha * g.p;
                g.rar_tmp = (g.r).cwiseProduct(g.ar).sum();
                g.r += (-alpha * g.ap);
            }
        );

        computeAr();

        thrust::for_each(
            thrust::device,
            grids.begin(),
            grids.end(),
            [=] __device__ (Grid& g) {
                float beta = (g.r).cwiseProduct(g.ar).sum() / g.rar_tmp;
                g.p = g.r + beta * g.p;
                g.ap = g.ar + beta * g.ap;
            }
        );
    }
}
#endif

__host__ void MPMSolver::updateDeformationGradient() {
    Grid *grid_ptr = thrust::raw_pointer_cast(&grids[0]);

    auto computeVelocityGradient = [=] __device__ (const Particle& p) -> Eigen::Matrix3f {
        float h_inv = 1.0f / PARTICLE_DIAM;
        Eigen::Vector3i pos((p.position * h_inv).cast<int>());
        Eigen::Matrix3f velocity_gradient(Eigen::Matrix3f::Zero());

        for (int z = -G2P; z <= G2P; z++) {
            for (int y = -G2P; y <= G2P; y++) {
                for (int x = -G2P; x <= G2P; x++) {
                    auto _pos = pos + Eigen::Vector3i(x, y, z);
                    if (!IN_GRID(_pos)) continue;

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

        for (int z = -G2P; z <= G2P; z++) {
            for (int y = -G2P; y <= G2P; y++) {
                for (int x = -G2P; x <= G2P; x++) {
                    auto _pos = pos + Eigen::Vector3i(x, y, z);
                    if (!IN_GRID(_pos)) continue;

                    Eigen::Vector3f diff = (p.position - (_pos.cast<float>() * PARTICLE_DIAM)) * h_inv;
                    int grid_idx = getGridIndex(_pos);
                    float w = weight(diff.cwiseAbs());
                    auto grid = grid_ptr[grid_idx];
                    velocity_pic += grid.velocity_star * w;
                    velocity_flip += (grid.velocity_star - grid.velocity) * w;
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
            p.velocity = applyBoundaryCollision(p.position + TIMESTEP * p.velocity, p.velocity);
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

__host__ void MPMSolver::simulate() {
    resetGrid();
    if (initial_transfer) {
        initialTransfer();
        computeVolumes();
        initial_transfer = false;
    }
    else {
        transferData();
    }
    updateVelocities();
    bodyCollisions();
#if ENABLE_IMPLICIT
    integrateImplicit();
#endif
    updateDeformationGradient();
    updateParticleVelocities();
    particleBodyCollisions();
    updateParticlePositions();
}

__host__ void MPMSolver::bindGLBuffer(const GLuint buffer) {
    cudaError_t ret;
    ret = cudaGraphicsGLRegisterBuffer(&vbo_resource, buffer, cudaGraphicsMapFlagsWriteDiscard);
    assert(ret == cudaSuccess);
}

__host__ void MPMSolver::writeGLBuffer() {
    cudaError_t ret;
    float4 *bufptr;
    size_t size;

    ret = cudaGraphicsMapResources(1, &vbo_resource, NULL);
    assert(ret == cudaSuccess);
    ret = cudaGraphicsResourceGetMappedPointer((void **)&bufptr, &size, vbo_resource);
    assert(ret == cudaSuccess);

    assert(bufptr != nullptr && size >= particles.size() * sizeof(float4));
    thrust::transform(
        thrust::device,
        particles.begin(),
        particles.end(),
        bufptr,
        [=] __device__ (Particle& p) -> float4 {
            return make_float4(5.0 * p.position(0) - 2.5, 5.0 * p.position(1), 5.0 * p.position(2) - 2.5, 1.0);
        }
    );

    ret = cudaGraphicsUnmapResources(1, &vbo_resource, NULL);
    assert(ret == cudaSuccess);
}

__host__ void MPMSolver::writeToFile(const std::string& filename) {
    std::ofstream output(filename, std::ios::binary | std::ios::out);
    int num_particles = particles.size();
    float min_bound_x = 0, max_bound_x = GRID_BOUND_X;
    float min_bound_y = 0, max_bound_y = GRID_BOUND_Y;
    float min_bound_z = 0, max_bound_z = GRID_BOUND_Z;

    output.write(reinterpret_cast<char *>(&num_particles), sizeof(int));
    output.write(reinterpret_cast<char *>(&min_bound_x), sizeof(float));
    output.write(reinterpret_cast<char *>(&max_bound_x), sizeof(float));
    output.write(reinterpret_cast<char *>(&min_bound_y), sizeof(float));
    output.write(reinterpret_cast<char *>(&max_bound_y), sizeof(float));
    output.write(reinterpret_cast<char *>(&min_bound_z), sizeof(float));
    output.write(reinterpret_cast<char *>(&max_bound_z), sizeof(float));

    thrust::copy(
        particles.begin(),
        particles.end(),
        std::ostream_iterator<Particle>(output)
    );

    output.close();
}
