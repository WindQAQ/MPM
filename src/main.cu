#include <iostream>
#include <cstdio>
#include <vector>
#include <Eigen/Dense>
#include <thrust/copy.h>
#include <thrust/for_each.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include "grid.h"
#include "constant.h"
#include "particle.h"
#include "mpm_solver.h"

int main() {
    std::vector<Grid> grids;
    std::vector<Particle> particles;

    // reserve space
    grids.reserve(GRID_BOUND_X * GRID_BOUND_Y * GRID_BOUND_Z);
    particles.reserve(20 * 20 * 21);

    // generate grids
    for (int x = 0; x < GRID_BOUND_X; x++)
        for (int y = 0; y < GRID_BOUND_Y; y++)
            for (int z = 0; z < GRID_BOUND_Z; z++) {
                grids.push_back(Grid(Eigen::Vector3i(x, y, z)));
            }

    // create a cube of particles
    float ptcl_lambda = LAMBDA,
          ptcl_mu = MU;
    for (int i = 0; i < 20; i++) {
        float pi = PARTICLE_DIAM * (40 + i);
        for (int j = 0; j < 20; j++) {
            float pj = PARTICLE_DIAM * (40 + j);
            for (int k = 0; k < 20; k++) {
                float pk = PARTICLE_DIAM * (40 + k);
                particles.push_back(
                    Particle(
                        Eigen::Vector3f(pi, pj, pk),
                        Eigen::Vector3f(0.0f, 0.0f, 0.0f),
                        1.0f,
                        ptcl_lambda,
                        ptcl_mu
                    )
                );
            }
        }
    }

    MPMSolver mpm_solver(particles, grids);

    // cudaDeviceSynchronize();

    return 0;
}
