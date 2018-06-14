#ifndef MPM_SOLVER_H_
#define MPM_SOLVER_H_

#include <vector>
#include <glad/glad.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <cuda_gl_interop.h>

#include "grid.h"
#include "constant.h"
#include "particle.h"

class MPMSolver {
  public:
    __host__ MPMSolver(const std::vector<Particle>& _particles, const std::vector<Grid>& _grids) {
        particles.resize(_particles.size());
        grids.resize(_grids.size());

        thrust::copy(_particles.begin(), _particles.end(), particles.begin());
        thrust::copy(_grids.begin(), _grids.end(), grids.begin());
    }
    __host__ ~MPMSolver() {}

    __host__ void resetGrid();
    __host__ void initialTransfer();
    __host__ void transferData();
    __host__ void computeVolumes();
    __host__ void updateVelocities();
    __host__ void bodyCollisions();
    __host__ void updateDeformationGradient();
    __host__ void updateParticleVelocities();
    __host__ void particleBodyCollisions();
    __host__ void updateParticlePositions();

    __host__ void simulate();
    __host__ void bindGLBuffer(const GLuint);
    __host__ void writeGLBuffer();

  private:
    thrust::device_vector<Particle> particles;
    thrust::device_vector<Grid> grids;
    struct cudaGraphicsResource *vbo_resource;
    bool initial_transfer = true;
};

#endif  // MPM_SOLVER_H_
