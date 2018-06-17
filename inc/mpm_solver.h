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
    __host__ MPMSolver(const std::vector<Particle>&);
    __host__ MPMSolver(const std::vector<Particle>&, const std::vector<Grid>&);
    __host__ ~MPMSolver() {}

    __host__ void resetGrid();
    __host__ void initialTransfer();
    __host__ void transferData();
    __host__ void computeVolumes();
    __host__ void updateVelocities();
    __host__ void bodyCollisions();
#if ENABLE_IMPLICIT
    __host__ void computeAr();
    __host__ void integrateImplicit();
#endif
    __host__ void updateDeformationGradient();
    __host__ void updateParticleVelocities();
    __host__ void particleBodyCollisions();
    __host__ void updateParticlePositions();

    __host__ void simulate();
    __host__ void bindGLBuffer(const GLuint);
    __host__ void writeGLBuffer();
    __host__ void writeToFile(const std::string&);

  private:
    thrust::device_vector<Particle> particles;
    thrust::device_vector<Grid> grids;
    struct cudaGraphicsResource *vbo_resource;
    bool initial_transfer = true;
};

#endif  // MPM_SOLVER_H_
