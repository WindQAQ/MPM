#ifndef PARTICLE_H_
#define PARTICLE_H_

#include <Eigen/Dense>

#include <thrust/pair.h>
#include <thrust/tuple.h>

#include "constant.h"

class Particle {
  public:
    float volume, mass;
    Eigen::Vector3f position, velocity;

    float hardening;
    float lambda, mu;
    float compression, stretch;

    Eigen::Matrix3f def_elastic, def_plastic;

#if ENABLE_IMPLICIT
    Eigen::Matrix3f polar_r, polar_s;
#endif

    __host__ __device__ Particle() {}
    __host__ __device__ Particle(const Eigen::Vector3f&, const Eigen::Vector3f&, float, float, float, float, float, float);

    __host__ __device__ virtual ~Particle() {}

    __host__ friend std::ostream& operator<<(std::ostream&, const Particle&);

    __host__ __device__ void updatePosition();
    __host__ __device__ void updateVelocity(const Eigen::Vector3f&, const Eigen::Vector3f&);
    __host__ __device__ void updateDeformationGradient(const Eigen::Matrix3f&);
    __host__ __device__ void applyBoundaryCollision();
    __host__ __device__ const Eigen::Matrix3f energyDerivative() const;
#if ENABLE_IMPLICIT
    __host__ __device__ const Eigen::Matrix3f computeDeltaForce(const Eigen::Matrix3f&) const;
#endif

  private:
    __host__ __device__ const thrust::pair<float, float> computeHardening() const;
};

#endif  // PARTICLE_H_
