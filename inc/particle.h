#ifndef PARTICLE_H_
#define PARTICLE_H_

#include <Eigen/Dense>

class Particle {
  public:
    float volume, mass, density;
    Eigen::Vector3f position, velocity;

    Eigen::Matrix3f velocity_gradient;

    float lambda, mu;

    Eigen::Matrix3f def_elastic, def_plastic;

    Eigen::Matrix3f svd_u, svd_s, svd_v;

    Eigen::Matrix3f polar_r, polar_s;

    Eigen::Vector3f grid_position;
    Eigen::Vector3f weight_gradient[16];
    float weight[16];

    __host__ __device__ Particle() {}
    __host__ __device__ Particle(const Eigen::Vector3f& _position, const Eigen::Vector3f& _velocity, float _mass, float _lambda, float _mu)
        : position(_position), velocity(_velocity), mass(_mass), lambda(_lambda), mu(_mu),
          def_elastic(Eigen::Matrix3f::Identity()), def_plastic(Eigen::Matrix3f::Identity()),
          svd_u(Eigen::Matrix3f::Identity()), svd_s(Eigen::Matrix3f::Identity()), svd_v(Eigen::Matrix3f::Identity()),
          polar_r(Eigen::Matrix3f::Identity()), polar_s(Eigen::Matrix3f::Identity())
    {}

    __host__ __device__ ~Particle() {}

    __host__ __device__ void updatePosition();
    __host__ __device__ void updateGradient();

    __host__ __device__ void applyPlasticity();

    __host__ __device__ const Eigen::Matrix3f energyDerivative();

    __host__ __device__ const Eigen::Vector3f deltaForce(const Eigen::Vector3f&, const Eigen::Vector3f&);
};

#endif  // PARTICLE_H_
