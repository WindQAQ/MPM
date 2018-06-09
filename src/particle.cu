#include "particle.h"
#include "constant.h"
#include "linalg.h"

__host__ __device__ void Particle::updatePosition() {
    position += TIMESTEP * velocity;
}

__host__ __device__ void Particle::updateGradient() {
    velocity_gradient *= TIMESTEP;
    velocity_gradient.diagonal().array() += 1.0f;
    def_elastic = velocity_gradient * def_elastic;
}

__host__ __device__ void Particle::applyPlasticity() {
    Eigen::Matrix3f force_all(def_elastic * def_plastic);

    auto& u = svd_u;
    auto& s = svd_s;
    auto& v = svd_v;
    auto e = s.diagonal().array();

    linalg::svd3(def_elastic, u, s, v);

    // clip values
    e = e.min(CRIT_STRETCH).max(CRIT_COMPRESS);

#if ENABLE_IMPLICIT
    polar_r = u * v.transpose();
    polar_s = v;
    polar_s.diagonal().array() *= s.diagonal().array();
    polar_s = polar_s * v.transpose();
#endif

    Eigen::Matrix3f u_tmp(u), v_tmp(v);
    u_tmp.diagonal().array() *= e;
    v_tmp.diagonal().array() /= e;

    def_plastic = v_tmp * u.transpose() * force_all;
    def_elastic = u_tmp * v.transpose();
}

__host__ __device__ const Eigen::Matrix3f Particle::energyDerivative() {
    auto& u = svd_u;
    auto& s = svd_s;
    auto& v = svd_v;

    float harden = HARDENING * expf(1 - linalg::determinant(def_plastic)),
                   je = s.diagonal().prod();

    Eigen::Matrix3f tmp(2.0f * mu * (def_elastic - u * v.transpose()) * def_elastic.transpose());

    tmp.diagonal().array() += (lambda * je * (je - 1));

    return volume * harden * tmp;
}

#if ENABLE_IMPLICT
__host__ __device__ Eigen::Vector3f Particle::deltaForce(const Eigen::Vector3f& u, const Eigen::Vector3f& wg) {
    Matrix3f delta_elastic = TIMESTEP * (u * v.transpose()) * def_elastic;

    // TODO: wtf is the implicit math pdf???
}
#endif
