#include "particle.h"
#include "constant.h"
#include "linalg.h"

__host__ __device__ Particle::Particle(const Eigen::Vector3f& _position, const Eigen::Vector3f& _velocity, float _mass,
                                       float _hardening, float young, float poisson, float _compression, float _stretch)
    : position(_position), velocity(_velocity), mass(_mass),
      hardening(_hardening),
      compression(_compression), stretch(_stretch) {

    lambda = (poisson * young) / ((1.0f + poisson) * (1.0f - 2.0f * poisson));
    mu = young / (2.0f * (1.0f + poisson));

    def_elastic.setIdentity();
    def_plastic.setIdentity();
#if ENABLE_IMPLICIT
    polar_r.setIdentity();
    polar_s.setIdentity();
#endif
}

__host__ std::ostream& operator<<(std::ostream& os, const Particle& p) {
    unsigned short x = p.position(0) * 65535.0f / (GRID_BOUND_X * PARTICLE_DIAM),
                   y = p.position(1) * 65535.0f / (GRID_BOUND_Y * PARTICLE_DIAM),
                   z = p.position(2) * 65535.0f / (GRID_BOUND_Z * PARTICLE_DIAM);

    os.write(reinterpret_cast<char *>(&x), sizeof(unsigned short));
    os.write(reinterpret_cast<char *>(&y), sizeof(unsigned short));
    os.write(reinterpret_cast<char *>(&z), sizeof(unsigned short));

    return os;
}

__host__ __device__ void Particle::updatePosition() {
    position += TIMESTEP * velocity;
}

__host__ __device__ void Particle::updateVelocity(const Eigen::Vector3f& velocity_pic, const Eigen::Vector3f& velocity_flip) {
    velocity = (1 - ALPHA) * velocity_pic + ALPHA * velocity_flip;
}

__host__ __device__ void Particle::updateDeformationGradient(const Eigen::Matrix3f& velocity_gradient) {
    def_elastic = (Eigen::Matrix3f::Identity() + (TIMESTEP * velocity_gradient)) * def_elastic;

    Eigen::Matrix3f force_all(def_elastic * def_plastic);

    Eigen::Matrix3f u, s, v;

    linalg::svd3(def_elastic, u, s, v);

    // clip values
    auto e = s.diagonal().array();
    e = e.min(1 + stretch).max(1 - compression);

#if ENABLE_IMPLICIT
    polar_r = u * v.transpose();
    polar_s = v;
    polar_s.array().rowwise() *= e.transpose();
    polar_s = polar_s * v.transpose();
#endif

    Eigen::Matrix3f u_tmp(u), v_tmp(v);
    u_tmp.array().rowwise() *= e.transpose();
    v_tmp.array().rowwise() /= e.transpose();

    def_plastic = v_tmp * u.transpose() * force_all;
    def_elastic = u_tmp * v.transpose();
}

__host__ __device__ const thrust::pair<float, float> Particle::computeHardening() const {
    float factor = expf(hardening * (1 - linalg::determinant(def_plastic)));
    return thrust::make_pair(mu * factor, lambda * factor);
}

__host__ __device__ const Eigen::Matrix3f Particle::energyDerivative() const {
    Eigen::Matrix3f u, s, v;

    linalg::svd3(def_elastic, u, s, v);

    float _mu, _lambda;
    thrust::tie(_mu, _lambda) = computeHardening();
    float je = linalg::determinant(def_elastic);

    Eigen::Matrix3f tmp(2.0f * _mu * (def_elastic - u * v.transpose()) * def_elastic.transpose());

    tmp.diagonal().array() += (_lambda * je * (je - 1));

    return volume * tmp;
}

#if ENABLE_IMPLICIT
__host__ __device__ const Eigen::Matrix3f Particle::computeDeltaForce(const Eigen::Matrix3f& delta_elastic) const {
    // See
    // http://alexey.stomakhin.com/research/siggraph2013_tech_report.pdf
    // https://nccastaff.bournemouth.ac.uk/jmacey/MastersProjects/MSc15/05Esther/thesisEMdeJong.pdf
    // for more details (difficult math)

    auto computeDeltaR = [=] __host__ __device__ () -> Eigen::Matrix3f {
        auto& r = polar_r;
        auto& s = polar_s;
        Eigen::Matrix3f v = r.transpose() * delta_elastic - delta_elastic.transpose() * r;

        Eigen::Matrix3f a;
        a <<  s(0) + s(4),  s(5), -s(2),
              s(5),  s(0) + s(8), s(1),
             -s(2),  s(1),  s(4) + s(8);

        Eigen::Vector3f b(v(3), v(6), v(7));

        auto x = linalg::solve(a, b);

        Eigen::Matrix3f r_trans_delta_r;
        r_trans_delta_r <<  0.0f,  x(0), x(1),
                           -x(0),  0.0f, x(2),
                           -x(1), -x(2), 0.0f;

        return r * r_trans_delta_r;
    };

    auto computeDeltaJeFeInvTrans = [=] __host__ __device__ () -> Eigen::Matrix3f {
        auto& f = def_elastic;
        auto& d = delta_elastic;

        Eigen::Matrix3f ret;
        ret(0) = f(4) * d(8) - f(5) * d(7) - f(7) * d(5) + f(8) * d(4),
        ret(1) = f(5) * d(6) - f(3) * d(8) + f(6) * d(5) - f(8) * d(3),
        ret(2) = f(3) * d(7) - f(4) * d(6) - f(6) * d(4) + f(7) * d(3),
        ret(3) = f(2) * d(7) - f(1) * d(8) + f(7) * d(2) - f(8) * d(1),
        ret(4) = f(0) * d(8) - f(2) * d(6) - f(6) * d(2) + f(8) * d(0),
        ret(5) = f(1) * d(6) - f(0) * d(7) + f(6) * d(1) - f(7) * d(0),
        ret(6) = f(1) * d(5) - f(2) * d(4) - f(4) * d(2) + f(5) * d(1),
        ret(7) = f(2) * d(3) - f(0) * d(5) + f(3) * d(2) - f(5) * d(0),
        ret(8) = f(0) * d(4) - f(1) * d(3) - f(3) * d(1) + f(4) * d(0);

        return ret;
    };

    float _mu, _lambda;
    thrust::tie(_mu, _lambda) = computeHardening();

    auto je = linalg::determinant(def_elastic);
    auto je_fe_inv_trans = linalg::cofactor(def_elastic);
    auto delta_r = computeDeltaR();
    auto Ap = 2 * _mu * (delta_elastic - delta_r)
        + _lambda * je_fe_inv_trans * (je_fe_inv_trans.cwiseProduct(delta_elastic)).sum()
        + _lambda * (je - 1) * computeDeltaJeFeInvTrans();

    return -volume * Ap * def_elastic.transpose();
}
#endif
