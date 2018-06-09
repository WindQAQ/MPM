#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>

#include "point_cloud.h"

PointCloud::PointCloud() {
    size = 0;
    max_velocity = 0.0;
}

PointCloud::PointCloud(const std::vector<Particle>& vec) {
    size = vec.size();
    particles = vec;
}

PointCloud::PointCloud(const PointCloud& orig) {}

PointCloud::~PointCloud() {}

void PointCloud::scale(const Eigen::Vector3f& origin, const float scale) {
    thrust::for_each(
        thrust::device,
        particles.begin(),
        particles.end(),
        [=] __device__ (Particle& p) {
            p.position = origin + (p.position - origin) * scale;
        }
    );
}

void PointCloud::translate(const Eigen::Vector3f& offset) {
    thrust::for_each(
        thrust::device,
        particles.begin(),
        particles.end(),
        [=] __device__ (Particle& p) {
            p.position = p.position + offset;
        }
    );
}

void PointCloud::update() {
    thrust::for_each(
        thrust::device,
        particles.begin(),
        particles.end(),
        [=] __device__ (Particle& p) {
            p.updatePosition();
            p.updateGradient();
            p.applyPlasticity();
        }
    );
    max_velocity = thrust::transform_reduce(
        thrust::device,
        particles.begin(),
        particles.end(),
        [=] __device__ (const Particle& p) -> float { return p.velocity.squaredNorm(); },
        0,
        thrust::maximum<float>()
    );
}

void PointCloud::merge(const PointCloud& other) {
    size += other.size;
    particles.reserve(size);
    particles.insert(particles.end(), other.particles.begin(), other.particles.end());
}

void PointCloud::bounds(Eigen::Vector3f& minAxis, Eigen::Vector3f& maxAxis) {
    // TODO
}
