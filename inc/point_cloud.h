#ifndef POINT_CLOUD_H_
#define POINT_CLOUD_H_

#include <vector>
#include <thrust/device_vector.h>
#include <Eigen/Dense>

#include "particle.h"

class PointCloud {
  public:
    int size;
    float max_velocity;
    std::vector<Particle> particles;
    thrust::device_vector<Particle> dv_particles;

    PointCloud();
    PointCloud(int);
    PointCloud(const PointCloud&);
    virtual ~PointCloud();

    void scale(const Eigen::Vector3f&, const Eigen::Vector3f&);
    void translate(const Eigen::Vector3f&);
    void update();
    void merge(const PointCloud&);
    void bounds(Eigen::Vector3f&, Eigen::Vector3f&);
};

#endif  // POINT_CLOUD_H_
