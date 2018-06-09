#ifndef POINT_CLOUD_H_
#define POINT_CLOUD_H_

#include <vector>
#include <Eigen/Dense>
#include <thrust/device_vector.h>

#include "particle.h"

class PointCloud {
  public:
    int size;
    float max_velocity;
    thrust::device_vector<Particle> particles;

    PointCloud();
    PointCloud(const std::vector<Particle>&);
    PointCloud(const PointCloud&);
    virtual ~PointCloud();

    void scale(const Eigen::Vector3f&, const float);
    void translate(const Eigen::Vector3f&);
    void update();
    void merge(const PointCloud&);
    void bounds(Eigen::Vector3f&, Eigen::Vector3f&);
};

#endif  // POINT_CLOUD_H_
