#ifndef POINT_LOADER_H_
#define POINT_LOADER_H_

#include <cassert>
#include <fstream>
#include <sstream>

#include <Eigen/Dense>

class PointLoader {
  public:
    std::vector<Eigen::Vector3f> positions;
    PointLoader(const std::string& fname) : PointLoader(fname, Eigen::Vector3f(0.0f, 0.0f, 0.0f), 1.0f) {}
    PointLoader(const std::string&, const Eigen::Vector3f&, const float);
};

#endif  // POINT_LOADER_H_
