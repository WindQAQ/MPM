#include "point_loader.h"

PointLoader::PointLoader(const std::string &fname,
                         const Eigen::Vector3f& translate,
                         const float scale) {
    std::ifstream pntfile;

    pntfile.open(fname);
    assert(pntfile.is_open());

    std::string line;
    float x, y, z;
    Eigen::Vector3f minp(1e9, 1e9, 1e9);
    while (getline(pntfile, line)) {
        std::istringstream iss(line);
        iss >> x >> y >> z;
        minp = minp.cwiseMin(Eigen::Vector3f(x, y, z));
        positions.push_back(Eigen::Vector3f(x, y, z));
    }

    std::for_each(positions.begin(), positions.end(), [=](Eigen::Vector3f &pos) {
        pos -= minp;
        pos *= scale;
        pos += translate;
    });

    pntfile.close();
}
