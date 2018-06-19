#ifndef PROGRAM_OPTIONS_H_
#define PROGRAM_OPTIONS_H_

#include <boost/program_options.hpp>

#include <Eigen/Dense>

namespace parser {

    struct ModelConfig {
        std::string path;
        Eigen::Vector3f translate, velocity;
        float scale, mass, hardening, young, poisson, compression, stretch;

        ModelConfig(const std::string& _path, const Eigen::Vector3f _translate, float _scale,
                    float _mass, const Eigen::Vector3f& _velocity,
                    float _hardening, float _young, float _poisson,
                    float _compression, float _stretch)
            : path(_path), translate(_translate), scale(_scale),
              mass(_mass), velocity(_velocity),
              hardening(_hardening), young(_young), poisson(_poisson),
              compression(_compression), stretch(_stretch)
        {}
    };

    boost::program_options::variables_map parseArgs(const int argc, const char *[]);
    std::vector<ModelConfig> parseModel(const std::string&);
} // parser

#endif // PROGRAM_OPTIONS_H_
