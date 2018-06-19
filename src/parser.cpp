#include <cstdlib>

#include <iostream>
#include <fstream>

#include <boost/foreach.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <Eigen/Dense>

#include "parser.h"

namespace boost {
    void validate(boost::any& v, const std::vector<std::string>& values, std::vector<float>*, int) {
        std::vector<float> vec;
        for(const auto& val : values)  {
            std::stringstream ss(val);
            std::copy(std::istream_iterator<float>(ss), std::istream_iterator<float>(), std::back_inserter(vec));
        }
        v = vec;
    }

    void validate(boost::any& v, const std::vector<std::string>& values, Eigen::Vector3f*, int) {
        Eigen::Vector3f c;
        std::vector<float> fvalues;
        for(const auto& val : values)  {
            std::stringstream ss(val);
            std::copy(std::istream_iterator<float>(ss), std::istream_iterator<float>(),
            std::back_inserter(fvalues));
        }
        if(fvalues.size() != 3) {
            throw boost::program_options::invalid_option_value("Invalid coordinate specification");
        }
        c(0) = fvalues[0];
        c(1) = fvalues[1];
        c(2) = fvalues[2];
        v = c;
    }
} // boost

namespace parser {
    namespace po = boost::program_options;
    namespace pt = boost::property_tree;

    po::variables_map parseArgs(const int argc, const char *argv[]) {
        po::options_description desc("Material Point Method Simulation");

        desc.add_options()
            ("help,h", "produce help message")
            ("save,s", po::bool_switch()->default_value(false), "save frame")
            ("config", po::value<std::string>()->required(), "configuration file")
        ;

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);

        if (vm.count("help")) {
            std::cout << desc << std::endl;
            exit(0);
        }

        if (vm.count("save")) {
            std::string save = vm["save"].as<bool>()? "On": "Off";
            std::cout << "Save frame: " << save << std::endl;
        }

        if (vm.count("config")) {
            std::cout << "Configuration file: " << vm["config"].as<std::string>() << std::endl;

            po::options_description config("Model Config");

            config.add_options()
                ("timestep,t", po::value<float>()->default_value(1e-4), "simulation timestep")
                ("gravity,g", po::value<Eigen::Vector3f>()->multitoken()->default_value(Eigen::Vector3f(0.0f, 9.8f, 0.0f)), "gravity")
                ("alpha,a", po::value<float>(), "interpolation coefficient of PIC and FLIP")
                ("model,m", po::value<std::string>()->required(), "model file")
            ;

            std::ifstream fs(vm["config"].as<std::string>());
            po::store(po::parse_config_file(fs, config), vm);
            po::notify(vm);

            fs.close();
        }

        return vm;
    }

    std::vector<ModelConfig> parseModel(const std::string& filename) {
        pt::ptree model_tree;

        std::cout << "Parsing " << filename << std::endl;

        pt::read_json(filename, model_tree);

        auto getVector3f = [] (const pt::ptree& pt, const pt::ptree::key_type& key) -> Eigen::Vector3f {
            std::vector<float> vec;
            BOOST_FOREACH (const auto& v, pt.get_child(key)) vec.push_back(v.second.get_value<float>());
            assert (vec.size() == 3);
            return Eigen::Vector3f(vec[0], vec[1], vec[2]);
        };

        std::vector<ModelConfig> vec;

        BOOST_FOREACH (const auto& model, model_tree) {
            std::cout << "\tModel name: " << model.first << std::endl;
            const auto& prop = model.second;
            auto path = prop.get<std::string>("path");
            auto translate = getVector3f(prop, "translate");
            auto scale = prop.get<float>("scale", 1.0f);
            auto mass = prop.get<float>("mass", 0.0002f);
            auto velocity = getVector3f(prop, "velocity");
            auto hardening = prop.get<float>("hardening", 10.0f);
            auto young = prop.get<float>("young", 1.4e5);
            auto poisson = prop.get<float>("poisson", 0.2);
            auto compression = prop.get<float>("compression", 5.0e-2);
            auto stretch = prop.get<float>("stretch", 2.5e-3);

            auto model_config = ModelConfig(path, translate, scale,
                                            mass, velocity,
                                            hardening, young, poisson,
                                            compression, stretch);

            vec.push_back(model_config);
        }

        return vec;
    }
} // parser
