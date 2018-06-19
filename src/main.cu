#include <iostream>
#include <cstdio>
#include <vector>
#include <Eigen/Dense>
#include <thrust/device_new.h>

#include "grid.h"
#include "constant.h"
#include "particle.h"
#include "mpm_solver.h"
#include "point_loader.h"
#include "parser.h"

#include <glad/glad.h>

#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "renderer.h"

const unsigned int WIDTH = 800;
const unsigned int HEIGHT = 600;

void errorCallback(int error, const char* description) {
    std::cerr << "Errors: " << description << std::endl;
}

void processInput(GLFWwindow *window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

int main(int argc, const char *argv[]) {

    auto vm = parser::parseArgs(argc, argv);

    std::vector<Particle> particles;

    if (vm.count("model")) {
        auto model_vec = parser::parseModel(vm["model"].as<std::string>());

        for (const auto& config: model_vec) {
            auto model = PointLoader(config.path, config.translate * PARTICLE_DIAM, config.scale);
            for (const auto& pos: model.positions) {
                particles.push_back(
                    Particle(
                        pos,
                        config.velocity,
                        config.mass,
                        config.hardening,
                        config.young,
                        config.poisson,
                        config.compression,
                        config.stretch
                    )
                );
            }
        }
    }

    /*
    {
        // two balls
        // TIMESTEP 1e-4
        // HARDENING 10.0f
        // CRIT_COMPRESS 1.9e-2
        // CRIT_STRETCH 7.5e-3
        // ALPHA 0.95f
        // PATICLE_DIAM 0.010
        // STICKY_WALL 0.9
        // FRICTION 1.0
        // DENSITY 400
        // YOUNG 1.4e5
        // POSSION 0.2
        const int height = 70;
        Eigen::Vector3i center(70, height, 80);
        createSphere(particles, center, 20, Eigen::Vector3f(0.0f, 0.0f, -3.0f), mass, lambda, mu, 4);
        center(2) = 30;
        createSphere(particles, center, 7, Eigen::Vector3f(0.0f, 0.0f, 15.0f), mass, lambda, mu, 50);
    }
    */

    std::cout << "number of particles: " << particles.size() << ", number of bytes in particles: " << particles.size() * sizeof(Particle) << std::endl;

    MPMSolver mpm_solver(particles);

    auto ret = cudaGetLastError();
    assert(ret == cudaSuccess);

    ret = cudaGetLastError();
    assert(ret == cudaSuccess);

    // glfw: initialize and configure
    if (!glfwInit()) return EXIT_FAILURE;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // glfw window creation
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Sand-MPM", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return EXIT_FAILURE;
    }
    glfwMakeContextCurrent(window);

    // glfw setting callback
    glfwSetErrorCallback(errorCallback);

    // glad: load all OpenGL function pointers
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return EXIT_FAILURE;
    }

    std::cerr << "OpenGL version: " << glGetString(GL_VERSION) << std::endl;
    glEnable(GL_DEPTH_TEST);

    ret = cudaGetLastError();
    assert(ret == cudaSuccess);

    Renderer renderer(WIDTH, HEIGHT, particles.size());
    ret = cudaGetLastError();
    assert(ret == cudaSuccess);

    mpm_solver.bindGLBuffer(renderer.getSnowVBO());

    // render loop
    int step = 0;
    while (!glfwWindowShouldClose(window)) {
        processInput(window);
        if (glfwGetKey(window, GLFW_KEY_O) == GLFW_PRESS)
            renderer.setOrigin();
        if (glfwGetKey(window, GLFW_KEY_U) == GLFW_PRESS)
            renderer.setUp();
        if (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS)
            renderer.setFront();
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            renderer.setSide();

        std::cout << "step: " << step << std::endl;

        if (vm["save"].as<bool>()) {
            char pnt_fname[128];
            sprintf(pnt_fname, "points_%05d.dat", step);
            mpm_solver.writeToFile(pnt_fname);
        }
        step++;

        mpm_solver.simulate();
        mpm_solver.writeGLBuffer();
        renderer.render();

        // glfw: swap buffers and poll IO events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
