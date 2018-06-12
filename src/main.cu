#include <iostream>
#include <cstdio>
#include <vector>
#include <Eigen/Dense>

#include "grid.h"
#include "constant.h"
#include "particle.h"
#include "mpm_solver.h"

#include <glad/glad.h>

#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

const unsigned int WIDTH = 800;
const unsigned int HEIGHT = 600;

void errorCallback(int error, const char* description) {
    std::cerr << "Errors: " << description << std::endl;
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
}

int main() {
    std::vector<Grid> grids;
    std::vector<Particle> particles;

    // reserve space
    grids.reserve(GRID_BOUND_X * GRID_BOUND_Y * GRID_BOUND_Z);
    particles.reserve(20 * 20 * 21);

    // generate grids
    for (int x = 0; x < GRID_BOUND_X; x++)
        for (int y = 0; y < GRID_BOUND_Y; y++)
            for (int z = 0; z < GRID_BOUND_Z; z++) {
                grids.push_back(Grid(Eigen::Vector3i(x, y, z)));
            }

    // create a cube of particles
    float ptcl_lambda = LAMBDA,
          ptcl_mu = MU;
    for (int i = 0; i < 20; i++) {
        float pi = PARTICLE_DIAM * (40 + i);
        for (int j = 0; j < 20; j++) {
            float pj = PARTICLE_DIAM * (40 + j);
            for (int k = 0; k < 20; k++) {
                float pk = PARTICLE_DIAM * (40 + k);
                particles.push_back(
                    Particle(
                        Eigen::Vector3f(pi, pj, pk),
                        Eigen::Vector3f(0.0f, 0.0f, 0.0f),
                        1.0f,
                        ptcl_lambda,
                        ptcl_mu
                    )
                );
            }
        }
    }

    MPMSolver mpm_solver(particles, grids);

    // cudaDeviceSynchronize();

    // glfw: initialize and configure
    if (!glfwInit()) return EXIT_FAILURE;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // glfw window creation
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "LearnOpenGL", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return EXIT_FAILURE;
    }
    glfwMakeContextCurrent(window);

    // glfw setting callback
    glfwSetErrorCallback(errorCallback);
    glfwSetKeyCallback(window, keyCallback);

    // glad: load all OpenGL function pointers
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return EXIT_FAILURE;
    }

    std::cerr << "OpenGL version: " << glGetString(GL_VERSION) << std::endl;
    glEnable(GL_DEPTH_TEST);

    // render loop
    while (!glfwWindowShouldClose(window)) {
        // render
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // glfw: swap buffers and poll IO events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
