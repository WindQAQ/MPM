#include <iostream>
#include <cstdio>
#include <vector>
#include <Eigen/Dense>
#include <thrust/device_new.h>

#include "grid.h"
#include "constant.h"
#include "particle.h"
#include "mpm_solver.h"

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

void createSphere(std::vector<Particle>& particles,
                  const Eigen::Vector3i& center, const float radius,
                  const Eigen::Vector3f& velocity, const float mass,
                  const float lambda, const float mu,
                  const int sample_rate=1) {

    float sphere_radius = PARTICLE_DIAM * radius / 2.0;

    for (int x = 0; x < GRID_BOUND_X; x++) {
        for (int y = 0; y < GRID_BOUND_Y; y++) {
            for (int z = 0; z < GRID_BOUND_Z; z++) {
                for (int s = 0; s < sample_rate; s++) {
                    float r1 = 1e-3 + static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                    float r2 = 1e-3 + static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                    float r3 = 1e-3 + static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                    Eigen::Vector3f jitter(r1, r2, r3);
                    jitter.array() *= PARTICLE_DIAM;

                    Eigen::Vector3f pos = Eigen::Vector3i(x, y, z).cast<float>() * PARTICLE_DIAM + jitter;

                    if ((pos - center.cast<float>() * PARTICLE_DIAM).norm() < sphere_radius) {
                        particles.push_back(
                            Particle(
                                pos,
                                velocity,
                                mass,
                                lambda,
                                mu
                            )
                        );
                    }
                }
            }
        }
    }
}

void createCuboid(std::vector<Particle>& particles,
                  const Eigen::Vector3i& corner, const Eigen::Vector3i& dims,
                  const Eigen::Vector3f& velocity, const float mass,
                  const float lambda, const float mu) {


    for (int x = 0; x < dims(0); x++) {
        for (int y = 0; y < dims(1); y++) {
            for (int z = 0; z < dims(2); z++) {
                float r1 = 1e-3 + static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                float r2 = 1e-3 + static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                float r3 = 1e-3 + static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                Eigen::Vector3f jitter(r1, r2, r3);
                jitter.array() *= PARTICLE_DIAM;

                Eigen::Vector3f pos = (corner.cast<float>() + Eigen::Vector3i(x, y, z).cast<float>()) * PARTICLE_DIAM + jitter;

                particles.push_back(
                    Particle(
                        pos,
                        velocity,
                        mass,
                        lambda,
                        mu
                    )
                );
            }
        }
    }
}

void createAlphabet(std::vector<Particle>& particles,
                    const Eigen::Vector3i& corner,
                    const Eigen::Vector3f& velocity,
                    const float mass, const float lambda, const float mu,
                    const char alphabet, const int scale=5, const int sample_rate=1) {
    const char G[7][6] = {" *** ", "*   *", "*    ", "*    ", "*  **", "*   *", " *** "};
    const char P[7][6] = {"**** ", "*   *", "*   *", "**** ", "*    ", "*    ", "*    "};
    const char U[7][6] = {"*   *", "*   *", "*   *", "*   *", "*   *", "*   *", " *** "};
    for (int y = 0; y < 7 * scale; y++) {
        for (int x = 0; x < 5 * scale; x++) {
            bool cont = false;
            switch (alphabet) {
            case 'G': if (G[6 - y / scale][4 - x / scale] == ' ') cont = true; break;
            case 'P': if (P[6 - y / scale][4 - x / scale] == ' ') cont = true; break;
            case 'U': if (U[6 - y / scale][4 - x / scale] == ' ') cont = true; break;
            default: cont = true;
            }

            if (cont) continue;

            for (int z = 0; z < 10; z++) {
                for (int s = 0; s < sample_rate; s++) {
                    float r1 = 1e-3 + static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                    float r2 = 1e-3 + static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                    float r3 = 1e-3 + static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                    Eigen::Vector3f jitter(r1, r2, r3);
                    jitter.array() *= PARTICLE_DIAM;

                    Eigen::Vector3f pos = (corner.cast<float>() + Eigen::Vector3i(x, y, z).cast<float>()) * PARTICLE_DIAM + jitter;

                    particles.push_back(
                        Particle(
                            pos,
                            velocity,
                            mass,
                            lambda,
                            mu
                        )
                    );
                }
            }
        }
    }
}

int main(int argc, char *argv[]) {

    // configure
    bool save_frame = false;

    // read command-line arguments
    for (int i = 1; i < argc; i++) {
        std::string args(argv[i]);
        if (args.find("-save_frame") != std::string::npos) save_frame = true;
    }

    std::vector<Grid> grids;
    std::vector<Particle> particles;

    // reserve space
    grids.reserve(GRID_BOUND_X * GRID_BOUND_Y * GRID_BOUND_Z);
    particles.reserve(20 * 20 * 21);

    // generate grids
    /*
    for (int x = 0; x < GRID_BOUND_X; x++)
        for (int y = 0; y < GRID_BOUND_Y; y++)
            for (int z = 0; z < GRID_BOUND_Z; z++) {
                grids.push_back(Grid(Eigen::Vector3i(x, y, z)));
            }
    */

    float lambda = LAMBDA, mu = MU;

    Eigen::Vector3i center(70, 50, 70);

    // two balls
    // createSphere(particles, center, 20, Eigen::Vector3f(0.0f, 0.0f, 0.0f), pow(PARTICLE_DIAM / 2, 3) * 3.14 * 4 / 3 * DENSITY, lambda, mu, 1);
    // center(1) = (int) (70 / 1.5);
    // center(2) = 70 / 4;
    // createSphere(particles, center, 5, Eigen::Vector3f(0.0f, 0.0f, 5.0f), pow(PARTICLE_DIAM / 2, 3) *  3.14 * 4 / 3 * DENSITY, lambda, mu, 30);

    // vertical
    // center(1) = 40;
    // createSphere(particles, center, 20, Eigen::Vector3f(0.0f, -5.0f, 0.0f), pow(PARTICLE_DIAM / 2, 3) * 3.14 * 4 / 3 * DENSITY, lambda, mu, 50);

    // cuboid
    // Eigen::Vector3i corner(70, 20, 70), dims(10, 50, 10);
    // createCuboid(particles, corner, dims, Eigen::Vector3f(0.0f, 0.0f, 0.0f), pow(PARTICLE_DIAM / 2, 3) * 3.14 * 4 / 3 * DENSITY, lambda, mu);

    // pyramid
    // for (int y = 0; y < 40; y++) {
    //      for (int x = 0; x < 40 - y; x++) {
    //          for (int z = 0; z < 40 - y; z++) {

    //              Eigen::Vector3f pos(40 - x, y, 40 - z);
    //              pos += Eigen::Vector3f(50, 20, 50);
    //              pos.array() *= PARTICLE_DIAM;

    //              particles.push_back(
    //                  Particle(
    //                      pos,
    //                      Eigen::Vector3f(0.0f, -5.0f, 0.0f),
    //                      pow(PARTICLE_DIAM / 2, 3) * 3.14 * 4 / 3 * DENSITY,
    //                      lambda,
    //                      mu
    //                  )
    //              );
    //          }
    //      }
    //  }

    {
        Eigen::Vector3i corner(165, 50, 70);
        createAlphabet(particles, corner, Eigen::Vector3f(0.0f, -10.0f, 0.0f), pow(PARTICLE_DIAM / 2, 3) * 3.14 * 4 / 3 * DENSITY, lambda, mu, 'G', 5, 15);
    }
    {
        Eigen::Vector3i corner(125, 50, 70);
        createAlphabet(particles, corner, Eigen::Vector3f(0.0f, -10.0f, 0.0f), pow(PARTICLE_DIAM / 2, 3) * 3.14 * 4 / 3 * DENSITY, lambda, mu, 'P', 5, 15);
    }
    {
        Eigen::Vector3i corner(85, 50, 70);
        createAlphabet(particles, corner, Eigen::Vector3f(0.0f, -10.0f, 0.0f), pow(PARTICLE_DIAM / 2, 3) * 3.14 * 4 / 3 * DENSITY, lambda, mu, 'G', 5, 15);
    }
    {
        Eigen::Vector3i corner(45, 50, 70);
        createAlphabet(particles, corner, Eigen::Vector3f(0.0f, -10.0f, 0.0f), pow(PARTICLE_DIAM / 2, 3) * 3.14 * 4 / 3 * DENSITY, lambda, mu, 'P', 5, 15);
    }
    {
        Eigen::Vector3i corner(5, 50, 70);
        createAlphabet(particles, corner, Eigen::Vector3f(0.0f, -10.0f, 0.0f), pow(PARTICLE_DIAM / 2, 3) * 3.14 * 4 / 3 * DENSITY, lambda, mu, 'U', 5, 15);
    }


    std::cout << "number of grids: " << grids.size() << ", number of bytes in grids: " << grids.size() * sizeof(Grid) << std::endl;
    std::cout << "number of particles: " << particles.size() << ", number of bytes in particles: " << particles.size() * sizeof(Particle) << std::endl;

    MPMSolver mpm_solver(particles);

    auto ret = cudaGetLastError();
    assert(ret == cudaSuccess);

    // grids.clear();
    // particles.clear();

    ret = cudaGetLastError();
    assert(ret == cudaSuccess);

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

        std::cout << "step: " << step << std::endl;

        if (save_frame) {
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
