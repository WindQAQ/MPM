#include <glad/glad.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "renderer.h"
#include "shader.h"
#include "texture.h"

Renderer::Renderer(int width, int height, int number)
        : width_(width),
          height_(height),
          number_(number) {
    aspect_ratio_ = static_cast<float>(width_) / height_;

    view_ = origin_camera_;
    projection_ = glm::perspective(glm::radians(fov_), static_cast<float>(width_) / height_, 0.1f, 100.0f);

    // bind textures on corresponding texture units
    texture1_ = loadTexture("./images/container.jpg");
    texture2_ = loadTexture("./images/container.jpg");
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture1_);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, texture2_);

    // plane
    // ------------------------------------------------------------------------
    plane_shader_ = loadShader("./src/plane.vert", "./src/plane.frag");

    glUseProgram(plane_shader_);
    glUniformMatrix4fv(glGetUniformLocation(plane_shader_, "projection"), 1, GL_FALSE, glm::value_ptr(projection_));

    glGenVertexArrays(1, &plane_buffers_.vao);
    glBindVertexArray(plane_buffers_.vao);

    glGenBuffers(1, &plane_buffers_.vbo);
    glBindBuffer(GL_ARRAY_BUFFER, plane_buffers_.vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(wall_vertices_), wall_vertices_, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // texture coord attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glGenBuffers(1, &plane_buffers_.ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, plane_buffers_.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices_), indices_, GL_STATIC_DRAW);
    // ------------------------------------------------------------------------

    // snow
    // ------------------------------------------------------------------------
    snow_shader_ = loadShader("./src/snow.vert", "./src/snow.frag");

    glUseProgram(snow_shader_);
    glUniformMatrix4fv(glGetUniformLocation(snow_shader_, "projection"), 1, GL_FALSE, glm::value_ptr(projection_));
    glUniform1f(glGetUniformLocation(snow_shader_, "radius"), radius_);
    glUniform1f(glGetUniformLocation(snow_shader_, "scale"), width_ / aspect_ratio_ * (1.0f / tanf(fov_ * 0.5f)));

    glGenVertexArrays(1, &snow_buffers_.vao);
    glBindVertexArray(snow_buffers_.vao);

    glGenBuffers(1, &snow_buffers_.vbo);
    glBindBuffer(GL_ARRAY_BUFFER, snow_buffers_.vbo);
    glBufferData(GL_ARRAY_BUFFER, number_ * 4 * sizeof(float), 0, GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
    // ------------------------------------------------------------------------
}

void Renderer::render() {
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // renderWall();
    renderFloor();
    renderSnow();
}

void Renderer::renderWall() {
    glUseProgram(plane_shader_);
    glUniform1i(glGetUniformLocation(plane_shader_, "texture1"), 1);
    glBindVertexArray(plane_buffers_.vao);

    glUniformMatrix4fv(glGetUniformLocation(plane_shader_, "view"), 1, GL_FALSE, glm::value_ptr(view_));
    glm::mat4 model(1.0f);
    model = glm::translate(model, glm::vec3(0.0f, 1.5f, -1.5f));
    model = glm::scale(model, glm::vec3(3.0f, 3.0f, 3.0f));
    glUniformMatrix4fv(glGetUniformLocation(plane_shader_, "model"), 1, GL_FALSE, glm::value_ptr(model));
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}

void Renderer::renderFloor() {
    glUseProgram(plane_shader_);
    glUniform1i(glGetUniformLocation(plane_shader_, "texture1"), 0);
    glBindVertexArray(plane_buffers_.vao);

    glUniformMatrix4fv(glGetUniformLocation(plane_shader_, "view"), 1, GL_FALSE, glm::value_ptr(view_));
    glm::mat4 model(1.0f);
    model = glm::scale(model, glm::vec3(5.0f, 5.0f, 5.0f));
    model = glm::rotate(model, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    glUniformMatrix4fv(glGetUniformLocation(plane_shader_, "model"), 1, GL_FALSE, glm::value_ptr(model));
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}

void Renderer::renderSnow() {
    glUseProgram(snow_shader_);
    glBindVertexArray(snow_buffers_.vao);

    glUniformMatrix4fv(glGetUniformLocation(snow_shader_, "view"), 1, GL_FALSE, glm::value_ptr(view_));
    glm::mat4 model(1.0f);
    glUniformMatrix4fv(glGetUniformLocation(snow_shader_, "model"), 1, GL_FALSE, glm::value_ptr(model));
    glDrawArrays(GL_POINTS, 0, GLsizei(number_));
}

GLuint Renderer::getSnowVBO() {
    return snow_buffers_.vbo;
}

void Renderer::setOrigin() {
    view_ = origin_camera_;
}

void Renderer::setUp() {
    view_ = up_camera_;
}

void Renderer::setFront() {
    view_ = front_camera_;
}

void Renderer::setSide() {
    view_ = side_camera_;
}
