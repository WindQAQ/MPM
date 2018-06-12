#include <glad/glad.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "renderer.h"
#include "shader.h"
#include "texture.h"
#include "camera.h"

Renderer::Renderer(int width, int height, int number)
        : width(width),
          height(height),
          number(number) {
    aspect_ratio = static_cast<float>(width) / height;

    view = camera.getLookAt();
    projection = glm::perspective(glm::radians(camera.getFOV()), static_cast<float>(width) / height, 0.1f, 100.0f);

    plane_shader = loadShader("./src/plane.vert", "./src/plane.frag");

    glGenVertexArrays(1, &plane_buffers.vao);

    glGenBuffers(1, &plane_buffers.vbo);
    glBindBuffer(GL_ARRAY_BUFFER, plane_buffers.vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(wall_vertices), wall_vertices, GL_STATIC_DRAW);

    glGenBuffers(1, &plane_buffers.ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, plane_buffers.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    glBindVertexArray(0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    texture1 = loadTexture("./container.jpg");
    texture2 = loadTexture("./doge.jpg");

    glUseProgram(plane_shader);
    glUniform1i(glGetUniformLocation(plane_shader, "texture1"), 0);
    glUniform1i(glGetUniformLocation(plane_shader, "texture2"), 1);

    snow_shader = loadShader("./src/snow.vert", "./src/snow.frag");

    glGenVertexArrays(1, &snow_buffers.vao);

	glGenBuffers(1, &snow_buffers.vbo);
	glBindBuffer(GL_ARRAY_BUFFER, snow_buffers.vbo);
	glBufferData(GL_ARRAY_BUFFER, number * 4 * sizeof(float), 0, GL_DYNAMIC_DRAW);

    glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Renderer::render() {
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    renderSnow();
    renderWall();
}

void Renderer::renderWall() {
    // bind textures on corresponding texture units
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture1);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, texture2);

    glUseProgram(plane_shader);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // texture coord attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glUniformMatrix4fv(glGetUniformLocation(plane_shader, "view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(plane_shader, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

    glBindVertexArray(plane_buffers.vao);
	glBindBuffer(GL_ARRAY_BUFFER, plane_buffers.vbo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, plane_buffers.ebo);

    for (int i = 1; i < 4; i++) {
        glm::mat4 model(1.0f);
        model = glm::scale(model, glm::vec3(3.0f, 3.0f, 3.0f));
        model = glm::rotate(model, i * glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        model = glm::translate(model, glm::vec3(0.0f, 0.0f, 0.5f));
        glUniformMatrix4fv(glGetUniformLocation(plane_shader, "model"), 1, GL_FALSE, glm::value_ptr(model));
	    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    }

    glBindVertexArray(0);
}

void Renderer::renderSnow() {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glUseProgram(snow_shader);
    glUniformMatrix4fv(glGetUniformLocation(snow_shader, "view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(snow_shader, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
    glUniform1f(glGetUniformLocation(snow_shader, "radius"), radius);
    glUniform1f(glGetUniformLocation(snow_shader, "scale"), width / aspect_ratio * (1.0f / tanf(camera.getFOV() * 0.5f)));

	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glEnable(GL_CULL_FACE);

	//Draw snow
	glBindVertexArray(snow_buffers.vao);
	glBindBuffer(GL_ARRAY_BUFFER, snow_buffers.vbo);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(0);
	glDrawArrays(GL_POINTS, 0, GLsizei(number));

    glBindVertexArray(0);
}
