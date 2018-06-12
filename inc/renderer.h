#ifndef RENDER_H
#define RENDER_H

#include "camera.h"

struct GLBuffers {
    GLuint vao;
    GLuint vbo;
    GLuint ebo;
};

struct CUDABUffers {
    GLuint vao;
    GLuint vbo;
};

class Renderer {
  public:
    Renderer(int width, int height, int number);
    void render();
    void renderWall();
    void renderSnow();

    CUDABUffers snow_buffers;

  private:
    const GLfloat wall_vertices[20] = {
        // positions        // texture coords
         0.5f,  0.5f, 0.0f, 1.0f, 1.0f,  //
        -0.5f,  0.5f, 0.0f, 0.0f, 1.0f,  //
        -0.5f, -0.5f, 0.0f, 0.0f, 0.0f,  //
         0.5f, -0.5f, 0.0f, 1.0f, 0.0f,  //
    };

    const GLuint indices[6] = {
		    0, 1, 2, //
		    2, 3, 0
	  };

    int width;
    int height;
    float aspect_ratio;

    int number;

    GLuint plane_shader;
    GLuint snow_shader;

    GLuint texture1;
    GLuint texture2;

    GLBuffers plane_buffers;

    CameraControl camera;
    glm::mat4 view;
    glm::mat4 projection;

    float radius = 0.008f;
};

#endif
