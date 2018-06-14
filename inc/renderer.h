#ifndef RENDER_H
#define RENDER_H

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
    GLuint getSnowVBO();

  private:
    const GLfloat wall_vertices_[20] = {
        // positions        // texture coords
         0.5f,  0.5f, 0.0f, 1.0f, 1.0f,  //
        -0.5f,  0.5f, 0.0f, 0.0f, 1.0f,  //
        -0.5f, -0.5f, 0.0f, 0.0f, 0.0f,  //
         0.5f, -0.5f, 0.0f, 1.0f, 0.0f,  //
    };

    const GLuint indices_[6] = {
        0, 1, 2, //
        2, 3, 0
    };

    // window size
    int width_;
    int height_;
    float aspect_ratio_;

    // particle number;
    int number_;

    GLuint plane_shader_;
    GLuint snow_shader_;

    GLuint texture1_;
    GLuint texture2_;

    GLBuffers plane_buffers_;
    CUDABUffers snow_buffers_;

    glm::mat4 view_;
    glm::mat4 projection_;
    float fov_ = 45.0f;

    // snow point size;
    GLfloat radius_ = 0.02f;

    void renderWall();
    void renderFloor();
    void renderSnow();
};

#endif
