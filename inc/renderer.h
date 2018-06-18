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
    void setOrigin();
    void setUp();
    void setFront();
    void setSide();

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

    glm::mat4 origin_camera_ = glm::lookAt(glm::vec3(5.5f, 1.2f, 5.5f),  // camera position
                                           glm::vec3(0.0f, 1.2f, 0.0f),  // target position
                                           glm::vec3(0.0f, 1.0f, 0.0f)   // up vector
    );

    glm::mat4 up_camera_ = glm::lookAt(glm::vec3(0.0f, 6.8f, 0.0f),  // camera position
                                       glm::vec3(0.0f, 1.2f, 0.0f),  // target position
                                       glm::vec3(0.0f, 0.0f, -1.0f)   // up vector
    );

    glm::mat4 front_camera_ = glm::lookAt(glm::vec3(0.0f, 1.2f, 7.0f),  // camera position
                                          glm::vec3(0.0f, 1.2f, 0.0f),  // target position
                                          glm::vec3(0.0f, 1.0f, 0.0f)   // up vector
    );

    glm::mat4 side_camera_ = glm::lookAt(glm::vec3(7.0f, 1.2f, 0.0f),  // camera position
                                         glm::vec3(0.0f, 1.2f, 0.0f),  // target position
                                         glm::vec3(0.0f, 1.0f, 0.0f)   // up vector
    );

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
    GLfloat radius_ = 0.015f;

    void renderWall();
    void renderFloor();
    void renderSnow();
};

#endif
