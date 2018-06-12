#ifndef CAMERA_H
#define CAMERA_H

#include <GLFW/glfw3.h>

#include <glm/glm.hpp>

class CameraControl {
 public:
  CameraControl();

  void recordTime();

  glm::mat4 getLookAt();
  float getFOV();

  void processCameraKey(GLFWwindow* window);
  void processCameraMouse(double x_position, double y_position);
  void processCameraScroll(double y_offset);

 private:
  void reset();

  glm::vec3 camera_position_;
  glm::vec3 camera_front_;
  glm::vec3 camera_up_;

  float key_sensitivity_;
  float mouse_sensitivity_;

  float last_frame_;
  float delta_time_;

  float fov_;
  float pitch_;
  float yaw_;
  float last_x_;
  float last_y_;

  bool first_mouse;
};

#endif
