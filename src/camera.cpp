#include <algorithm>

#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "camera.h"

CameraControl::CameraControl() {
  reset();
}

void CameraControl::recordTime() {
  float current_frame = glfwGetTime();
  delta_time_ = current_frame - last_frame_;
  last_frame_ = current_frame;
}

glm::mat4 CameraControl::getLookAt() {
  return glm::lookAt(camera_position_,                  // camera position
                     camera_position_ + camera_front_,  // target position
                     camera_up_                         // up vector
  );
}

float CameraControl::getFOV() { return fov_; }

void CameraControl::processCameraKey(GLFWwindow* window) {
  float camera_speed = key_sensitivity_ * delta_time_;
  if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
    camera_position_ += camera_speed * camera_front_;
  if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
    camera_position_ -= camera_speed * camera_front_;
  if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
    camera_position_ -=
        glm::normalize(glm::cross(camera_front_, camera_up_)) * camera_speed;
  if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
    camera_position_ +=
        glm::normalize(glm::cross(camera_front_, camera_up_)) * camera_speed;

  if (glfwGetKey(window, GLFW_KEY_R))
    reset();
}

void CameraControl::processCameraMouse(double x_position, double y_position) {
  if (first_mouse) {
    last_x_ = x_position;
    last_y_ = y_position;
    first_mouse = false;
  }

  float x_offset = x_position - last_x_;
  // reversed since y-coordinates range from bottom to top
  float y_offset = last_y_ - y_position;
  last_x_ = x_position;
  last_y_ = y_position;

  x_offset *= mouse_sensitivity_;
  y_offset *= mouse_sensitivity_;

  yaw_ += x_offset;
  pitch_ += y_offset;

  pitch_ = glm::clamp(pitch_, -89.0f, 89.0f);

  glm::vec3 front(cos(glm::radians(pitch_)) * cos(glm::radians(yaw_)),
                  sin(glm::radians(pitch_)),
                  cos(glm::radians(pitch_)) * sin(glm::radians(yaw_)));
  camera_front_ = glm::normalize(front);
}

void CameraControl::processCameraScroll(double y_offset) {
  fov_ = glm::clamp(fov_ - static_cast<float>(y_offset), 1.0f, 45.0f);
}

void CameraControl::reset() {
  camera_position_ = glm::vec3(0.0f, 0.0f, 6.0f);
  camera_front_ = glm::vec3(0.0f, 0.0f, -1.0f);
  camera_up_ = glm::vec3(0.0f, 1.0f, 0.0f);

  key_sensitivity_ = 10.0f;
  mouse_sensitivity_ = 0.05f;

  last_frame_ = 0.0f;
  delta_time_ = 0.0f;

  fov_ = 45.0f;
  pitch_ = 0.0f;
  yaw_ = -90.0f;
  last_x_ = 400;
  last_y_ = 300;

  first_mouse = true;
}
