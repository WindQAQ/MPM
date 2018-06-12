#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <glad/glad.h>

#include "shader.h"

void check_errors(GLuint id, const std::string& type) {
  GLint success;
  if (type == "SHADER") {
    glGetShaderiv(id, GL_COMPILE_STATUS, &success);
    if (!success) {
      GLint logLength;
      glGetShaderiv(id, GL_INFO_LOG_LENGTH, &logLength);
      std::vector<GLchar> infoLog((logLength > 1) ? logLength : 1);
      glGetShaderInfoLog(id, logLength, nullptr, infoLog.data());
      std::cerr << infoLog.data() << std::endl;
    }
  } else {
    glGetProgramiv(id, GL_LINK_STATUS, &success);
    if (!success) {
      GLint logLength;
      glGetProgramiv(id, GL_INFO_LOG_LENGTH, &logLength);
      std::vector<GLchar> infoLog((logLength > 1) ? logLength : 1);
      glGetProgramInfoLog(id, logLength, nullptr, infoLog.data());
      std::cerr << infoLog.data() << std::endl;
    }
  }
}

GLuint loadShader(const std::string& vertex_path,
                  const std::string& fragment_path) {
  // 1. retrieve the vertex/fragment source code from filePath
  std::string vertex_code;
  std::string fragment_code;
  std::ifstream vertex_shader_file;
  std::ifstream fragment_shader_file;
  // ensure ifstream objects can throw exceptions:
  vertex_shader_file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
  fragment_shader_file.exceptions(std::ifstream::failbit |
                                  std::ifstream::badbit);
  try {
    // open files
    vertex_shader_file.open(vertex_path);
    fragment_shader_file.open(fragment_path);
    std::stringstream vertex_shader_stream, fragment_shader_stream;
    // read file's buffer contents into streams
    vertex_shader_stream << vertex_shader_file.rdbuf();
    fragment_shader_stream << fragment_shader_file.rdbuf();
    // close file handlers
    vertex_shader_file.close();
    fragment_shader_file.close();
    // convert stream into string
    vertex_code = vertex_shader_stream.str();
    fragment_code = fragment_shader_stream.str();
  } catch (std::ifstream::failure e) {
    std::cerr << "Failed to load shader source file!" << std::endl;
  }

  // 2. compile shaders
  auto compile_shader = [](GLuint& id, const std::string& code) {
    const char* shader_code = code.c_str();
    glShaderSource(id, 1, &shader_code, nullptr);
    glCompileShader(id);
    check_errors(id, "SHADER");
  };

  GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
  compile_shader(vertex_shader, vertex_code);
  GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
  compile_shader(fragment_shader, fragment_code);

  // id Program
  GLuint shader_program = glCreateProgram();
  glAttachShader(shader_program, vertex_shader);
  glAttachShader(shader_program, fragment_shader);
  glLinkProgram(shader_program);
  check_errors(shader_program, "PROGRAM");

  // delete the shaders as they're linked into our program now and no longer
  // necessary
  glDeleteShader(vertex_shader);
  glDeleteShader(fragment_shader);

  return shader_program;
}
