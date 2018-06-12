#ifndef SHADER_H
#define SHADER_H

#include <string>

#include <glad/glad.h>

GLuint loadShader(const std::string& vertex_path, const std::string& fragment_path);

#endif
