#version 410 core
layout (location = 0) in vec3 a_position;
layout (location = 1) in vec2 a_texture_coordinate;

out vec2 texture_coordinate;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    gl_Position = projection * view * model * vec4(a_position, 1.0);
    texture_coordinate = a_texture_coordinate;
}
