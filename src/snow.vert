#version 410 core
layout (location = 0) in vec4 a_position;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform float radius;
uniform float scale;

void main() {
    gl_Position = projection * view * model * a_position;
    gl_PointSize = scale * (radius / gl_Position.w);
}
