#version 410 core
layout (location = 0) in vec4 a_position;

out vec3 pos;

uniform mat4 projection;
uniform mat4 view;
uniform float radius;
uniform float scale;

void main() {
	  vec4 view_position = view * a_position;
    gl_Position = projection * view_position;
	  pos = view_position.xyz;
	  gl_PointSize = scale * (radius / gl_Position.w);
}
