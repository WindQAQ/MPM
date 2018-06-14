#version 410 core
in vec2 texture_coordinate;

uniform sampler2D texture1;

void main() {
    gl_FragColor = texture(texture1, texture_coordinate);
}
