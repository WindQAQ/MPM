#version 410 core
in vec2 texture_coordinate;

uniform sampler2D texture1;
uniform sampler2D texture2;

void main() {
    gl_FragColor = mix(texture(texture1, texture_coordinate), texture(texture2, texture_coordinate), 0.7);
}
