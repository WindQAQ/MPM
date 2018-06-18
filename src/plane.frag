#version 410 core
in vec2 texture_coordinate;

uniform sampler2D texture1;

void main() {
    // if(mod(floor(texture_coordinate.x * 60), 10) == 0 || mod(floor(texture_coordinate.y * 60), 10) == 0)
    if (mod(mod(floor(texture_coordinate.x * 10), 10) + mod(floor(texture_coordinate.y * 10), 10), 2) == 0)
        gl_FragColor = vec4(0.8, 0.8, 0.8, 0.0);
    else
        gl_FragColor = vec4(0.2);
        // gl_FragColor = texture(texture1, texture_coordinate);
}
