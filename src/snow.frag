#version 410 core
in vec3 pos;

out vec4 frag_color;

uniform mat4 view;
uniform mat4 projection;
uniform float radius;

float rand(vec2 co) {
    return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
	//calculate normal
	vec3 normal;
	normal.xy = gl_PointCoord * 2.0 - 1.0;
	float r2 = dot(normal.xy, normal.xy);

	if (r2 > 1.0) {
		discard;
	}

	frag_color = vec4(1);
}
