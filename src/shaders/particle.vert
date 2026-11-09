#version 430 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;
layout(location = 2) in float aLifeFrac;
uniform mat4 viewProj;
uniform vec3 camPos;
out vec3  vColor;
out float vLifeFrac;
out float vCamDist;
void main() {
    vec4 cs = viewProj * vec4(aPos, 1.0);
    gl_Position = cs;
    float d = max(cs.w, 1.0);
    gl_PointSize = clamp(1600.0 / d, 3.0, 22.0);
    vColor    = aColor;
    vLifeFrac = aLifeFrac;
    vCamDist  = length(aPos - camPos);   // for the scene-occlusion test in the frag
}
