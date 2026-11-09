#version 430 core
in vec3  vColor;
in float vLifeFrac;
in float vCamDist;
out vec4 FragColor;
uniform sampler2D gPosMat;   // deferred G-buffer: xyz world hit pos, w mat (-1 = sky)
uniform vec3 camPos;
uniform vec2 viewport;
void main() {
    if (vLifeFrac < 0.08) discard;
    // Occlude against the scene: the raymarched world has no GL depth buffer, so
    // sample the G-buffer world position at this pixel and hide the particle if
    // a solid surface is nearer to the camera than the particle is.
    vec4 g = texture(gPosMat, gl_FragCoord.xy / viewport);
    if (g.w > -0.5 && vCamDist > length(g.xyz - camPos) + 0.5) discard;
    FragColor = vec4(vColor, 1.0);
}
