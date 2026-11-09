#version 430 core
in vec2 vUV;
uniform vec2 viewport;
out vec4 FragColor;
void main() {
    vec2 px = gl_FragCoord.xy;
    vec2 ctr = viewport * 0.5;
    vec2 d = abs(px - ctr);
    vec2 sz = vec2(380.0, 160.0);
    if (d.x > sz.x * 0.5 || d.y > sz.y * 0.5) {
        FragColor = vec4(0.0, 0.0, 0.0, 0.55);
        return;
    }
    bool border = d.x > sz.x * 0.5 - 4.0 || d.y > sz.y * 0.5 - 4.0;
    vec2 r = px - ctr;
    bool icon = (abs(r.x + 22.0) < 9.0 && abs(r.y) < 36.0) ||
                (abs(r.x - 22.0) < 9.0 && abs(r.y) < 36.0);
    if (icon)   { FragColor = vec4(0.95, 0.95, 0.95, 1.0); return; }
    if (border) { FragColor = vec4(0.92, 0.92, 0.92, 1.0); return; }
    FragColor = vec4(0.10, 0.12, 0.18, 0.92);
}
