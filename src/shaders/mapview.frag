#version 430 core
// Explored-world map overlay composite: blend the TAA-resolved diorama over
// the live scene. mapTex alpha is coverage (1 = diorama/pin, 0 = see-through);
// the uncovered remainder gets a dark veil so the map stays readable over a
// bright scene. Premultiplied output — blend with (ONE, ONE_MINUS_SRC_ALPHA).
in vec2 vUV;
out vec4 FragColor;

uniform sampler2D mapTex;
uniform float veil;   // 0..1 dim strength on the uncovered scene

void main() {
    vec4 m = texture(mapTex, vUV);
    float cover = clamp(m.a, 0.0, 1.0);
    FragColor = vec4(m.rgb * cover, cover + veil * (1.0 - cover));
}
