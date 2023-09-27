#version 460 core

out vec4 fragColor;
uniform sampler2D trailmap;

in vec2 UVs;

void main() {
    // vec2 texCoord = (gl_FragCoord.xy + 1.0) / 2.0;
    fragColor = texture(trailmap, UVs);
}
