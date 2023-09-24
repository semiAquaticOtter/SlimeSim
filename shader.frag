#version 460 core

out vec4 fragColor;
uniform sampler2D trailmap;

in vec2 UVs;

void main() {
    fragColor = texture(trailmap, UVs);
}
