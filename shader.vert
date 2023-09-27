#version 460 core

in vec3 pos;
in vec2 uvs;
in vec2 inPosition;

out vec2 UVs;

void main() {
    gl_Position = vec4(pos.x, pos.y, pos.z, 1.0);
    // gl_Position = vec4(inPosition, 0.0, 1.0);
    UVs = uvs;
}