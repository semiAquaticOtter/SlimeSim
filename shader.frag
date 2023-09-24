#version 460 core

in vec2 TexCoord;
out vec4 FragColor;

void main()
{
    // Use the TexCoord to generate a gradient color
    FragColor = vec4(TexCoord.x, TexCoord.y, 0.0, 1.0);
}
