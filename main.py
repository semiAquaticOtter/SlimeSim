import pygame as pg 
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GL.ARB import shader_storage_buffer_object
from OpenGL.GLUT import *
from OpenGL.GLU import *

from dataclasses import dataclass

from ctypes import sizeof, c_float, Structure, byref

import numpy as np
import random
import math

#SECTION - Varibles

# width, height
width, height = 720, 480
displaySize = (width, height)

vertices = [
    -1.0, -1.0, 0.0, 0.0, 1.0,
    -1.0, 1.0, 0.0, 0.0, 0.0,
    1.0, 1.0, 0.0, 1.0, 0.0,
    1.0, -1.0, 0.0, 1.0, 1.0
]

indices = [
    0, 2, 1,
    0, 3, 2
]

#!SECTION - Varibles
#SECTION - Start
pg.init()
pg.display.set_mode(displaySize, pg.DOUBLEBUF|pg.OPENGL)

gl_version = glGetString(GL_VERSION)
print(gl_version.decode('utf-8'))

if pg.get_error() != "":
    print("Pygame error:", pg.get_error())
    pg.quit()
    quit()

#ANCHOR - Vertex Shader
vertexSource = open("./shader.vert", "r")
vertexShader = glCreateShader(GL_VERTEX_SHADER)
glShaderSource(vertexShader, vertexSource)
glCompileShader(vertexShader)

#ANCHOR - Fragment Shader
fragmentSource = open("./shader.frag", "r")
fragmentShader = glCreateShader(GL_FRAGMENT_SHADER)
glShaderSource(fragmentShader, fragmentSource)
glCompileShader(fragmentShader)

#ANCHOR - Compute Shader
computeSource = open("./shader.comp", "r")
computeShader = glCreateShader(GL_COMPUTE_SHADER)
glShaderSource(computeShader, computeSource)
glCompileShader(computeShader)

#ANCHOR - Check Compile Status
if glGetShaderiv(vertexShader, GL_COMPILE_STATUS) != GL_TRUE:
    print("Vertex shader compilation failed")
    err = glGetShaderInfoLog(vertexShader)
    print(err.decode('utf-8'))
    pg.quit()
    quit()

# Check fragment shader compilation status
if glGetShaderiv(fragmentShader, GL_COMPILE_STATUS) != GL_TRUE:
    print("Fragment shader compilation failed")
    err = glGetShaderInfoLog(fragmentShader)
    print(err.decode('utf-8'))
    pg.quit()
    quit()

# Check compute shader compilation status
if glGetShaderiv(computeShader, GL_COMPILE_STATUS) != GL_TRUE:
    print("Compute shader compilation failed")
    err = glGetShaderInfoLog(computeShader)
    print(err.decode('utf-8'))
    pg.quit()
    quit()

shader_program = glCreateProgram()
compute_program = glCreateProgram()

glAttachShader(shader_program, vertexShader)
glAttachShader(shader_program, fragmentShader)
glAttachShader(compute_program, computeShader)

glLinkProgram(shader_program)
glLinkProgram(compute_program)

#ANCHOR - Check Link status
# if glGetProgramiv(shader_program, GL_LINK_STATUS) != GL_TRUE:
#     print("Shader program linking failed")
#     err = glGetShaderInfoLog(shader_program)
#     print(err.decode('utf-8'))
#     pg.quit()
#     quit()

if glGetProgramiv(compute_program, GL_LINK_STATUS) != GL_TRUE:
    print("Compute program linking failed")
    print(glGetProgramInfoLog(compute_program))
    pg.quit()
    quit()


#ANCHOR - creating the render texture
trailmap = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D, trailmap)

glTextureParameteri(trailmap, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
glTextureParameteri(trailmap, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
glTextureParameteri(trailmap, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
glTextureParameteri(trailmap, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

glTextureStorage2D(trailmap, 1, GL_RGBA32F, width, height)
glBindImageTexture(0, trailmap, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F)

#ANCHOR - Create VAO and VBO for full-screen quad
screen_quad_vao = glGenVertexArrays(1)
glBindVertexArray(screen_quad_vao)

screen_quad_vbo = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, screen_quad_vbo)
glBufferData(GL_ARRAY_BUFFER, np.array(vertices, dtype=np.float32), GL_STATIC_DRAW)

# Position attribute (3 floats per vertex)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), None)
glEnableVertexAttribArray(0)

# Texture coordinate attribute (2 floats per vertex)
glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), ctypes.c_void_p(3 * sizeof(GLfloat)))
glEnableVertexAttribArray(1)

# Unbind the VAO
glBindVertexArray(0)

#!SECTION - Start

if __name__ == "__main__":
    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                #NOTE - exit stuff
                pg.quit()
                quit()
        glViewport(0, 0, width, height)


        glUseProgram(compute_program)
        glDispatchCompute(width // 8, height // 4, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        glUseProgram(shader_program)
        glBindVertexArray(screen_quad_vao)
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4)
        glBindVertexArray(0)  

        pg.display.flip()