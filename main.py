import pygame as pg 
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GL.ARB import shader_storage_buffer_object
from OpenGL.GLUT import *
from OpenGL.GLU import *

from dataclasses import dataclass

from ctypes import sizeof, c_float, Structure, byref, c_void_p

import numpy as np
import random
import math

#SECTION - Varibles

# width, height
width, height = 720, 480
displaySize = (width, height)
numAgents = 100

#!SECTION - Varibles
#SECTION - Dataclasses
#ANCHOR - Vector 2
@dataclass
class Vec2:
    x: float
    y: float

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Vec2(self.x * other, self.y * other)
        elif isinstance(other, Vec2):
            return Vec2(self.x * other.x, self.y * other.y)
        else:
            raise TypeError("Unsupported opperand type")
        
    def __add__(self, other):
        if isinstance(other, Vec2):
            return Vec2(self.x + other.x, self.y + other.y)
        elif isinstance(other, (int, float)):
            return Vec2(self.x + other, self.y + other)
        else:
            raise TypeError("Unsupported opperand type")
        
    def __str__(self):
        return f"({self.x}, {self.y})"
    
#ANCHOR - Agent
class AgentStruct(Structure):
    _fields_ = [("pos", c_float * 2), ("angle", c_float)]

@dataclass
class Agent:
    pos: Vec2 = Vec2(0.0, 0.0)
    angle: float = 0.0

    def to_struct(self):
        pos_array = (c_float * 2)(self.pos.x, self.pos.y)
        return AgentStruct(pos_array, self.angle)
    
    @classmethod
    def from_struct(cls, agent_struct):
        return cls(agent_struct.pos, agent_struct.angle)
    
class Agent2:
    def __init__(self, x, y, angle):
        self.x = x
        self.y = y
        self.angle = angle
    
#!SECTION - Dataclasses
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
if glGetProgramiv(shader_program, GL_LINK_STATUS) != GL_TRUE:
    print("Shader program linking failed")
    err = glGetShaderInfoLog(shader_program)
    print(err.decode('utf-8'))
    pg.quit()
    quit()

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

def random_inside_unit_circle():
    while True:
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x**2 + y**2 <= 1:
            return Vec2(x, y)

#ANCHOR - Create agent array
agentsArray = [Agent2(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 2 * math.pi)) for _ in range(numAgents)]
npAgentsArray = np.array([(agent.x, agent.y, agent.angle) for agent in agentsArray], dtype=np.float32)

ubo = glGenBuffers(1)
glBindBuffer(GL_UNIFORM_BUFFER, ubo)
glBufferData(GL_UNIFORM_BUFFER, npAgentsArray.nbytes, npAgentsArray, GL_DYNAMIC_DRAW)

UBO_bindPoint = 0
glBindBufferBase(GL_UNIFORM_BUFFER, UBO_bindPoint, ubo)
# print(flattened_data)

#ANCHOR - Triangles?
vertices =np.array([
    # Vertex positions (x, y) followed by texture coordinates (u, v)
    -1.0, -1.0, 0.0, 0.0,
    -1.0, 1.0, 0.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 
    -1.0, -1.0, 0.0, 0.0,
    1.0, 1.0, 1.0, 1.0,
    1.0, -1.0, 1.0, 0.0
], dtype=np.float32)

quad_vao = glGenVertexArrays(1)
glBindVertexArray(quad_vao)

quad_vbo = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, quad_vbo)
glBufferData(GL_ARRAY_BUFFER, vertices, GL_STATIC_DRAW)

glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), None)
glEnableVertexAttribArray(0)

glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), c_void_p(2 * sizeof(GLfloat)))
glEnableVertexAttribArray(1)

#!SECTION - Start

fbo = glGenFramebuffers(1)

if __name__ == "__main__":
    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                #NOTE - exit stuff
                # glDeleteBuffers(1, [ssbo])
                pg.quit()
                quit()

        glUseProgram(compute_program)
        # glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo)
        glUniform1ui(glGetUniformLocation(compute_program, "numAgents"), numAgents)
        glUniform1i(glGetUniformLocation(compute_program, "trailmap_width"), width)
        glUniform1i(glGetUniformLocation(compute_program, "trailmap_height"), height)
        glDispatchCompute(width, height, 1)
        glMemoryBarrier(GL_ALL_BARRIER_BITS)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0, 0, width, height)
        glClear(GL_COLOR_BUFFER_BIT)

        error_code = glGetError()
        if error_code != GL_NO_ERROR:
            print(f"OpenGL error before glUseProgram: {error_code}")

        glUseProgram(shader_program)

        error_code = glGetError()
        if error_code != GL_NO_ERROR:
            print(f"OpenGL error after glUseProgram: {error_code}")

        glBindVertexArray(quad_vao)
        glBindTexture(GL_TEXTURE_2D, trailmap)
        glDrawArrays(GL_TRIANGLES, 0, 6)

        pg.display.flip()