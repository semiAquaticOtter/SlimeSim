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

# VARS
numAgents = 1000
width, height = 720, 480

# dataclasses
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
        else:
            raise TypeError("Unsupported opperand type")
        
    def __str__(self):
        return f"({self.x}, {self.y})"

@dataclass
class Vec3f:
    x: float
    y: float
    z: float

class AgentStruct(Structure):
    _fields_ = [("pos", c_float * 2), ("angle", c_float)]

@dataclass
class Agent:
    pos: Vec2 = Vec2(0, 0)
    angle: float = 0.0

    def to_struct(self):
        pos_array = (c_float * 2)(self.pos.x, self.pos.y)
        return AgentStruct(pos_array, self.angle)
    
    @classmethod
    def from_struct(cls, agent_struct):
        return cls(agent_struct.pos, agent_struct.angle)

# read shaders
vertexShader = open("./shader.vert", "r")
fragShader = open("./shader.frag", "r")
computeShader = open("./shader.comp", "r")

pg.init()
display = (width, height)
pg.display.set_mode(display, DOUBLEBUF|OPENGL)

# Create and compile the vertex shader
vertex_shader = glCreateShader(GL_VERTEX_SHADER)
glShaderSource(vertex_shader, vertexShader)
glCompileShader(vertex_shader)

# Create and compile the fragment shader
fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
glShaderSource(fragment_shader, fragShader)
glCompileShader(fragment_shader)

# Create and compile the compute shader
compute_shader = glCreateShader(GL_COMPUTE_SHADER)
glShaderSource(compute_shader, computeShader)
glCompileShader(compute_shader)

shader_program = glCreateProgram()
compute_program = glCreateProgram()

# Check vertex shader compilation status
if glGetShaderiv(vertex_shader, GL_COMPILE_STATUS) != GL_TRUE:
    print("Vertex shader compilation failed")
    err = glGetShaderInfoLog(vertex_shader)
    print(err.decode('utf-8'))
    pg.quit()
    quit()

# Check fragment shader compilation status
if glGetShaderiv(fragment_shader, GL_COMPILE_STATUS) != GL_TRUE:
    print("Fragment shader compilation failed")
    err = glGetShaderInfoLog(fragment_shader)
    print(err.decode('utf-8'))
    pg.quit()
    quit()

# Check compute shader compilation status
if glGetShaderiv(compute_shader, GL_COMPILE_STATUS) != GL_TRUE:
    print("Compute shader compilation failed")
    err = glGetShaderInfoLog(compute_shader)
    print(err.decode('utf-8'))
    pg.quit()
    quit()

glAttachShader(shader_program, vertex_shader)
glAttachShader(shader_program, fragment_shader)
glAttachShader(compute_program, compute_shader)

glLinkProgram(shader_program)
glLinkProgram(compute_program)

glValidateProgram(compute_program)

# Check shader program linking status
if glGetProgramiv(shader_program, GL_LINK_STATUS) != GL_TRUE:
    print("Shader program linking failed")
    print(glGetProgramInfoLog(shader_program))
    pg.quit()
    quit()

if glGetProgramiv(compute_program, GL_LINK_STATUS) != GL_TRUE:
    print("Compute program linking failed")
    print(glGetProgramInfoLog(compute_program))
    pg.quit()
    quit()

glDispatchCompute(shader_program)

# textures
trailmap_ID = glGenTextures(1)
trailmapDiff_ID = glGenTextures(1)
display_texture_ID = glGenTextures(1)

Tcolour = (0, 0, 0)
texture_data = np.zeros((width, height, 3), dtype=np.uint8)
texture_data[:,:] = Tcolour
texture_data_bytes = texture_data.tobytes()

glBindTexture(GL_TEXTURE_2D, trailmap_ID)
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, texture_data)
glBindTexture(GL_TEXTURE_2D, trailmapDiff_ID)
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, texture_data)
glBindTexture(GL_TEXTURE_2D, display_texture_ID)
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, texture_data)

# Create a texture to store the output of the compute shader
output_texture_ID = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D, output_texture_ID)
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
glBindImageTexture(1, output_texture_ID, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F)

# Create a Shader Storage Buffer Object (SSBO) to pass 'agents' data to the compute shader
agents_buffer = glGenBuffers(1)
glBindBuffer(GL_SHADER_STORAGE_BUFFER, agents_buffer)
glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(AgentStruct) * numAgents, None, GL_DYNAMIC_DRAW)
glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, agents_buffer)

glBindImageTexture(0, trailmap_ID, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F)

fbo = glGenFramebuffers(1)
glBindFramebuffer(GL_FRAMEBUFFER, fbo)
glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, display_texture_ID, 0)

# Check if the FBO is complete
if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
    print("Framebuffer is not complete")
    pg.quit()
    quit()

glBindFramebuffer(GL_FRAMEBUFFER, 0)

agents = [Agent() for _ in range(numAgents)]
for i in range(numAgents):
    centre = Vec2(width/2,height/2)
    random_angle = random.uniform(0.0, 2.0 * math.pi)
    random_offset = Vec2(random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0))
    startPos = centre + random_offset * height * 0.15
    agents[i] = Agent(startPos, random_angle)

agents_structs = [agent for agent in agents]
agent_size = sizeof(AgentStruct)
agents_structs_bytes = b"".join(agent.to_struct() for agent in agents_structs)
glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(AgentStruct) * numAgents, agents_structs_bytes)

if __name__ == "__main__":
    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                glDeleteShader(vertex_shader)
                glDeleteShader(fragment_shader)
                glDeleteShader(compute_shader)
                glDeleteFramebuffers(1, [fbo])
                glDeleteTextures(1, [trailmap_ID, trailmapDiff_ID, display_texture_ID, output_texture_ID])
                glDeleteBuffers(1, [agents_buffer])
                pg.quit()
                quit()

        glUseProgram(compute_program)
        glDispatchCompute(numAgents, 1, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        # Render the result to the display
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        glUseProgram(shader_program)
        glClear(GL_COLOR_BUFFER_BIT)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        
        pg.display.flip()
 
