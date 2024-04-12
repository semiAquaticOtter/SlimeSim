import pygame as pg 
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

from dataclasses import dataclass

from ctypes import sizeof, c_float, Structure, byref, c_void_p

import numpy as np
import random
import math
import time
from tqdm import trange

from datetime import datetime, timedelta

#MARK: vars
width, height = 720, 480
displaySize = (width, height)
numAgents = 2500000 # 10*(10**3)
dimStrength = 1

scaleing = 1

lrw, lrh = int(width*scaleing), int(height*scaleing)

start_time = time.time()
previous_time = start_time

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
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
#SECTION - Start

print(f"""
      Generating {numAgents:,d} agents.
      This may take some time if theres a lot.
""")
# printProgressBar(0, numAgents, prefix = 'Progress:', suffix = 'Complete', length = 50)
agentsArray = []
for i in trange(numAgents):
    # x = random.uniform(0.0, 1.0)
    # y = random.uniform(0.0, 1.0)
    x = 0.5
    y = 0.5
    angle = 0
    agent = Agent2(x, y, angle)
    agentsArray.append(agent)
    # printProgressBar(i + 1, numAgents, prefix = 'Progress:', suffix = 'Complete', length = 50)

print("converting to numpy array")
npAgentsArray = np.array([(agent.x, agent.y, agent.angle) for agent in agentsArray], dtype=np.float32)

pg.init()
pg.display.set_mode(displaySize, OPENGL | DOUBLEBUF)

gl_version = glGetString(GL_VERSION)
print(gl_version.decode('utf-8'))

if pg.get_error() != "":
    print("Pygame error:", pg.get_error())
    pg.quit()
    quit()

#ANCHOR - Vertex Shader
print("compiling vertex shader")
vertexSource = open("./shader.vert", "r")
vertexShader = glCreateShader(GL_VERTEX_SHADER)
glShaderSource(vertexShader, vertexSource)
glCompileShader(vertexShader)

#ANCHOR - Fragment Shader
print("compiling fragment shader")
fragmentSource = open("./shader.frag", "r")
fragmentShader = glCreateShader(GL_FRAGMENT_SHADER)
glShaderSource(fragmentShader, fragmentSource)
glCompileShader(fragmentShader)

#ANCHOR - Compute Shader
print("compiling compute shader")
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
glTextureParameteri(trailmap, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
glTextureParameteri(trailmap, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
glTextureParameteri(trailmap, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
glTextureParameteri(trailmap, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

glTextureStorage2D(trailmap, 1, GL_RGBA32F, lrw, lrh)
glBindImageTexture(0, trailmap, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F)

def random_inside_unit_circle():
    while True:
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x**2 + y**2 <= 1:
            # Normalize the coordinates to [0, 1]
            magnitude = math.sqrt(x**2 + y**2)
            x_normalized = (x + 1) / 2
            y_normalized = (y + 1) / 2
            return Vec2(x_normalized, y_normalized)

#ANCHOR - Create agent array
# agentsArray = [Agent2(random.uniform(0.2, 0.8), random.uniform(0.2, 0.8), random.uniform(0, 2 * math.pi)) for _ in range(numAgents)]
# npAgentsArray = np.array([(agent.x, agent.y, agent.angle) for agent in agentsArray], dtype=np.float32)

ssbo = glGenBuffers(1)
glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo)
glBufferData(GL_SHADER_STORAGE_BUFFER, len(npAgentsArray) * sizeof(GLfloat) * 3, npAgentsArray, GL_DYNAMIC_DRAW)

ssbo_bindPoint = 1
glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ssbo_bindPoint, ssbo)

#ANCHOR - Triangles?
vertices = np.array([
    # Vertex positions (x, y) followed by texture coordinates (u, v)
    -1.0, -1.0, 0.0, 0.0,
    -1.0,  1.0, 0.0, 1.0,
     1.0,  1.0, 1.0, 1.0, 
    -1.0, -1.0, 0.0, 0.0,
     1.0,  1.0, 1.0, 1.0,
     1.0, -1.0, 1.0, 0.0
], dtype=np.float32)

quad_vao = glGenVertexArrays(1)
glBindVertexArray(quad_vao)

quad_vbo = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, quad_vbo)
glBufferData(GL_ARRAY_BUFFER, vertices, GL_DYNAMIC_DRAW)

glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), None)
glEnableVertexAttribArray(0)

glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), c_void_p(2 * sizeof(GLfloat)))
glEnableVertexAttribArray(1)

#!SECTION - Start

def check_gl_error():
    error_code = glGetError()
    if error_code != GL_NO_ERROR:
        print(f"OpenGL error: {error_code}")

# print(npAgentsArray)

if __name__ == "__main__":
    clock = pg.time.Clock()
    while True:
        t = clock.tick(60)
        dt = t/1000
        for event in pg.event.get():
            if event.type == pg.QUIT:
                #NOTE - exit stuff
                # glDeleteBuffers(1, [ssbo])
                pg.quit()
                quit()
            elif event.type == pg.KEYDOWN and event.key == K_ESCAPE:
                pg.quit()
                quit()

        glUseProgram(compute_program)
        # glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ssbo_bindPoint, ssbo)
        glUniform1i(glGetUniformLocation(compute_program, "numAgents"), numAgents)
        glUniform1i(glGetUniformLocation(compute_program, "width"), width)
        glUniform1i(glGetUniformLocation(compute_program, "height"), height)
        glUniform1f(glGetUniformLocation(compute_program, "deltaTime"), dt)
        glUniform1f(glGetUniformLocation(compute_program, "dimStrength"), dimStrength)
        glDispatchCompute(width, height, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo)
        # glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, len(npAgentsArray) * sizeof(GLfloat) * 3, npAgentsArray)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0, 0, lrw, lrh)
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

        # print(delta_time)
        # print("Buffer Data:", np.frombuffer(npAgentsArray, dtype=np.float32))

        check_gl_error()
        pg.display.flip()

        