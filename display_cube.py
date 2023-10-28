import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import gluPerspective
import numpy as np
from itertools import combinations
from physics import *
import math
# Camera variables
angle_x = 0
angle_y = 0
mouse_dragging = False
last_mouse_x, last_mouse_y = 0, 0
camera_distance = 10  # Adjust this for initial zoom level
camera_translation = [0, 0]  # Translation offsets for panning


massLocations = [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1), (0, 0, 2), (0, 1, 2), (1, 0, 2), (1, 1, 2)]
massValues = [1, 1, 1, 1, 1, 1, 1, 1]
masses = generateMasses(massLocations, massValues)


class Cube:
    def __init__(self):
        self.vertices = self.update_vertices(masses)
        self.vertex_size = np.zeros((len(masses), 1))
        # self.edges = self.create_edges(springs)

    def update_vertices(self, masses):
        masses = masses
        self.vertices = masses[:, 3, :]
        # print("vertices: ", self.vertices)
        self.vertex_sizes = masses[:, 0, 0]/20
        # print("vertex_sizes ", self.vertex_sizes)


    def create_edges(self, springs):
        springs = springs
        edges = []
        for spring in springs:
            edges.append([spring[0], spring[1]])
        self.edges = edges
        # print("edges: ", self.edges)

def draw_checkered_ground(size, squares):
    half_size = size / 2
    square_size = size / squares

    for x in range(squares):
        for y in range(squares):  # Changed z to y
            # Determine the color
            if (x + y) % 2 == 0:
                glColor3f(0.5, 0.5, 0.5)  # Light gray
            else:
                glColor3f(0.9, 0.9, 0.9)  # Dark gray

            # Draw the square
            glBegin(GL_QUADS)
            glVertex3f(-half_size + x * square_size, -half_size + y * square_size, 0)  # Adjusted z to 0
            glVertex3f(-half_size + x * square_size, -half_size + (y+1) * square_size, 0)  # Adjusted z to 0
            glVertex3f(-half_size + (x+1) * square_size, -half_size + (y+1) * square_size, 0)  # Adjusted z to 0
            glVertex3f(-half_size + (x+1) * square_size, -half_size + y * square_size, 0)  # Adjusted z to 0
            glEnd()


def draw_cube(cube):
    glColor3f(0, 0, 1)  # Set color to blue
    glLineWidth(5)  # Set line width to 5
    glBegin(GL_LINES)
    for edge in cube.edges:
        for vertex in edge:
            # print("vertex: ", cube.vertices[int(vertex)])
            glVertex3fv(cube.vertices[int(vertex)])
    glEnd()

def draw_shadow(cube):
    glColor3f(0.3, 0.3, 0.3)
    glLineWidth(5)  # Set line width to 5
    glBegin(GL_LINES)
    for edge in cube.edges:
        for vertex in edge:
            point = cube.vertices[int(vertex)].copy()
            point[2] = 0
            glVertex3fv(point)
    glEnd()

def draw_spheres_at_vertices(cube):
    glColor3f(1, 0, 0)  # Color of the spheres
    for i in range(len(cube.vertices)):
        glPushMatrix()
        glTranslatef(*(cube.vertices[i]))
        glutSolidSphere(cube.vertex_sizes[i], 20, 20)  # Draw a sphere of radius 0.1 with 20 slices and 20 stacks
        glPopMatrix()

def mouse_button_callback(event):
    global mouse_dragging, last_mouse_x, last_mouse_y, camera_distance

    if event.type == pygame.MOUSEBUTTONDOWN:
        if event.button == 1:  # Left button press for rotation
            mouse_dragging = "DRAG"
            last_mouse_x, last_mouse_y = event.pos
        elif event.button == 3:  # Right button press for panning
            mouse_dragging = "PAN"
            last_mouse_x, last_mouse_y = event.pos
        elif event.button == 4:  # Mouse wheel up
            camera_distance -= 1.0
        elif event.button == 5:  # Mouse wheel down
            camera_distance += 1.0

    elif event.type == pygame.MOUSEBUTTONUP:
        if event.button in [1, 3]:  # Left or right button release
            mouse_dragging = False

def mouse_motion_callback(event):
    global angle_x, angle_y, last_mouse_x, last_mouse_y, camera_translation

    dx = event.pos[0] - last_mouse_x
    dy = event.pos[1] - last_mouse_y

    if mouse_dragging == "DRAG":  # Rotation
        angle_x += dy * 0.5
        angle_y += dx * 0.5
    elif mouse_dragging == "PAN":  # Panning
        camera_translation[0] += dx * 0.05
        camera_translation[1] -= dy * 0.05

    last_mouse_x, last_mouse_y = event.pos
    # ... [rest of the code remains unchanged]


def main():
    massLocations = [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1), (0, 0, 2), (0, 1, 2), (1, 0, 2), (1, 1, 2)]
    massValues = [1, 1, 1, 1, 1, 1, 1, 1]



    all_combinations = list(combinations(range(8), 2))
    springs = np.array([[comb[0], comb[1], 10000, np.linalg.norm(np.array(massLocations[comb[0]]) - np.array(massLocations[comb[1]]))] for comb in all_combinations])



    masses = generateMasses(massLocations, massValues)
    masses = torch.tensor(masses, dtype=torch.float)
    springs = torch.tensor(springs, dtype=torch.float)


    cube = Cube()
    cube.update_vertices(masses.numpy())
    cube.create_edges(springs.numpy())

    dt = 0.001
    T = 0
    N = masses.size(0)
    netForces = torch.zeros((N, 3))
    omega = 20

    og = springs[:, 3].clone()
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -12) # Adjusted to have a top-down view
    # Initialization of Masses and Springs
    
    while True:
        # springs[:, 3] = og + 0.15 * torch.sin(torch.tensor(T*omega))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            mouse_button_callback(event)
            if mouse_dragging:
                mouse_motion_callback(event)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
        glTranslatef(camera_translation[0], camera_translation[1], -camera_distance)
        glRotatef(angle_x, 1, 0, 0)
        glRotatef(angle_y, 0, 0, 1)

        netForces = 0
        netForces = netForces + compute_net_spring_forces(masses, springs) # Add spring forces
        netForces = netForces + computeGravityForces(masses) # Add gravity
        netForces = netForces + computeGroundCollisionForces(masses)
        # Integration Step
        # Calculate Acceleration
        masses[:, 1] = torch.div(netForces, masses[:, 0, 0].unsqueeze(-1))
        # Calculate Velocity
        masses[:, 2] = masses[:, 2] + masses[:, 1] * dt
        # Calculate Position
        masses[:, 3] = masses[:, 3] + masses[:, 2] * dt

        # Apply dampening
        masses[:, 2] = masses[:, 2] * 0.999
        
        draw_checkered_ground(20, 10)
        draw_shadow(cube)


        # Lift and rotate the cube

        draw_cube(cube)
        draw_spheres_at_vertices(cube)
        
        
        netForces = 0
        T += dt
        pygame.display.flip()
        pygame.time.wait(10)

if __name__ == "__main__":
    main()
