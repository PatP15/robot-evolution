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


class MassSpringSystem:
    def __init__(self, masses, springs, materials):
        # print("init: ", springs)
        self.masses = masses.clone()
        self.springs = springs.clone()
        self.materials = materials.clone()
        self.og = springs[:, 3].clone()
        self.update_vertices()
        self.create_edges()

        # self.masses = masses.clone()
        # print("init edges: ", self.edges)

    def update_vertices(self):
        self.vertices = self.masses[:, 3, :]
        # print("vertices: ", self.vertices)
        self.vertex_sizes = self.masses[:, 0, 0]/20
        # print("vertex_sizes ", self.vertex_sizes)

    def create_edges(self):
        # print("create_edges: ", springs)
        # print(self.springs[:, :2])
        self.edges = self.springs[:, :2]  # Get the first two columns which are the vertex indices for each spring
        # print("edges: ", self.edges)


    def simulate(self, dt):
        mu_s = 1  # Static friction coefficient
        mu_k = 0.8 # Kinetic friction coefficient

        # Compute forces
        netForces = compute_net_spring_forces(self.masses, self.springs)  # Spring forces
        netForces += computeGravityForces(self.masses)  # Gravity forces
        groundCollisionForces = computeGroundCollisionForces(self.masses)
        netForces += groundCollisionForces  # Ground collision forces
        # print(netForces)
        # Compute friction forces and apply only to the masses at or below ground level
        # Compute friction forces and apply only to the masses at or below ground level
        frictionForces = computeFrictionForces(self.masses, netForces, groundCollisionForces, mu_s, mu_k)
        ground_indices = (self.masses[:, 3, 2] <= 0)

        # Update net forces with friction forces for ground-contacting masses
        netForces[ground_indices, :2] += frictionForces[ground_indices, :2]

        # Integration step
        # Calculate acceleration
        self.masses[:, 1] = netForces / self.masses[:, 0, 0].unsqueeze(-1)
        # Calculate velocity
        self.masses[:, 2] += self.masses[:, 1] * dt
        # Calculate position
        self.masses[:, 3] += self.masses[:, 2] * dt


        # Apply dampening
        self.masses[:, 2] = self.masses[:, 2] * 0.999

    # Update spring properties in-place according to material

    def updateSprings(self, w, T):
        # Update spring constant
        self.springs[self.materials == 1, 2] = 1000
        self.springs[self.materials == 2, 2] = 2000
        self.springs[self.materials == 3, 2] = 5000
        self.springs[self.materials == 4, 2] = 5000
        # Update resting lengths
        self.springs[self.materials == 1, 3] = self.og[self.materials == 1]
        self.springs[self.materials == 2, 3] = self.og[self.materials == 2]
        self.springs[self.materials == 3, 3] = self.og[self.materials == 3] * (1 + 0.25 * np.sin(w*T))
        self.springs[self.materials == 4, 3] = self.og[self.materials == 4] * (1 + 0.25 * np.sin(w*T+torch.pi))


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
            glVertex3fv(cube.vertices[int(vertex)].numpy())
    glEnd()

def draw_shadow(cube):
    glColor3f(0.3, 0.3, 0.3)
    glLineWidth(5)  # Set line width to 5
    glBegin(GL_LINES)
    for edge in cube.edges:
        for vertex in edge:
            point = cube.vertices[int(vertex)].clone()
            point[2] = 0
            # print(point)
            glVertex3fv(point.numpy())
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

def generateSprings(massLocations, massIdxs):
    numMasses = len(massIdxs)
    all_combinations = list(combinations(range(numMasses), 2))
    springs = np.array([[massIdxs[comb[0]], massIdxs[comb[1]], 10000, np.linalg.norm(np.array(massLocations[massIdxs[comb[0]]]) - np.array(massLocations[massIdxs[comb[1]]]))] for comb in all_combinations])
    # springs = np.delete(springs, massIdxs, axis=0)
    return springs


def main():
    '''
        materials
        1: k=1000 b=c=0
        2: k=20000 b=c=0
        3: k=5000 b=0.25 c=0
        4: k=5000 b=0.25 c=pi
        w=2*pi
    '''
    massLocations = [(0, 0, 0),
                     (0, 1, 0),
                     (0, 3, 0),
                     (0, 4, 0),
                     (1, 0, 0),
                     (1, 1, 0),
                     (1, 3, 0),
                     (1, 4, 0),
                     (0, 0, -1),
                     (0, 1, -1),
                     (0, 3, -1),
                     (0, 4, -1),
                     (1, 0, -1),
                     (1, 1, -1),
                     (1, 3, -1),
                     (1, 4, -1),
                     (0.5, 0.5, -2),
                     (0.5, 3.5, -2)]
    massLocations = [(x, y, z + 2) for x, y, z in massLocations]

    massValues = [1] * 36
    # print(massValues)

    lefthip_masses = [0, 1, 4, 5, 8, 9, 12, 13]
    lefthip_springs = generateSprings(massLocations, lefthip_masses)
    
    middle_masses = [1,2,5,6,9,10,13,14]
    middle_springs = generateSprings(massLocations, middle_masses)

    righthip_masses = [2,3,6,7,10,11,14,15]
    righthip_springs = generateSprings(massLocations, righthip_masses)

    frontlegs = np.array([
        [12, 16, 10000, math.sqrt(1.5)],
        [13, 16, 10000, math.sqrt(1.5)],
        [8, 16, 10000, math.sqrt(1.5)],
        [9, 16, 10000, math.sqrt(1.5)],
        [10, 17, 10000, math.sqrt(1.5)],
        [11, 17, 10000, math.sqrt(1.5)],
        [14, 17, 10000, math.sqrt(1.5)],
        [15, 17, 10000, math.sqrt(1.5)],
    ])
    backlegs = frontlegs.copy()
    backlegs[:, :2] += 18
    # print(backlegs)
    # all_combinations = list(combinations(range(18), 2))
    # springs = np.array([[comb[0], comb[1], 10000, np.linalg.norm(np.array(massLocations[comb[0]]) - np.array(massLocations[comb[1]]))] for comb in all_combinations])

    # Front half of dog
    og = massLocations.copy()
    for x,y,z in og:
        massLocations.append((x +4, y, z))

    # print(len(massLocations))
    lefthip_masses = np.array([0, 1, 4, 5, 8, 9, 12, 13]) + 18
    lefthip2_springs = generateSprings(massLocations, lefthip_masses)
    
    middle_masses = np.array([1,2,5,6,9,10,13,14]) + 18
    middle2_springs = generateSprings(massLocations, middle_masses)

    righthip_masses = np.array([2,3,6,7,10,11,14,15]) + 18
    righthip2_springs = generateSprings(massLocations, righthip_masses)

    torso_masses = np.array([1, 2, 9, 10])
    torso_masses = np.concatenate((torso_masses, torso_masses + 18))
    torso_springs = generateSprings(massLocations, torso_masses)

    masses = generateMasses(massLocations, massValues)

    springs = np.concatenate((lefthip_springs, middle_springs, righthip_springs, frontlegs, 
                              lefthip2_springs, middle2_springs, righthip2_springs, backlegs, torso_springs), axis=0)

    grid_dimensions = (1, 1)
    spacing = 3 # adjust this value for the distance between cubes in the grid

    objs = []

    for i in range(grid_dimensions[0]):
        for j in range(grid_dimensions[1]):
        
            masses = torch.tensor(masses, dtype=torch.float)
            springs = torch.tensor(springs, dtype=torch.float)
            materials = torch.randint(1, 4, size=(springs.size()[0],))
            objs.append(MassSpringSystem(masses, springs, materials))

    
    # print(springs.size())
    # print(materials.size())
    w = 2*np.pi
    # og = springs[:, 3].clone()
    # print(og)
    
    dt = 0.002
    T = 0
    N = masses.size(0)
    netForces = torch.zeros((N, 3))
    omega = 20

    # og = springs[:, 3].clone()
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -12) # Adjusted to have a top-down view
    # Initialization of Masses and Springs

    # print(len(objs))
    while True:
        
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
        # print(cube.edges)
        
        
        draw_checkered_ground(30, 30)
        # print(cube.edges)
        # print(len(objs))
        for obj in objs:
            obj.updateSprings(w, T)
            obj.simulate(dt)
            draw_shadow(obj)

        for obj in objs:

            draw_cube(obj)
            # draw_cube_faces(cube)
            draw_spheres_at_vertices(obj)
        
        
        

        T += dt
        pygame.display.flip()
        pygame.time.wait(1)

if __name__ == "__main__":
    main()