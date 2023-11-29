import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import gluPerspective
import numpy as np
from itertools import combinations
from physics import *
import pickle
import math
import time

# Camera variables
device = "cuda:0" if torch.cuda.is_available() else "cpu"
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
        mu_k = 0.5 # Kinetic friction coefficient

        # Compute forces
        netForces = compute_net_spring_forces(self.masses, self.springs)  # Spring forces
        netForces += computeGravityForces(self.masses)  # Gravity forces
        groundCollisionForces = computeGroundCollisionForces(self.masses)
        netForces += groundCollisionForces  # Ground collision forces
        # Compute friction forces and apply only to the masses at or below ground level
        frictionForces = computeFrictionForces(self.masses, netForces, groundCollisionForces, mu_s, mu_k)
        ground_indices = (self.masses[:, 3, 2] <= 0)

        # Update net forces with friction forces for ground-contacting masses
        netForces[ground_indices, :2] = frictionForces[ground_indices, :2]
        # print(netForces)
        # Integration step
        # Calculate acceleration
        self.masses[:, 1] = netForces / self.masses[:, 0, 0].unsqueeze(-1)
        # Calculate velocity
        self.masses[:, 2] += self.masses[:, 1] * dt
        # Calculate position
        self.masses[:, 3] += self.masses[:, 2] * dt


        # Apply dampening
        self.masses[:, 2] = self.masses[:, 2] * 0.999
        # print(self.masses[:, 2])
    # Update spring properties in-place according to material

    def updateSprings(self, w, T):
        # Update spring constant
        # print("materials: ", self.materials)
        # print("springs: ", self.springs.shape)
        self.springs[self.materials == 1, 2] = 1000
        self.springs[self.materials == 2, 2] = 2000
        self.springs[self.materials == 3, 2] = 5000
        self.springs[self.materials == 4, 2] = 5000
        # Update resting lengths
        self.springs[self.materials == 1, 3] = self.og[self.materials == 1]
        self.springs[self.materials == 2, 3] = self.og[self.materials == 2]
        self.springs[self.materials == 3, 3] = self.og[self.materials == 3] * (1 + 0.25 * np.sin(w*T))
        self.springs[self.materials == 4, 3] = self.og[self.materials == 4] * (1 + 0.25 * np.sin(w*T+torch.pi))

def concatenate_masses_and_springs(masses, springs, n_copies):
    # Check if n_copies is valid
    if n_copies < 1:
        raise ValueError("Number of copies should be at least 1")

    # Initialize with the original masses and springs
    concatenated_masses = masses.clone()
    concatenated_springs = springs.clone()

    num_masses = masses.shape[0]
    # print(num_masses)
    for i in range(1, n_copies):
        # Update indices for springs
        new_springs = springs.clone()
        # print(i * num_masses)
        new_springs[:, :2] = new_springs[:, :2] + (num_masses*i)
        # print("NEW SPRINGS: ", new_springs )
        # Concatenate masses and springs
        concatenated_masses = torch.cat([concatenated_masses, masses], dim=0)
        concatenated_springs = torch.cat([concatenated_springs, new_springs], dim=0)


    return concatenated_masses, concatenated_springs

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

def makeOneWorm():
    massLocations = [(0, 0, 0), #0
                     (0, 1, 0), #1
                     (0, 2, 0), #2
                     (1, 0, 0), #3
                     (1, 1, 0), #4
                     (1, 2, 0), #5
                     (0, 0, 1), #6
                     (0, 1, 1), #7
                     (0, 2, 1), #8
                     (1, 0, 1), #9
                     (1, 1, 1), #10
                     (1, 2, 1)]
    # massLocations = [(x, y, z + 2) for x, y, z in massLocations]

    massValues = [1] * len(massLocations)
    # print(massValues)

    lefthip_masses = [0, 1, 3, 4, 6, 7, 9, 10]
    lefthip_springs = generateSprings(massLocations, lefthip_masses)

    righthip_masses = [2, 1, 4, 5, 7, 8, 10, 11]
    righthip_springs = generateSprings(massLocations, righthip_masses)
    


   
    masses = generateMasses(massLocations, massValues)

    springs = np.concatenate((lefthip_springs,righthip_springs), axis=0)

    
    
    masses = torch.tensor(masses, dtype=torch.float)
    springs = torch.tensor(springs, dtype=torch.float)


    return masses, springs

def make_multilayer_sphere(radius, num_masses_per_layer, num_layers=1):
    massLocations = []
    # angle_step = 2 * np.pi / num_masses_per_layer
    layer_radii = np.linspace(0.5 * radius, radius, num_layers)  # Radii of each spherical layer
    print("layerradii ", layer_radii)

    # Calculate positions for each spherical layer
    for layer, layer_radius in enumerate(layer_radii):
        # Add top and bottom masses (poles)
        massLocations.append((0, 0, layer_radius))  # Top (North Pole)
        massLocations.append((0, 0, -layer_radius)) # Bottom (South Pole)

        # Evenly distribute other masses
        for lat in range(1, num_masses_per_layer - 1):  # Avoid poles
            phi = lat * (np.pi / num_masses_per_layer)  # Angle from z-axis
            for lon in range(num_masses_per_layer):
                theta = lon * (2 * np.pi / num_masses_per_layer)  # Angle in xy-plane
                x = layer_radius * np.sin(phi) * np.cos(theta)
                y = layer_radius * np.sin(phi) * np.sin(theta)
                z = layer_radius * np.cos(phi)
                massLocations.append((x, y, z))
    massLocations = [(x, y, z + radius) for x, y, z in massLocations]
    # Generate springs between adjacent masses and between layers
    springs = []
    spring_constant = 10000

    # Connect adjacent masses in the same layer
    for layer in range(num_layers):
        base_index = layer * num_masses_per_layer * num_masses_per_layer
        for lat in range(num_masses_per_layer):
            for lon in range(num_masses_per_layer-1):
                current_index = base_index + lat * num_masses_per_layer + lon
                # Connect with next mass in the same latitude
                next_lon_index = base_index + lat * num_masses_per_layer + (lon + 1) % num_masses_per_layer
                restinglength = np.linalg.norm(np.array(massLocations[current_index]) - np.array(massLocations[next_lon_index]))
                springs.append((current_index, next_lon_index, spring_constant, restinglength))  # Resting length to be calculated
                # Connect with next mass in the same longitude
                restinglength = np.linalg.norm(np.array(massLocations[current_index]) - np.array(massLocations[(current_index + num_masses_per_layer) % (num_masses_per_layer * num_masses_per_layer)]))
                next_lat_index = base_index + ((lat + 1) % num_masses_per_layer) * num_masses_per_layer + lon
                springs.append((current_index, next_lat_index, spring_constant, restinglength))  # Resting length to be calculated

    # Connect masses between layers
    # [This part of the code would need to be carefully written to ensure proper connections between layers]

    massValues = [1] * len(massLocations)
    masses = generateMasses(massLocations, massValues)
    print("Masses len: ", len(masses))
    masses = torch.tensor(masses, dtype=torch.float)
    springs = torch.tensor(springs, dtype=torch.float)

    return masses, springs


def simulate(popCenterLocs, popCenterMats, visualize=False):
    '''
        materials
        1: k=1000 b=c=0
        2: k=20000 b=c=0
        3: k=5000 b=0.25 c=0
        4: k=5000 b=0.25 c=pi
        w=2*pi
    '''
    # print("Pop device: ", popCenterLocs.device)
    populationSize = popCenterLocs.size()[0]
    # Example usage
    radius = 4  # Radius of the sphere
    num_masses_per_level = 8  # Number of masses per level
    masses, springs = make_multilayer_sphere(radius, num_masses_per_level)
    # masses, springs = makeOneWorm()
    masses, springs = concatenate_masses_and_springs(masses, springs, populationSize)
    masses = masses.to(device)
    springs = springs.to(device)
    # print("spring", len(springs))
    # print("dog1: ", springs[:len(springs)//2])
    # print("dog2: ", springs[len(springs)//2:])

    materials = assignMaterials(masses, springs, popCenterLocs, popCenterMats) # torch.randint(1, 4, size=(springs.size()[0],))
    # print(materials.size())
    dog = (MassSpringSystem(masses, springs, materials))

    initial_positions = dog.masses[::36, 3, :].clone()
    # print("Materials:\n\n", materials)
    # print("Springs:\n\n", springs)

    # exit(1)
    # print(springs.size())
    # print(materials.size())
    w = 2*np.pi
    # og = springs[:, 3].clone()
    # print(og)
    
    dt = 0.001
    T = 0
    N = masses.size(0)
    netForces = torch.zeros((N, 3))
    omega = 20

    # og = springs[:, 3].clone()
    if visualize:
        pygame.init()
        display = (800, 600)
        pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
        gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
        glTranslatef(0.0, 0.0, -12) # Adjusted to have a top-down view
    # Initialization of Masses and Springs

    # print(len(objs))
    movingAverage = []
    while T < 5:
        # print("T: ", T)
        start = time.time()
        if visualize:
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
        
        
            draw_checkered_ground(100, 100)

        
        dog.updateSprings(w, T)
        dog.simulate(dt)
        if visualize:
            draw_shadow(dog)
            draw_cube(dog)
            # draw_cube_faces(cube)
            draw_spheres_at_vertices(dog)
        

        T += dt

        if visualize:
            pygame.display.flip()
            pygame.time.wait(1)
        
        end = time.time()
        movingAverage.append(end - start)
        # print(sum(movingAverage) / len(movingAverage))
        
        if int(T*1000) % 1000 == 0:
           distances = torch.norm(dog.masses[::36, 3, :][:, :2] - initial_positions[:, :2], dim=1)
        #    print(distances)

    final_positions = dog.masses[::36, 3, :].clone()
    distances = torch.norm(final_positions[:, :2] - initial_positions[:, :2], dim=1)
    #print(distances)
    return distances

if __name__ == "__main__":
    
    with open("best_robot.pkl", 'rb') as f:
        bestBot = pickle.load(f)

    # with open("best_robot_rs.pkl", 'rb') as f:
    #     rsBot = pickle.load(f)

    # print(bestBot)
    # rsBot_loc = torch.tensor(rsBot[0]).unsqueeze(0).to(device)
    # rsBot_mat = torch.tensor(rsBot[1]).unsqueeze(0).to(device)
    # print(bestBot[1])
    # bestBot = (np.array([[0.5, 0, 0], [0.5, 2, 0]]), np.array([[1], [2]]))
    popCenterLocs = torch.tensor(bestBot[0]).unsqueeze(0).to(device)
    # popCenterLocs = torch.concat([popCenterLocs, rsBot_loc], axis=0)

    popCenterMats = torch.tensor(bestBot[1]).unsqueeze(0).to(device)
    # popCenterMats = torch.concat([popCenterMats, rsBot_mat], axis=0)

    # print("Size: ", popCenterLocs.size(), popCenterMats.size())
    simulate(popCenterLocs, popCenterMats, visualize=True)