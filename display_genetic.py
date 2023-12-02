import itertools
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
import argparse
from display_dog import makeOneDog
# Camera variables
device = "cuda:0" if torch.cuda.is_available() else "cpu"
angle_x = 0
angle_y = 0
mouse_dragging = False
last_mouse_x, last_mouse_y = 0, 0
camera_distance = 10  # Adjust this for initial zoom level
camera_translation = [0, 0]  # Translation offsets for panning



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
        if int(edge[2]) != 5:
            edge_vertices = edge[:2]
            for vertex in edge_vertices:
                # print("vertex: ", cube.vertices[int(vertex)])
                glVertex3fv(cube.vertices[int(vertex)].numpy())
    glEnd()

def draw_shadow(cube):
    glColor3f(0.3, 0.3, 0.3)
    glLineWidth(5)  # Set line width to 5
    glBegin(GL_LINES)
    for edge in cube.edges:
        # print("edge", edge)
        if int(edge[2]) != 5:
            edge_vertices = edge[:2]
            # print("edge_vertices", edge_vertices)
            for vertex in edge_vertices:
                point = cube.vertices[int(vertex)].clone()
                point[2] = 0
                # print(point)
                glVertex3fv(point.numpy())
    glEnd()

def draw_spheres_at_vertices(cube):
      # Color of the spheres
    for i in range(len(cube.vertices)):
        glPushMatrix()
        glTranslatef(*(cube.vertices[i]))
        #change color if z<0
        if cube.vertices[i][2] < 0:
            glColor3f(0, 1, 0)
        else:
            glColor3f(1, 0, 0)
        glutSolidSphere(cube.vertex_sizes[i], 20, 20)  # Draw a sphere of radius 0.1 with 20 slices and 20 stacks
        glPopMatrix()


def mouse_button_callback(event):
    global mouse_dragging, last_mouse_x, last_mouse_y, camera_distance, shift_pressed

    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
            shift_pressed = True

    if event.type == pygame.KEYUP:
        if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
            shift_pressed = False

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
    global angle_x, angle_y, last_mouse_x, last_mouse_y, camera_translation, shift_pressed

    dx = event.pos[0] - last_mouse_x
    dy = event.pos[1] - last_mouse_y

    if shift_pressed:
        # Panning logic
        camera_translation[0] += dx * 0.05
        camera_translation[1] -= dy * 0.05
    else:
        # Rotating around the center of mass
        angle_x += dy * 0.5
        angle_y += dx * 0.5

    last_mouse_x, last_mouse_y = event.pos
    
def calculate_center_of_mass(masses):
    # Assumes masses is a tensor with shape [N, 4, 3] where N is the number of masses
    # and each mass has x, y, z positions
    total_mass_position = torch.sum(masses[:, 3, :], dim=0)
    number_of_masses = masses.shape[0]
    center_of_mass = total_mass_position / number_of_masses
    return center_of_mass

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


def makeBoxes():
    massLocations = []
    springs = []

    # Define parameters for the worm
    num_cubes = 5  # Number of cubes in each dimension
    cube_size = 1  # Size of each cube

    # Function to add cube masses
    def addCubeMasses(x_base, y_base, z_base):
        cube_masses = []
        for x in range(2):
            for y in range(2):
                for z in range(2):
                    mass = (x_base + x * cube_size, y_base + y * cube_size, z_base + z * cube_size)
                    if mass not in massLocations:
                        massLocations.append(mass)
                    cube_masses.append(massLocations.index(mass))
        return cube_masses

    # Generate masses and springs for each cube
    for x in range(num_cubes):
        for y in range(num_cubes):
            for z in range(num_cubes):
                cube_mass_indices = addCubeMasses(x * cube_size, y * cube_size, z * cube_size)
                cube_springs = generateSprings(massLocations, cube_mass_indices)
                springs.extend(cube_springs)

    # Convert to tensors
    massValues = [1] * len(massLocations)  # Assuming each mass has a value of 1
    masses = torch.tensor(generateMasses(massLocations, massValues), dtype=torch.float)

    # print("Springs: ", springs)
    # print("Springs len: ", len(springs))

    # Remove any duplicate springs (i.e. springs that connect the same two masses)
    springs = np.unique(springs, axis=0)
    # print("Springs len: ", len(springs))
    springs = torch.tensor(springs, dtype=torch.float)

    return masses, springs


def make_multilayer_sphere(radius, num_masses_per_layer, num_layers=5):
    massLocations = []
    springs = []
    spring_constant = 10000

    # Generate mass locations for each layer
    for layer in range(num_layers):
        layer_radius = radius * (layer + 1) / num_layers

        # Even distribution excluding poles
        for lat in range(1, num_masses_per_layer - 1):  # Exclude the poles
            phi = lat * (np.pi / (num_masses_per_layer - 1))  # Angle from z-axis
            for lon in range(num_masses_per_layer):
                theta = lon * (2 * np.pi / num_masses_per_layer)
                x = layer_radius * np.sin(phi) * np.cos(theta)
                y = layer_radius * np.sin(phi) * np.sin(theta)
                z = layer_radius * np.cos(phi)
                massLocations.append((x, y, z))

    # Adjust mass locations for ground level
    massLocations = [(x, y, z + radius) for x, y, z in massLocations]

    # Connect masses within each layer and between layers
    for layer in range(num_layers):
        layer_base_index = layer * (num_masses_per_layer - 2) * num_masses_per_layer

        for lat in range(num_masses_per_layer - 2):
            for lon in range(num_masses_per_layer):
                current_index = layer_base_index + lat * num_masses_per_layer + lon

                # Connect with next mass in the same latitude (wrap-around)
                next_lon_index = layer_base_index + lat * num_masses_per_layer + (lon + 1) % num_masses_per_layer
                resting_length_lon = np.linalg.norm(np.array(massLocations[current_index]) - np.array(massLocations[next_lon_index]))
                springs.append((current_index, next_lon_index, spring_constant, resting_length_lon))

                # Connect with next mass in the same longitude
                if lat < num_masses_per_layer - 3:
                    next_lat_index = layer_base_index + (lat + 1) * num_masses_per_layer + lon
                    resting_length_lat = np.linalg.norm(np.array(massLocations[current_index]) - np.array(massLocations[next_lat_index]))
                    springs.append((current_index, next_lat_index, spring_constant, resting_length_lat))

        # Connect to corresponding masses in the next layer
        if layer < num_layers - 1:
            next_layer_base_index = (layer + 1) * (num_masses_per_layer - 2) * num_masses_per_layer
            for i in range((num_masses_per_layer - 2) * num_masses_per_layer):
                current_mass_index = layer_base_index + i
                next_layer_mass_index = next_layer_base_index + i
                resting_length_inter_layer = np.linalg.norm(np.array(massLocations[current_mass_index]) - np.array(massLocations[next_layer_mass_index]))
                springs.append((current_mass_index, next_layer_mass_index, spring_constant, resting_length_inter_layer))


    massValues = [1] * len(massLocations)
    masses = generateMasses(massLocations, massValues)
    print("Masses len: ", len(masses))
    print("Springs len: ", len(springs))
    # print("Springs: ", springs)
    masses = torch.tensor(masses, dtype=torch.float)
    springs = torch.tensor(springs, dtype=torch.float)

    return masses, springs

def makeOnePyramid():
    massLocations = []
    springs = []

    # Define parameters for the pyramid
    pyramid_height = 4  # Number of layers in the pyramid
    cube_size = 1       # Size of each cube

    # Function to add cube masses
    def addCubeMasses(x_base, y_base, z_base):
        cube_masses = []
        for x in range(2):
            for y in range(2):
                for z in range(2):
                    mass = (x_base + x * cube_size, y_base + y * cube_size, z_base + z * cube_size)
                    if mass not in massLocations:
                        massLocations.append(mass)
                    cube_masses.append(massLocations.index(mass))
        return cube_masses

    # Generate masses and springs for each cube in the pyramid
    for layer in range(pyramid_height):
        for x in range(pyramid_height - layer):
            for y in range(pyramid_height - layer):
                z = layer  # Height of the layer in the pyramid
                cube_mass_indices = addCubeMasses(x * cube_size, y * cube_size, z * cube_size)
                cube_springs = generateSprings(massLocations, cube_mass_indices)
                springs.extend(cube_springs)

    # Convert to tensors
    massValues = [1] * len(massLocations)  # Assuming each mass has a value of 1
    masses = torch.tensor(generateMasses(massLocations, massValues), dtype=torch.float)
    springs = np.unique(springs, axis=0)
    springs = torch.tensor(springs, dtype=torch.float)

    return masses, springs

shift_pressed = False
def simulate(popCenterLocs, popCenterMats, ogMasses, ogSprings, visualize=False):
    '''
        materials
        1: k=1000 b=c=0
        2: k=20000 b=c=0
        3: k=5000 b=0.25 c=0
        4: k=5000 b=0.25 c=pi
        5: k=b=c=0
        w=2*pi
    '''
    # print("Pop device: ", popCenterLocs.device)
    populationSize = popCenterLocs.size()[0]
    

    ogMassNum = ogMasses.size()[0]
    masses, springs = concatenate_masses_and_springs(ogMasses.clone(), ogSprings.clone(), populationSize)

    # print
    masses = masses.to(device)
    springs = springs.to(device)
    # print("spring", len(springs))
    # print("dog1: ", springs[:len(springs)//2])
    # print("dog2: ", springs[len(springs)//2:])

    materials = assignMaterials(masses, springs, popCenterLocs, popCenterMats) # torch.randint(1, 4, size=(springs.size()[0],))
    # print(materials.size())
    obj = (MassSpringSystem(masses, springs, materials))
    # print(obj.masses.size())
    initial_positions = obj.masses[::ogMassNum, 3, :].clone()
    # print("Initial Positions: ", initial_positions)
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
            center_of_mass = calculate_center_of_mass(obj.masses)
            if not shift_pressed:
                # Focus camera on center of mass
                camera_target_position = center_of_mass.numpy()
                glTranslatef(-camera_target_position[0], -camera_target_position[1], -camera_distance - camera_target_position[2])
                glRotatef(angle_x, 1, 0, 0)
                glRotatef(angle_y, 0, 0, 1)
            else:
                # Panning logic
                glTranslatef(camera_translation[0], camera_translation[1], -camera_distance)
                glRotatef(angle_x, 1, 0, 0)
                glRotatef(angle_y, 0, 0, 1)
            # print(cube.edges)
        
        
            draw_checkered_ground(100, 100)

        
        obj.updateSprings(w, T)
        obj.simulate(dt)
        if visualize:
            draw_shadow(obj)
            draw_cube(obj)
            # draw_cube_faces(cube)
            draw_spheres_at_vertices(obj)
        

        T += dt

        if visualize:
            pygame.display.flip()
            pygame.time.wait(1)
        
        end = time.time()
        movingAverage.append(end - start)
        # print(sum(movingAverage) / len(movingAverage))
        
        if int(T*10) % 10 == 0:
            distances = torch.abs(torch.max(obj.masses[::ogMassNum, 3, :][:, :2] - initial_positions[:, :2], dim=1).values)
            #print("Distances: ", distances)
        #    print(distances)

    final_positions = obj.masses[::ogMassNum, 3, :].clone()
   # print("Final Positions: ", final_positions)
    distances = torch.abs(torch.max(final_positions[:, :2] - initial_positions[:, :2], dim=1).values)
    #print(distances)
    return distances

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s","--shape", type=str, default="box", help="Starting shape")
    args = parser.parse_args()

    if args.shape == "sphere":
        filename = "box_best_robot.pkl"
        masses, springs = make_multilayer_sphere(3, 10, 5)

    elif args.shape == "pyramid":
        filename = "dog_best_robot.pkl"
        masses, springs = makeOnePyramid()

    elif args.shape == "box":
        filename = "pyramid_best_robot.pkl"
        masses, springs = makeBoxes()

    elif args.shape == "dog":
        filename = "sphere_best_robot.pkl"
        masses, springs = makeOneDog()
   
    with open(filename, 'rb') as f:
        bestBot = pickle.load(f)
    # with open("best_robot_rs.pkl", 'rb') as f:
    #     rsBot = pickle.load(f)

    # print(bestBot)
    # rsBot_loc = torch.tensor(rsBot[0]).unsqueeze(0).to(device)
    # rsBot_mat = torch.tensor(rsBot[1]).unsqueeze(0).to(device)
    print(bestBot[0])
    print(bestBot[1])
    # bestBot = (np.array([[0.5, 0, 0], [0.5, 2, 0]]), np.array([[1], [2]]))
    popCenterLocs = torch.tensor(bestBot[0]).unsqueeze(0).to(device)
    # popCenterLocs = torch.concat([popCenterLocs, rsBot_loc], axis=0)

    popCenterMats = torch.tensor(bestBot[1]).unsqueeze(0).to(device)
    # popCenterMats = torch.concat([popCenterMats, rsBot_mat], axis=0)

    # print("Size: ", popCenterLocs.size(), popCenterMats.size())

    radius = 2  # Radius of the sphere
    num_masses_per_level = 8  # Number of masses per level
    base_size = 1
    height = 1
    num_levels = 3
    
   

    simulate(popCenterLocs, popCenterMats, masses, springs, visualize=True)
