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

def concatenate_masses_and_springs(masses, springs, n_copies):
    # Check if n_copies is valid
    if n_copies < 1:
        raise ValueError("Number of copies should be at least 1")

    # Initialize with the original masses and springs
    concatenated_masses = masses.clone()
    concatenated_springs = springs.clone()

    num_masses = masses.shape[0]
    print(num_masses)
    for i in range(1, n_copies):
        # Update indices for springs
        new_springs = springs.clone()
        print(i * num_masses)
        new_springs[:, :2] = new_springs[:, :2] + (num_masses*i)
        print("NEW SPRINGS: ", new_springs )
        # Concatenate masses and springs
        concatenated_masses = torch.cat([concatenated_masses, masses], dim=0)
        concatenated_springs = torch.cat([concatenated_springs, new_springs], dim=0)


    return concatenated_masses, concatenated_springs



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
        massLocations.append((x + 4, y, z))

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

    
    
    masses = torch.tensor(masses, dtype=torch.float)
    springs = torch.tensor(springs, dtype=torch.float)

    masses, springs = concatenate_masses_and_springs(masses, springs, 10)

    materials = torch.randint(1, 4, size=(springs.size()[0],))
    dog = (MassSpringSystem(masses, springs, materials))

    
    # print(springs.size())
    # print(materials.size())
    w = 2*np.pi
    # og = springs[:, 3].clone()
    # print(og)
    
    dt = 0.001
    T = 0
    N = masses.size(0)

    # Initialization of Masses and Springs

    # print(len(objs))
    while True:
        

        dog.updateSprings(w, T)
        dog.simulate(dt)

        
        
        

        T += dt


if __name__ == "__main__":
    main()