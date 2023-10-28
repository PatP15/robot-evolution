import numpy as np
import torch 
from itertools import combinations

def generateMasses(massLocs, massVals):
    numMasses = len(massLocs)
    masses = np.zeros((numMasses, 4, 3))
    massLocs = np.array(massLocs)
    massVals = np.array(massVals)
    masses[:, 3, :] = masses[:, 3, :] + massLocs
    masses[:, 0, 0] = masses[:, 0, 0] + massVals
    return masses

def randomRotation():
    a, b, c = np.random.random(size=3) * 2 * np.pi
    yaw = np.array([[np.cos(a), -np.sin(a), 0],
                    [np.sin(a), np.cos(a), 0],
                    [0, 0, 1]])
    pitch = np.array([[np.cos(b), 0, np.sin(b)],
                    [0, 1, 0],
                    [-np.sin(b), 0, np.cos(b)]])
    roll = np.array([[1, 0, 0],
                    [0, np.cos(c), -np.sin(c)],
                    [0, np.sin(c), np.cos(c)]])
    rotation = yaw.dot(pitch).dot(roll)
    return rotation

def generateCube(translation, rotation):
    massLocations = [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1), (0, 0, 2), (0, 1, 2), (1, 0, 2), (1, 1, 2)]
    massValues = [1, 1, 1, 1, 1, 1, 1, 1]
    all_combinations = list(combinations(range(8), 2))
    springs = np.array([[comb[0], comb[1], 10000, np.linalg.norm(np.array(massLocations[comb[0]]) - np.array(massLocations[comb[1]]))] for comb in all_combinations])
    masses = generateMasses(massLocations, massValues)
    for mass in masses:
        position = mass[3].T
        newPosition = rotation.dot(position)
        mass[3] = newPosition.T
    masses[:, 3] = masses[:, 3] + translation



def compute_spring_forces(masses, springs):
    idx1 = springs[:, 0].long()
    idx2 = springs[:, 1].long()
    k = springs[:, 2]
    restLength = springs[:, 3]

    pos1 = masses[idx1, 3]
    pos2 = masses[idx2, 3]

    delta = pos2 - pos1
    length = torch.norm(delta, dim=1)
    # print("\nLengths\n", length)
    force_magnitude = -k * (length- restLength)
    # print("\nForce Magnitude\n", force_magnitude)

    # Normalize the displacement to get the direction
    direction = delta / (length.unsqueeze(-1) + 1e-8)
    forces = direction * force_magnitude.unsqueeze(-1)

    return forces

def aggregate_spring_forces(springs, forces, masses):
    N = masses.size(0)
    force_accumulator = torch.zeros((N, 3))#.cuda()

    idx1 = springs[:, 0].long()
    idx2 = springs[:, 1].long()

    force_accumulator.index_add_(0, idx1, -forces)  # negative because action and reaction are opposite
    force_accumulator.index_add_(0, idx2, forces)

    return force_accumulator

def compute_net_spring_forces(masses, springs):
    spring_forces = compute_spring_forces(masses, springs)
    # print("\nSpring Forces\n", spring_forces)
    net_forces = aggregate_spring_forces(springs, spring_forces, masses)
    # print("\nNet Spring Forces\n", net_forces)
    return net_forces

def computeGravityForces(masses):
    N = masses.size(0)
    gravityForces = torch.zeros((N, 3))#.cuda()
    gravityForces[:, 2] = -9.81 * masses[:, 0, 0]
    return gravityForces

def computeGroundCollisionForces(masses, K_g=100000):
    N = masses.size(0)
    groundCollisionForces = torch.zeros((N, 3))#.cuda()
    groundCollisionForces[masses[:, 3, 2] < 0, 2] = -masses[masses[:, 3, 2] < 0, 3, 2] * K_g
    return groundCollisionForces