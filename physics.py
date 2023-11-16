import numpy as np
import torch 
from itertools import combinations

"""
    masses: (n x 4 x 3)
    n = number of masses,
    each mass has a mass (stored in first index), acceleration, velocity, and position

    springs: (m x 4)
    m = number of springs,
    each row ~ (massIndex1, massIndex2, k spring constant, rest length)
"""

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
    #fix bottom z
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
    return masses, springs

def generateTetra(translation, rotation):
    massLocations = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1)]
    massValues = [1, 1, 1, 1]
    all_combinations = list(combinations(range(4), 2))
    springs = np.array([[comb[0], comb[1], 10000, np.linalg.norm(np.array(massLocations[comb[0]]) - np.array(massLocations[comb[1]]))] for comb in all_combinations])
    masses = generateMasses(massLocations, massValues)
    for mass in masses:
        position = mass[3].T
        newPosition = rotation.dot(position)
        mass[3] = newPosition.T
    masses[:, 3] = masses[:, 3] + translation
    return masses, springs

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

def computeGroundCollisionForces(masses, K_g=5000):
    N = masses.size(0)
    groundCollisionForces = torch.zeros((N, 3))#.cuda()
    groundCollisionForces[masses[:, 3, 2] < 0, 2] = -masses[masses[:, 3, 2] < 0, 3, 2] * K_g
    return groundCollisionForces

def batch_assign_materials_to_masses(batch_masses, batch_center_positions, batch_material_properties):
    # Assuming the first dimension is the batch dimension (num_robots)
    batch_distances = torch.cdist(batch_masses, batch_center_positions)  # Shape (num_robots, num_masses, num_centers)

    # Find the index of the closest center for each mass in each robot
    closest_center_indices = torch.argmin(batch_distances, dim=2)

    # Using advanced indexing to select the right materials for each mass in each robot
    batch_mass_materials = batch_material_properties[torch.arange(batch_material_properties.size(0))[:, None, None], 
                                                     closest_center_indices[..., None].expand(-1, -1, batch_material_properties.size(-1))].squeeze(-2)
    return batch_mass_materials

def batch_compute_spring_parameters(batch_springs, batch_mass_materials):
    # batch_idx1, batch_idx2 = batch_springs[..., 0].long(), batch_springs[..., 1].long()
    batch_idx1 = batch_springs[..., 0].long()

    # Gather the material properties for each mass of each spring in each robot
    batch_material1 = batch_mass_materials[torch.arange(batch_mass_materials.size(0))[:, None, None], batch_idx1[..., None].expand(-1, -1, batch_mass_materials.size(-1))]
    # batch_material2 = batch_mass_materials[torch.arange(batch_mass_materials.size(0))[:, None, None], batch_idx2[..., None].expand(-1, -1, batch_mass_materials.size(-1))]

    # Calculate the average properties for each spring
    batch_spring_properties = batch_material1
    return batch_spring_properties

def assignMaterials(masses, springs, popCenterLocs, popCenterMats):
    populationSize = popCenterLocs.size()[0]
    # Assuming you have your batched tensors ready
    popMasses = masses.reshape(populationSize, -1, ...) # Tensor of shape (num_robots, num_masses, 3)
    # popCenterLocs should be a Tensor of shape (num_robots, num_centers, 3)
    # popCenterMats should be a Tensor of shape (num_robots, num_centers, material_property_dim)

    # Assign materials to masses for each robot
    pop_mass_materials = batch_assign_materials_to_masses(popMasses, popCenterLocs, popCenterMats)

    # Compute spring parameters for each robot
    popSprings = springs.reshape(populationSize, -1, ...) # Tensor of shape (num_robots, num_springs, 3)
    popSpringProperties = batch_compute_spring_parameters(popSprings, pop_mass_materials)
    return popSpringProperties

def computeFrictionForces(masses, netForces, groundCollisionForces, mu_s, mu_k):
    N = masses.size(0)
    frictionForces = torch.zeros((N, 3))

    # Indices where mass is at or below the ground
    ground_indices = (masses[:, 3, 2] <= 0)

    if ground_indices.any():
        # Normal force magnitudes (only in z-direction)
        Fn = -groundCollisionForces[ground_indices, 2]

        # Horizontal force magnitudes (only in x and y directions)
        FH_x = netForces[ground_indices, 0]
        FH_y = netForces[ground_indices, 1]
        FH_magnitude = torch.sqrt(FH_x**2 + FH_y**2)
        unit_vector_x = FH_x / (FH_magnitude + 1e-8)  # Adding a small value to avoid division by zero
        unit_vector_y = FH_y / (FH_magnitude + 1e-8)

        # Apply kinetic friction condition
        kinetic_friction_indices = FH_magnitude >= (Fn * mu_s)  
        # Update friction forces for masses on the ground
        frictionForces[ground_indices][kinetic_friction_indices, 0] = FH_x[kinetic_friction_indices] - Fn[kinetic_friction_indices] * mu_k * unit_vector_x[kinetic_friction_indices]
        frictionForces[ground_indices][kinetic_friction_indices, 1] = FH_y[kinetic_friction_indices] - Fn[kinetic_friction_indices] * mu_k * unit_vector_y[kinetic_friction_indices]

        frictionForces[ground_indices][~kinetic_friction_indices, 0] = 0
        frictionForces[ground_indices][~kinetic_friction_indices, 1] = 0
        frictionForces[ground_indices, 2] = -Fn

    return frictionForces


    # return frictionForces

