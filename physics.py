import numpy as np
import torch 
from itertools import combinations
# from genetic_algorithm import device
device = "cuda:0" if torch.cuda.is_available() else "cpu"

"""
    masses: (n x 4 x 3)
    n = number of masses,
    each mass has a mass (stored in first index), acceleration, velocity, and position

    springs: (m x 4)
    m = number of springs,
    each row ~ (massIndex1, massIndex2, k spring constant, rest length)
"""
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
        self.edges = self.springs[:, :3]  # Get the first two columns which are the vertex indices for each spring
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
        staticFrictionIndices, kinecticFrictionForces = newComputeFrictionForces(self.masses, netForces, mu_s, mu_k)
        # frictionForces = computeFrictionForces(self.masses, netForces, groundCollisionForces, mu_s, mu_k)
        ground_indices = (self.masses[:, 3, 2] <= 0)

        # Update net forces with friction forces for ground-contacting masses
        # print(ground_indices.size(), "\n\n\n", staticFrictionIndices.size())
        kineticFrictionMassIndices = torch.logical_and(ground_indices, torch.logical_not(staticFrictionIndices))
        netForces[kineticFrictionMassIndices, :2] += kinecticFrictionForces[kineticFrictionMassIndices, :2]
        # print(netForces)
        # Integration step
        # Calculate acceleration
        self.masses[:, 1] = netForces / self.masses[:, 0, 0].unsqueeze(-1)
        # Calculate velocity
        self.masses[:, 2] += self.masses[:, 1] * dt
        # Zero the velocity for static friction masses
        staticFrictionMassIndices = torch.logical_and(ground_indices, staticFrictionIndices)
        self.masses[staticFrictionMassIndices, 2, :2] = 0.0
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
        self.springs[self.materials == 5, 2] = 0
        # Update resting lengths
        self.springs[self.materials == 1, 3] = self.og[self.materials == 1]
        self.springs[self.materials == 2, 3] = self.og[self.materials == 2]
        self.springs[self.materials == 3, 3] = self.og[self.materials == 3] * (1 + 0.25 * np.sin(w*T))
        self.springs[self.materials == 4, 3] = self.og[self.materials == 4] * (1 + 0.25 * np.sin(w*T+torch.pi))
        self.springs[self.materials == 5, 3] = self.og[self.materials == 5]

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
    force_accumulator = torch.zeros((N, 3)).to(device)

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
    gravityForces = torch.zeros((N, 3)).to(device)
    gravityForces[:, 2] = -9.81 * masses[:, 0, 0]
    return gravityForces

def computeGroundCollisionForces(masses, K_g=5000):
    N = masses.size(0)
    groundCollisionForces = torch.zeros((N, 3)).to(device)
    groundCollisionForces[masses[:, 3, 2] < 0, 2] = -masses[masses[:, 3, 2] < 0, 3, 2] * K_g
    return groundCollisionForces

def batch_assign_materials_to_masses(batch_masses, batch_center_positions, batch_material_properties):
    # Assuming the first dimension is the batch dimension (num_robots)
    # print(batch_masses.device)
    # print(batch_center_positions.device)
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

    # print("Batch idx1 size: ", batch_idx1.size())
    # Gather the material properties for each mass of each spring in each robot
    # print(batch_mass_materials)
    # batch_material1 = # match_mass_materials[:, batch_idx1]
    batch_material1 = batch_mass_materials[torch.arange(batch_mass_materials.size(0))[:, None, None], batch_idx1[..., None].expand(-1,  -1, batch_mass_materials.size(-1))]
    # batch_material2 = batch_mass_materials[torch.arange(batch_mass_materials.size(0))[:, None, None], batch_idx2[..., None].expand(-1, -1, batch_mass_materials.size(-1))]

    # Calculate the average properties for each spring
    batch_spring_properties = batch_material1
    return batch_spring_properties

def assignMaterials(masses, springs, popCenterLocs, popCenterMats):
    populationSize = popCenterLocs.size()[0]
    # Assuming you have your batched tensors ready
    # print(masses.size())
    masses = masses.clone()[:, 3, :] # Tensor of shape (num_robots, num_masses, 3)
    # print(masses.size())
    # print(populationSize)
    popMasses = masses.reshape(populationSize, -1, 3) # Tensor of shape (num_robots, num_masses, 3)
    # print(popMasses.size())
    # popCenterLocs should be a Tensor of shape (num_robots, num_centers, 3)
    # popCenterMats should be a Tensor of shape (num_robots, num_centers, material_property_dim)

    # Assign materials to masses for each robot
    pop_mass_materials = batch_assign_materials_to_masses(popMasses, popCenterLocs, popCenterMats)
    # print("Pop Mass Materials shape: ", pop_mass_materials.size())

    # Compute spring parameters for each robot
    popSprings = springs.clone().reshape(populationSize, -1, 4) # Tensor of shape (num_robots, num_springs, 3)
    numMasses = popMasses.size(1)
    popSprings[..., :2] = popSprings[..., :2] % numMasses
    popSpringProperties = batch_compute_spring_parameters(popSprings, pop_mass_materials)
    popSpringProperties = popSpringProperties.reshape(-1)
    return popSpringProperties

def newComputeFrictionForces(masses, netForces, mu_s, mu_k):
    horizontalForces = torch.norm(netForces[:, :2], dim=1)
    verticalForces = netForces[:, 2]
    normalForces = torch.abs(verticalForces) # torch.norm(verticalForces, dim=1)
    velocities = masses[:, 2, :] # (n x 3)
    moving = torch.norm(velocities, dim=1) > 1e-4
    maxStaticFriction = mu_s * normalForces
    # Normalize the velocity vectors and multiply by the friction coefficient and normal force
    normalizedVelocities = velocities / (velocities.norm(dim=1, keepdim=True) + 1e-8)
    frictionalForces = -normalizedVelocities * mu_k * normalForces.unsqueeze(-1) * moving.float().unsqueeze(-1)
    return horizontalForces < maxStaticFriction, frictionalForces

def computeFrictionForces(masses, netForces, groundCollisionForces, mu_s, mu_k):
    N = masses.size(0)
    frictionForces = torch.zeros((N, 3)).to(device)

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

