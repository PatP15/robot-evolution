import torch
import numpy as np
from display_genetic import simulate
from display_genetic import makeBoxes, make_multilayer_sphere, makeOnePyramid
import csv 
import pickle
import argparse
device = "cuda:0" if torch.cuda.is_available() else "cpu"
from display_dog import makeOneDog
class GeneticAlgorithm():

    def __init__(self, populationSize, numCenters):
        self.populationSize = populationSize
        self.numCenters = numCenters
        # self.ages = torch.zeros(size=(self.populationSize, 1), dtype=torch.float)
        self.centerLocs, self.centerMats = self.randomSample()
        

    def randomSample(self):
        '''
            Center Location Tensor: (populationSize x numCenters x 3)
            Center Material Tensor: (populationSize x numCenters x 1)

            x: 0, 5 -> length
            y: 0, 4 -> width
            z: 0, 2 -> height
        '''
        centerLocations = torch.rand(size=(self.populationSize, self.numCenters, 3), dtype=torch.float)
        centerLocations[..., 0] = centerLocations[..., 0] * 5
        centerLocations[..., 1] = centerLocations[..., 1] * 4
        centerLocations[..., 2] = centerLocations[..., 2] * 2
        centerMaterials = torch.randint(low=1, high=6, size=(self.populationSize, self.numCenters, 1), dtype=torch.float)
        return centerLocations.to(device), centerMaterials.to(device)

    def evaluate(self):
        return simulate(self.centerLocs, self.centerMats)

    def select(self):
        distances = self.evaluate()
        # distances[distances > 100] = 0
        # Optionally normalize the tensor to make it a probability distribution
        # distances = distances / distances.sum()

        # Sampling with replacement
        # print("Children Pop: ", len(distances))
        selectedIndices = torch.multinomial(distances, self.populationSize//2, replacement=False)

        distances = distances[selectedIndices]
        self.centerLocs = self.centerLocs[selectedIndices]
        self.centerMats = self.centerMats[selectedIndices]
        
        sortedIndices = torch.argsort(-1 * distances) # -1 is to sort from largest to smallest
        distances = distances[sortedIndices]
        self.centerLocs = self.centerLocs[sortedIndices]
        self.centerMats = self.centerMats[sortedIndices]

        return distances[0]

    def mutate(self, alpha=0.1):
        # print("begin mutate: ", self.centerLocs.device)
        maxPos = torch.ones_like(self.centerLocs)
        maxPos[..., 0] = maxPos[..., 0] * 5
        maxPos[..., 1] = maxPos[..., 1] * 5
        maxPos[..., 2] = maxPos[..., 2] * 5
        mutated_locs = self.centerLocs + alpha * torch.randn_like(self.centerLocs)
        zeroes = torch.zeros_like(self.centerLocs)
        torch.clip(mutated_locs, zeroes, maxPos, out=self.centerLocs)  
        self.centerMats = torch.round(torch.clip(self.centerMats + torch.randn_like(self.centerMats), min=1, max=5))
        # print("end mutate: ", self.centerLocs.device)

    def clone(self):
        self.centerLocs[self.centerLocs.shape[0] // 2:, ...] = self.centerLocs[:self.centerLocs.shape[0] // 2, ...].clone()
        self.centerMats[self.centerMats.shape[0] // 2:, ...] = self.centerMats[:self.centerMats.shape[0] // 2, ...].clone()

    def recombine(self, mc):
        # Recombine Center Locations
        # print("Before recombine: ", self.centerLocs.shape)
        # print("centerLocs: ", self.centerLocs)
        # tempCenterLocs = self.centerLocs.reshape((self.centerLocs.shape[0] // 2, -1))
        split = self.centerLocs.shape[0] // 2
        parents1 = self.centerLocs[:split, :, ...]
        if self.centerLocs.shape[0] % 2 == 1:
            split += 1
        
        parents2 = self.centerLocs[split:, :, ...]
        # print("parents1: ", parents1)
        # print("parents2: ", parents2)
        children1 = mc * parents1 + (1 - mc) * parents2
        children2 = (1 - mc) * parents1 + mc * parents2
        children = torch.concat([children1, children2], axis=0)
        
        # children = children.reshape((-1, 2, self.centerLocs.shape[2]))
        # print("children: ", children)
        # print("children shape: ", children.shape)
        self.centerLocs = torch.concat([self.centerLocs, children], axis=0)

        # Recombine Center Materials
        # tempCenterMats = self.centerMats.reshape((self.centerMats.shape[0] // 2, 2))
        split = self.centerMats.shape[0] // 2
        parents1 = self.centerMats[:split, :, ...]
        if self.centerMats.shape[0] % 2 == 1:
            split += 1
        parents2 = self.centerMats[split:, :, ...]
        # print("parents1: ", parents1)
        # print("parents2: ", parents2)
        children1 = mc * parents1 + (1 - mc) * parents2
        children2 = (1 - mc) * parents1 + mc * parents2
        children = torch.concat([children1, children2], axis=0)
        # children = children.reshape((-1, 2, self.centerMats.shape[2]))
        self.centerMats = torch.concat([self.centerMats, children], axis=0)
        self.centerMats = torch.round(torch.clip(self.centerMats, min=1, max=5))

        # print("After recombine: self center locs ", self.centerLocs.size(), self.centerMats.size())

    def run(self, iterations=100, repeat=1):
        with open("evolve_robot.csv", 'w', newline='') as outFile:
            writer = csv.writer(outFile)
            writer.writerow(["Iteration", "Distance", "Repeat"])

        for j in range(repeat):
            maxDistance = 0.0
            bestBot = None
            for i in range(iterations):
                # print("Iteration: ", i)
                # print("Population Size: ", self.centerLocs.size()[0])
                # print("start run: ", self.centerLocs.device)
                torch.cuda.synchronize()
                tmpDistance = self.select() 
                torch.cuda.synchronize()
                print("Eval: ", i*self.populationSize, ": ", tmpDistance.item())
                if tmpDistance > maxDistance:
                    maxDistance = tmpDistance
                    bestBot = (np.array(self.centerLocs[0].cpu()), np.array(self.centerMats[0].cpu()))
                    with open("best_robot.pkl", 'wb') as f:
                        pickle.dump(bestBot, f) 
                with open("evolve_robot.csv", 'a', newline='') as outFile:
                    writer = csv.writer(outFile)
                    writer.writerow([i*self.populationSize, maxDistance.item(), j])
                
                self.mutate()
                self.recombine(mc=0.33)
                torch.cuda.synchronize()

            tmpDistance = self.select()
            if tmpDistance > maxDistance:
                maxDistance = tmpDistance
                bestBot = (np.array(self.centerLocs[0].cpu()), np.array(self.centerMats[0].cpu()))
                with open("evolve_robot.csv", 'a', newline='') as outFile:
                    writer = csv.writer(outFile)
                    writer.writerow([(i+1)*self.populationSize, maxDistance.item(), j])

                with open("best_robot.pkl", 'wb') as f:
                    pickle.dump(bestBot, f)

            self.centerLocs, self.centerMats = self.randomSample()
            print("Max Distance: ", maxDistance)
            print("Best Bot: ", bestBot)



class GeneticAlgorithmPareto():

    def __init__(self, populationSize, numCenters, initialShape):
        self.populationSize = populationSize
        self.numCenters = numCenters
        self.ages = torch.zeros(size=(self.populationSize,), dtype=torch.float).to(device)
        self.centerLocs, self.centerMats = self.randomSample()
        self.initialShape = initialShape    
        if initialShape == "box":
            self.obj_masses, self.obj_springs = makeBoxes()
        elif initialShape == "sphere":
            self.obj_masses, self.obj_springs = make_multilayer_sphere(3, 10, 5)
        elif initialShape == "pyramid":
            self.obj_masses, self.obj_springs = makeOnePyramid()
        elif initialShape == "dog":
            self.obj_masses, self.obj_springs = makeOneDog()
        

    def randomSample(self):
        '''
            Center Location Tensor: (populationSize x numCenters x 3)
            Center Material Tensor: (populationSize x numCenters x 1)

            x: 0, 5 -> length
            y: 0, 4 -> width
            z: 0, 2 -> height
        '''
        centerLocations = torch.rand(size=(self.populationSize, self.numCenters, 3), dtype=torch.float)
        centerLocations[..., 0] = centerLocations[..., 0] * 5
        centerLocations[..., 1] = centerLocations[..., 1] * 5
        centerLocations[..., 2] = centerLocations[..., 2] * 5
        centerMaterials = torch.randint(low=1, high=6, size=(self.populationSize, self.numCenters, 1), dtype=torch.float)
        return centerLocations.to(device), centerMaterials.to(device)
    
    def diversitySample(self, sampleSize=1):
        '''
            Center Location Tensor: (populationSize x numCenters x 3)
            Center Material Tensor: (populationSize x numCenters x 1)

            x: 0, 5 -> length
            y: 0, 4 -> width
            z: 0, 2 -> height
        '''
        centerLocations = torch.rand(size=(sampleSize, self.numCenters, 3), dtype=torch.float)
        centerLocations[..., 0] = centerLocations[..., 0] * 5
        centerLocations[..., 1] = centerLocations[..., 1] * 5
        centerLocations[..., 2] = centerLocations[..., 2] * 5
        centerMaterials = torch.randint(low=1, high=6, size=(sampleSize, self.numCenters, 1), dtype=torch.float)
        return centerLocations.to(device), centerMaterials.to(device)

    def evaluate(self):
        # change here to evaluate with different objects
        # for now just putting in boxes
        # print("Population Center Materials:\n", self.centerMats)
        return simulate(self.centerLocs, self.centerMats, self.obj_masses, self.obj_springs)
    
    def calculatePareto(self, distances, ages):
        points = torch.stack([distances, -ages], dim=1)
        # print("Pareto Points Tensor:\n", points)

        # Assuming points is a tensor of shape (n, 2)
        n = points.shape[0]

        # Expand dimensions to allow broadcasting: shapes become (n, 1, 2) and (1, n, 2)
        p1 = points.unsqueeze(1)  # Shape: (n, 1, 2)
        p2 = points.unsqueeze(0)  # Shape: (1, n, 2)

        # Compare points: (n, n, 2)
        # A point p1 dominates p2 if it is less or equal in all dimensions and strictly less in at least one dimension
        domination = torch.all(p1 <= p2, dim=2) & torch.any(p1 < p2, dim=2)

        # Count the number of dominations for each point: sum over rows
        domination_counts = domination.sum(dim=1).float()

        return domination_counts

    def select(self):
        distances = self.evaluate()

        numDominated = self.calculatePareto(distances, self.ages)
        # print("Number of times Dominated:\n", numDominated)

        # distances[distances > 100] = 0
        # Optionally normalize the tensor to make it a probability distribution
        # distances = distances / distances.sum()

        # Sampling with replacement
        # print("Children Pop: ", len(distances))
        selectedIndices = torch.argsort(numDominated)[:numDominated.size()[0] // 2] # torch.multinomial(numDoms, self.populationSize//2, replacement=False)

        # print("Selected Number of times Dominated:\n", numDominated[selectedIndices])

        distances = distances[selectedIndices]
        self.centerLocs = self.centerLocs[selectedIndices]
        self.centerMats = self.centerMats[selectedIndices]
        self.ages = self.ages[selectedIndices]
        
        # sortedIndices = torch.argsort(-1 * distances) # -1 is to sort from largest to smallest
        # distances = distances[sortedIndices]
        # self.centerLocs = self.centerLocs[sortedIndices]
        # self.centerMats = self.centerMats[sortedIndices]
        # self.ages = self.ages[sortedIndices]

        return distances[0]

    def mutate(self, alpha=0.1):
        # print("begin mutate: ", self.centerLocs.device)
        maxPos = torch.ones_like(self.centerLocs)
        maxPos[..., 0] = maxPos[..., 0] * 5
        maxPos[..., 1] = maxPos[..., 1] * 5
        maxPos[..., 2] = maxPos[..., 2] * 5
        mutated_locs = self.centerLocs + alpha * torch.randn_like(self.centerLocs)
        # zeroes = torch.zeros_like(self.centerLocs)
        # torch.clip(mutated_locs, zeroes, maxPos, out=self.centerLocs)
        self.centerLocs = mutated_locs
        self.centerMats = torch.round(torch.clip(self.centerMats + 4 * alpha * torch.randn_like(self.centerMats), min=1, max=5))
        # print("end mutate: ", self.centerLocs.device)

    def clone(self):
        self.centerLocs[self.centerLocs.shape[0] // 2:, ...] = self.centerLocs[:self.centerLocs.shape[0] // 2, ...].clone()
        self.centerMats[self.centerMats.shape[0] // 2:, ...] = self.centerMats[:self.centerMats.shape[0] // 2, ...].clone()

    def recombine(self, mc):
        # Recombine Center Locations
        # print("Before recombine: ", self.centerLocs.shape)
        # print("centerLocs: ", self.centerLocs)
        # print("Center Loc Shape ", self.centerLocs.size())
        tempCenterLocs = self.centerLocs.reshape((self.centerLocs.shape[0] // 2, 2, self.numCenters, 3))
        # split = self.centerLocs.shape[0] // 2
        # parents1 = self.tempCenterLocs[:split, :, ...]
        # parents2 = self.tempCenterLocs[split:, :, ...]
        parents1 = tempCenterLocs[:, 0]
        parents2 = tempCenterLocs[:, 1]
        # print("parents1: ", parents1)
        # print("parents2: ", parents2)
        children1 = mc * parents1 + (1 - mc) * parents2
        children2 = (1 - mc) * parents1 + mc * parents2
        children = torch.concat([children1, children2], axis=0)
        
        # children = children.reshape((-1, 2, self.centerLocs.shape[2]))
        # print("children: ", children)
        # print("children shape: ", children.shape)
        self.centerLocs = torch.concat([self.centerLocs, children], axis=0)

        # Recombine Center Materials
        tempCenterMats = self.centerMats.reshape((self.centerMats.shape[0] // 2, 2, self.numCenters, 1))
        # split = self.centerMats.shape[0] // 2
        # parents1 = self.centerMats[:split, :, ...]
        # parents2 = self.centerMats[split:, :, ...]
        parents1 = tempCenterMats[:, 0]
        parents2 = tempCenterMats[:, 1]
        # print("parents1: ", parents1)
        # print("parents2: ", parents2)
        children1 = mc * parents1 + (1 - mc) * parents2
        children2 = (1 - mc) * parents1 + mc * parents2
        children = torch.concat([children1, children2], axis=0)
        # children = children.reshape((-1, 2, self.centerMats.shape[2]))
        self.centerMats = torch.concat([self.centerMats, children], axis=0)
        self.centerMats = torch.round(torch.clip(self.centerMats, min=1, max=5))

        # split = self.ages.shape[0] // 2
        # parents1 = self.ages[:split]
        # parents2 = self.ages[split:]
        # newAges = torch.max(parents1, parents2).repeat_interleave(2) + 1
        newAges = self.ages.reshape((self.ages.shape[0] // 2, 2))
        newAges = torch.max(newAges, dim=1).values.repeat_interleave(2) + 1
        self.ages = self.ages + 1
        self.ages = torch.concat([self.ages, newAges], axis=0)

        # print("After recombine: self center locs ", self.centerLocs.size(), self.centerMats.size())

    def diversityInjection(self, diversityProp=0.1):
        numNew = int(self.populationSize * diversityProp)
        newCenterLocs, newCenterMats = self.diversitySample(sampleSize=numNew)
        self.centerLocs[-numNew:] = newCenterLocs
        self.centerMats[-numNew:] = newCenterMats
        self.ages[-numNew:] = 0

    def run(self, iterations=100, repeat=1):
        with open(self.initialShape + "_evolve_robot.csv", 'w', newline='') as outFile:
            writer = csv.writer(outFile)
            writer.writerow(["Iteration", "Distance", "Repeat"])

        for j in range(repeat):
            maxDistance = 0.0
            bestBot = None
            for i in range(iterations):
                # print("Iteration: ", i)
                # print("Population Size: ", self.centerLocs.size()[0])
                # print("start run: ", self.centerLocs.device)
                # print("Population Ages:\n", self.ages)
                torch.cuda.synchronize()
                tmpDistance = self.select()
                torch.cuda.synchronize()
                print("Eval: ", i*self.populationSize, ": ", tmpDistance.item())
                if tmpDistance > maxDistance:
                    maxDistance = tmpDistance
                    bestBot = (np.array(self.centerLocs[0].cpu()), np.array(self.centerMats[0].cpu()))
                    with open(self.initialShape + "_best_robot.pkl", 'wb') as f:
                        pickle.dump(bestBot, f) 
                with open(self.initialShape + "_evolve_robot.csv", 'a', newline='') as outFile:
                    writer = csv.writer(outFile)
                    writer.writerow([i*self.populationSize, maxDistance.item(), j])
                
                self.mutate(alpha=0.001)
                self.recombine(mc=0.33)
                self.diversityInjection(diversityProp=0.1)
                torch.cuda.synchronize()

            tmpDistance = self.select()
            if tmpDistance > maxDistance:
                maxDistance = tmpDistance
                bestBot = (np.array(self.centerLocs[0].cpu()), np.array(self.centerMats[0].cpu()))
                with open(self.initialShape + "_evolve_robot.csv", 'a', newline='') as outFile:
                    writer = csv.writer(outFile)
                    writer.writerow([(i+1)*self.populationSize, maxDistance.item(), j])

                with open(self.initialShape + "_best_robot.pkl", 'wb') as f:
                    pickle.dump(bestBot, f)

            self.centerLocs, self.centerMats = self.randomSample()
            print("Max Distance: ", maxDistance)
            print("Best Bot: ", bestBot)

def main(shape):
    
    ga = GeneticAlgorithmPareto(1000, 24, shape)
    ga.run(iterations=10000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Genetic Algorithm with Pareto Optimization")
    parser.add_argument("-s","--shape", type=str, default="box", help="Starting shape")
    args = parser.parse_args()
    main(args.shape)
