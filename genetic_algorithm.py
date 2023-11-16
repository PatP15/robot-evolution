import torch
import numpy as np
from display_dog import simulate

class GeneticAlgorithm():

    def __init__(self, populationSize, numCenters):
        self.populationSize = populationSize
        self.numCenters = numCenters
        self.centerLocs, self.centerMats = self.randomSample()
        

    def randomSample(self):
        '''
            Center Location Tensor: (populationSize x numCenters x 3)
            Center Material Tensor: (populationSize x numCenters x 1)
        '''
        centerLocations = torch.rand(size=(self.populationSize, self.numCenters, 3))
        centerMaterials = torch.randint(low=1, high=4, size=(self.populationSize, self.numCenters, 1))
        return centerLocations, centerMaterials

    def evaluate(self):
        return simulate(self.centerLocs, self.centerMats)

    def select(self):
        distances = self.evaluate()

        # Optionally normalize the tensor to make it a probability distribution
        # distances = distances / distances.sum()

        # Sampling with replacement
        selectedIndices = torch.multinomial(distances, self.populationSize // 2, replacement=False)

        distances = distances[selectedIndices]
        self.centerLocs = self.centerLocs[selectedIndices]
        self.centerMats = self.centerMats[selectedIndices]
        
        sortedIndices = torch.argsort(-1 * distances) # -1 is to sort from largest to smallest
        self.centerLocs = self.centerLocs[sortedIndices]
        self.centerMats = self.centerMats[sortedIndices]

        return distances[0]

    def mutate(self):
        self.centerLocs = torch.clip(self.centerLocs + torch.randn_like(self.centerLocs), min=0.0, max=1.0)
        self.centerMats = torch.round(torch.clip(self.centerMats + torch.randn_like(self.centerMats), min=1, max=4))
        
    def clone(self):
        self.population[self.population.shape[0] // 2:, ...] = self.population[:self.population.shape[0] // 2, ...].copy()

    def recombine(self, mc):
        # Recombine Center Locations
        tempCenterLocs = self.centerLocs.reshape((self.population.shape[0] // 2, 2, ...))
        parents1 = tempCenterLocs[:, 0, ...]
        parents2 = tempCenterLocs[:, 1, ...]
        children1 = mc * parents1 + (1 - mc) * parents2
        children2 = (1 - mc) * parents1 + mc * parents2
        children = torch.concat([children1, children2], axis=0)
        self.centerLocs = torch.concat([self.centerLocs, children], axis=0)

        # Recombine Center Materials
        tempCenterMats = self.centerMats.reshape((self.population.shape[0] // 2, 2, ...))
        parents1 = tempCenterMats[:, 0, ...]
        parents2 = tempCenterMats[:, 1, ...]
        children1 = mc * parents1 + (1 - mc) * parents2
        children2 = (1 - mc) * parents1 + mc * parents2
        children = torch.concat([children1, children2], axis=0)
        self.centerMats = torch.concat([self.centerMats, children], axis=0)

    def run(self, iterations=100, repeat=1):
        # with open(outPath + "gold_ga2_function.csv", 'w', newline='') as outFile:
        #     writer = csv.writer(outFile)
        #     writer.writerow(["Iteration", "RMS", "Repeat"])

        for j in range(repeat):
            self.population = self.randomSample()
            maxDistance = 0.0
            for i in range(iterations):
                tmpDistance = self.select()
                self.mutate()
                self.recombine(mc=0.33)
                # self.clone()

                print(i*self.populationSize, ": ", tmpDistance)
                if tmpDistance > maxDistance:
                  maxDistance = tmpDistance
                  bestBot = (self.centerLocs[0], self.centerMats[0])

                # with open(outPath + "gold_ga2_function.csv", 'a', newline='') as outFile:
                #     writer = csv.writer(outFile)
                #     writer.writerow([i*self.populationSize, minError**(0.5), j])

            tmpDistance = self.select()
            if tmpDistance > maxDistance:
                maxDistance = tmpDistance
                bestBot = (self.centerLocs[0], self.centerMats[0])
            print("Max Distance: ", maxDistance)
            # Reset population
            self.centerLocs, self.centerMats = self.randomSample()