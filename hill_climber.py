import torch
import numpy as np
from display_dog import simulate
import csv 
import pickle

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class HillClimber():

    def __init__(self, populationSize, numCenters):
        self.populationSize = populationSize
        self.numCenters = numCenters
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
        centerMaterials = torch.randint(low=1, high=4, size=(self.populationSize, self.numCenters, 1), dtype=torch.float)
        return centerLocations.to(device), centerMaterials.to(device)

    def evaluate(self):
        return simulate(self.centerLocs, self.centerMats)

    def select(self):
        distances = self.evaluate()

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
        maxPos[..., 1] = maxPos[..., 1] * 4
        maxPos[..., 2] = maxPos[..., 2] * 2
        mutated_locs = self.centerLocs + alpha * torch.randn_like(self.centerLocs)
        zeroes = torch.zeros_like(self.centerLocs)
        torch.clip(mutated_locs, zeroes, maxPos, out=self.centerLocs)  
        self.centerMats = torch.round(torch.clip(self.centerMats + torch.randn_like(self.centerMats), min=1, max=4))
        # print("end mutate: ", self.centerLocs.device)

    def clone(self):
        self.centerLocs = torch.concat([self.centerLocs, self.centerLocs])
        self.centerMats = torch.concat([self.centerMats, self.centerMats])

    def run(self, iterations=100, repeat=1):
        with open("evolve_robot_hc.csv", 'w', newline='') as outFile:
            writer = csv.writer(outFile)
            writer.writerow(["Iteration", "Distance", "Repeat"])

        for j in range(repeat):
            maxDistance = 0.0
            bestBot = None
            for i in range(iterations):
                tmpDistance = self.select()

                print("Eval: ", i*self.populationSize, ": ", tmpDistance.item())
                if tmpDistance > maxDistance:
                    maxDistance = tmpDistance
                    bestBot = (np.array(self.centerLocs[0].cpu()), np.array(self.centerMats[0].cpu()))
                    with open("best_robot_hc.pkl", 'wb') as f:
                        pickle.dump(bestBot, f) 
                    with open("evolve_robot_hc.csv", 'a', newline='') as outFile:
                        writer = csv.writer(outFile)
                        writer.writerow([i*self.populationSize, maxDistance.item(), j])
                
                self.mutate()
                self.clone() 

            tmpDistance = self.select()
            if tmpDistance > maxDistance:
                maxDistance = tmpDistance
                bestBot = (np.array(self.centerLocs[0].cpu()), np.array(self.centerMats[0].cpu()))
                with open("evolve_robot_hc.csv", 'a', newline='') as outFile:
                    writer = csv.writer(outFile)
                    writer.writerow([(i+1)*self.populationSize, maxDistance.item(), j])

                with open("best_robot_hc.pkl", 'wb') as f:
                    pickle.dump(bestBot, f)

            self.centerLocs, self.centerMats = self.randomSample()
            print("Max Distance: ", maxDistance)
            print("Best Bot: ", bestBot)

def main():
    hc = HillClimber(1000, 12)
    hc.run(iterations=25)

if __name__ == "__main__":
    main()
