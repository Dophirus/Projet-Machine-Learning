import random
import itertools
import numpy
from NeuralNetwork import *

nbFeatures = 8
nbActions = 4

class Game:
    def __init__(self, hauteur, largeur):
        self.grille = [[0]*hauteur  for _ in range(largeur)]
        self.hauteur, self.largeur = hauteur, largeur
        self.serpent = [[largeur//2-i-1, hauteur//2] for i in range(4)]
        for (x,y) in self.serpent: self.grille[x][y] = 1
        self.direction = 3
        self.accessibles = [[x,y] for (x,y) in list(itertools.product(range(largeur), range(hauteur))) if [x,y] not in self.serpent]
        self.fruit = [0,0]
        self.setFruit()
        self.enCours = True
        self.steps = 0
        self.score = 4
    
    def setFruit(self):
        if (len(self.accessibles)==0): return
        self.fruit = self.accessibles[random.randint(0, len(self.accessibles)-1)][:]
        self.grille[self.fruit[0]][self.fruit[1]] = 2

    def refresh(self):
        nextStep = self.serpent[0][:]
        match self.direction:
            case 0: nextStep[1]-=1
            case 1: nextStep[1]+=1
            case 2: nextStep[0]-=1
            case 3: nextStep[0]+=1

        if nextStep not in self.accessibles:
            self.enCours = False
            return
        self.accessibles.remove(nextStep)
        if self.grille[nextStep[0]][nextStep[1]]==2:
            self.setFruit()
            self.steps = 0
            self.score+=1
        else:
            self.steps+=1
            self.grille[self.serpent[-1][0]][self.serpent[-1][1]] = 0
            self.accessibles.append(self.serpent[-1][:])
            self.serpent = self.serpent[:-1]
            if self.steps>self.hauteur*self.largeur:
                self.enCours = False
                return

        self.grille[nextStep[0]][nextStep[1]] = 1
        self.serpent = [nextStep]+self.serpent

    def getFeatures(self):
        features = numpy.zeros(8)
        
        snake_head = self.serpent[0]
        snake_body = self.serpent[1::]

        #Neurone 1-4 TODO : Optimiser
        testUp = np.add(snake_head, [0,1])
        testDown = np.add(snake_head, [0,-1])
        testLeft = np.add(snake_head, [-1,0])
        testRight = np.add(snake_head, [1,0])

        for coords in snake_body:
            if np.all(testUp == coords) or not 0 <= testUp[1] < self.hauteur:
                features[0] = 1
            if np.all(testDown == coords) or not 0 <= testDown[1] < self.hauteur:
                features[1] = 1
            if np.all(testLeft == coords) or not 0 <= testLeft[0] < self.largeur:
                features[2] = 1
            if np.all(testRight == coords) or not 0 <= testRight[0] < self.largeur:
                features[3] = 1
            
        # Neurone 5-6
        features[4] = np.clip([snake_head[0] - self.fruit[0]],-1,1)
        features[5] = np.clip([snake_head[1] - self.fruit[-1]],-1,1)

        # Neurone 7
        features[6] = self.direction

        # Neurone 8
        match self.direction:
            case 0: features[7] = (snake_head[1]) / self.hauteur 
            case 1: features[7] = (self.hauteur - snake_head[1]) / self.hauteur 
            case 2: features[7] = (snake_head[0]) / self.largeur 
            case 3: features[7] = (self.largeur - snake_head[0]) / self.largeur 

        print(features)

        return features
    
    def print(self):
        print("".join(["="]*(self.largeur+2)))
        for ligne in range(self.hauteur):
            chaine = ["="]
            for colonne in range(self.largeur):
                if self.grille[colonne][ligne]==1: chaine.append("#")
                elif self.grille[colonne][ligne]==2: chaine.append("F")
                else: chaine.append(" ")
            chaine.append("=")
            print("".join(chaine))
        print("".join(["="]*(self.largeur+2))+"\n")

