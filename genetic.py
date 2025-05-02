import numpy
import random
from NeuralNetwork import *
from snake import *

def eval(sol, gameParams):
    score = 0
    for i in range(0,gameParams["nbGames"]):
        curGame = Game(gameParams["height"], gameParams["width"])
        #curGame.print()
        while curGame.enCours:
            pred = sol.nn.predict(curGame.getFeatures())
            curGame.direction = pred
            curGame.refresh()
            #curGame.print()
        score += 1000 * curGame.score + curGame.steps
    sol.score = score / (gameParams["nbGames"] * gameParams["height"] * gameParams["width"] * 1000)   

'''
Représente une solution avec
_un réseau de neurones
_un score (à maximiser)

vous pouvez ajouter des attributs ou méthodes si besoin
'''
class Individu:
    def __init__(self, nn):
        self.nn = nn
        self.score = 0


'''
La méthode d'initialisation de la population est donnée :
_on génère N individus contenant chacun un réseau de neurones (de même format)
_on évalue et on trie des individus
'''
def initialization(taillePopulation, arch, gameParams):
    population = []
    for i in range(taillePopulation):
        nn = NeuralNetwork((arch[0],))
        for j in range(1, len(arch)):
            nn.addLayer(arch[j], "elu")
        population.append(Individu(nn))

    for sol in population: eval(sol, gameParams)
    population.sort(reverse=True, key=lambda sol:sol.score)
    
    return population

def optimize(taillePopulation, tailleSelection, pc, mr, arch, gameParams, nbIterations, nbThreads, scoreMax):
    population = initialization(taillePopulation, arch, gameParams)

    for i in range(nbIterations):
        if population[0].score >= scoreMax:
            break

        bests = population[:tailleSelection]
        random.shuffle(bests)

        couples = [(bests[i], bests[i+1]) for i in range(0, len(bests), 2)]

        for couple in couples:
            children = croisement(couple, arch, pc, mr)
            population.append(Individu(children[0]))
            population.append(Individu(children[1]))

        for sol in filter(lambda sol: sol.score == 0, population):
        #for sol in population:
            eval(sol,gameParams)

        population.sort(reverse=True ,key=lambda sol:sol.score)

        population = population[:taillePopulation]
        scores = [sol.score for sol in population]

        print(f"Iteration {i} :  Max {population[0].score} / Moy {np.mean(scores)}")

    # Croisement (voir comment faire parce que je suis conne)
    # Mutation (des enfants seulement)
    # Suppression de X individus avec le score le plus bas jusqu'a ravoir le nombre initial de pop
    # Evaluer chaque individu (avec eval) (1ère fois déja faite avec ini, refaire avec les enfants non testés)
    # si nbIteration < iterationActuelles ou scoreMeilleur < scoreMax, on renvoie le meilleur 


    return population[0].nn

def croisement(parents, arch, pc, mr):
    parent1 = parents[0].nn
    parent2 = parents[1].nn

    children = [parent1.clone(),parent2.clone()]

    if random.random() > pc:
        return children
    
    for layer in range(len(arch[1:])):
        alpha = random.random()

        children[0].layers[layer].weights = alpha * parent1.layers[layer].weights + (1 - alpha) * parent2.layers[layer].weights
        children[1].layers[layer].weights = (1 - alpha) * parent1.layers[layer].weights + alpha * parent2.layers[layer].weights

        children[0].layers[layer].bias = alpha * parent1.layers[layer].bias + (1 - alpha) * parent2.layers[layer].bias
        children[1].layers[layer].bias = (1 - alpha) * parent1.layers[layer].bias + alpha * parent2.layers[layer].bias

        # wi,j(c1) = α × wi,j(p1) + (1 − α) × wi,j(p2)
        # wi,j(c2) = (1 − α) × wi,j(p1) + α × wi,j(p2)
            
    for child in children:
        for layer in child.layers:

            mutationBias = mr / layer.outputShape[0]
            mutationWeight = mr / layer.inputShape[0]

            for i in range(layer.outputShape[0]):
                rand = random.random()
                if rand < mutationBias:
                    layer.bias[i] += np.random.randn() * 0.1

            for i in range(layer.inputShape[0]):
                for y in range(layer.outputShape[0]):
                    rand = random.random()
                    if rand < mutationWeight:
                        layer.weights[i][y] += np.random.randn() * 0.1

    return children