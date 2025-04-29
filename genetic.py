import numpy
from NeuralNetwork import *
from snake import *

def eval(sol, gameParams):
    results = np.zeros((gameParams["nbGames"],2))
    for i in range(0,gameParams["nbGames"]):
        curGame = Game(gameParams["height"], gameParams["width"])
        #curGame.print()
        while curGame.enCours:
            pred = sol.nn.predict(curGame.getFeatures())
            curGame.direction = pred
            curGame.refresh()
            #curGame.print()
        results[i] = [curGame.score,curGame.steps]
        #print(f"game N°{i} : {results[i]}")
    #print(1 / (gameParams["nbGames"] * gameParams["height"] * gameParams["width"] * 1000) * sum(1000 * results[i][0] + results[i][1] for i in range(gameParams["nbGames"])))
    return 1 / (gameParams["nbGames"] * gameParams["height"] * gameParams["width"] * 1000) * sum(1000 * results[i][0] + results[i][1] for i in range(gameParams["nbGames"]))       

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

def optimize(taillePopulation, tailleSelection, pc, arch, gameParams, nbIterations, nbThreads, scoreMax):
    population = initialization(taillePopulation, arch, gameParams)

    '''
    TODO : corps de l'algorithme principal
    '''


    return population[0].nn
