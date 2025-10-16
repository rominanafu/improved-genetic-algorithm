
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

def dist_sites(x, y):
    """ Distance between two sites """
    return math.sqrt((coord[x][0] - coord[y][0])**2 + (coord[x][1] - coord[y][1])**2)

def evaluation(individual):
    """ Travel distance """
    distance = 0
    for helicopter in individual:
        nsites = len(helicopter)
        for i in range(nsites):
            c1 = helicopter[i]
            if i+1 == nsites:
                c2 = helicopter[0]
            else:
                c2 = helicopter[i+1]
            distance += dist_sites(c1, c2)
    return distance

def initialization(populationSize):
    """ Generation of initial population """
    res = []
    for i in range(populationSize-2):
        ## Every individual is a matrix of helicopters
        individual = []
        capacities = []
        times = []
        # Random order of rescue sites
        sites = list(range(1, len(coord)))
        random.shuffle(sites) 
        for j in sites:
            # Find where to insert site sites[j]
            sepudo = False
            for k in range(len(individual)):
                # Try to check if site[j] can be inserted in helicopter[k]
                if capacities[k] + demand[j] <= hcapacity and max(times[k], tws[j][0]) <= tws[j][1]:
                    # insert site[j] into helicopter[k]
                    capacities[k] += demand[j]
                    times[k] = max(times[k], tws[j][0]) + st[j]
                    individual[k].append(j)
                    sepudo = True
                    break
            if not sepudo:
                # insert a new helicopter
                individual.append([0])
                k = len(individual)-1
                individual[k].append(j)
                capacities.append(demand[j])
                times.append(tws[j][0] + st[j])

        for k in range(len(individual)):
            individual[k].append(0)
        res.append(individual)
    
    class Site:
        def __init__(self, index, demand, tws, coord, st):
            self.index = index
            self.demand = demand
            self.tws = tws
            self.coord = coord
            self.st = st
            self.dist_origin = np.sqrt(self.coord[0]**2 + self.coord[1]**2)
    
    # Create a list of Sites which can be sorted by different properties
    sites = []
    for i in range(1, len(coord)):
        sites.append(Site(i, demand[i], tws[i], coord[i], st[i]))
    
    sites.sort(key=lambda x:x.tws[0])
    individual = []
    capacities = []
    times = []
    for j in sites:
        # different options for inserting it
        helicopter = -1
        min_pos = np.inf
        for k in range(len(individual)):
            # check if it can be inserted into helicopter k
            if capacities[k] + j.demand <= hcapacity and max(times[k], j.tws[0]) <= j.tws[1]:
                if min_pos > len(individual[k]):
                    min_pos = len(individual[k])
                    helicopter = k
        if helicopter == -1:
            individual.append([0])
            capacities.append(0)
            times.append(0)
            helicopter = len(individual)-1
        # update helicopter where j is going to be inserted
        individual[helicopter].append(j.index)
        capacities[helicopter] += j.demand
        times[helicopter] = max(times[helicopter], j.tws[0]) + j.st
    for k in range(len(individual)):
        individual[k].append(0)
    
    res.append(individual)
    
    sites.sort(key=lambda x:x.dist_origin)
    individual = []
    capacities = []
    times = []
    for j in sites:
        # different options for inserting it
        helicopter = -1
        min_pos = np.inf
        for k in range(len(individual)):
            # check if it can be inserted into helicopter k
            if capacities[k] + j.demand <= hcapacity and max(times[k], j.tws[0]) <= j.tws[1]:
                if min_pos > len(individual[k]):
                    min_pos = len(individual[k])
                    helicopter = k
        if helicopter == -1:
            individual.append([0])
            capacities.append(0)
            times.append(0)
            helicopter = len(individual)-1
        # update helicopter where j is going to be inserted
        individual[helicopter].append(j.index)
        capacities[helicopter] += j.demand
        times[helicopter] = max(times[helicopter], j.tws[0]) + j.st
    for k in range(len(individual)):
        individual[k].append(0)
    
    res.append(individual)

    return res

def selection(population, exclude):
    """ Randomly select a different individual """
    possible = np.setdiff1d(np.arange(len(population)), exclude)
    return population[possible[np.random.randint(len(possible))]]

def crossover(parentA, parentB, cRate):
    """ Append random path from helicopter of another individual to last path of
    current individual and viceversa. Return best fitted one. """
    if (np.random.random() <= cRate):
        # Offspring 1
        hpath = parentB[np.random.randint(len(parentB))][1:]
        offspring1 = parentA[:]
        for i in range(len(offspring1)):
            # Remove sites visited in hpath
            offspring1[i] = [elem for elem in offspring1[i] if elem not in hpath[:-1]]
        offspring1[-1].pop()
        offspring1[-1].extend(hpath)

        # Offspring 2
        hpath = parentA[np.random.randint(len(parentA))][1:]
        offspring2 = parentB[:]
        for i in range(len(offspring2)):
            # Remove sites visited in hpath
            offspring2[i] = [elem for elem in offspring2[i] if elem not in hpath[:-1]]
        offspring2[-1].pop()
        offspring2[-1].extend(hpath)

        # Return fittest offspring
        eval1, eval2 = evaluation(offspring1), evaluation(offspring2)
        if (eval1 < eval2):
            return offspring1
        return offspring2

    else:
        offspring1 = parentA[:]
        offspring2 = parentB[:]

        # Return fittest offspring
        eval1, eval2 = evaluation(offspring1), evaluation(offspring2)
        if (eval1 < eval2):
            return offspring1
        return offspring2

def mutation(individual, mRate):
    """ Swap two rescue sites """
    if (np.random.random() <= mRate):
        i, j = 0, 0
        ii, jj = 0, 0
        while ((i == ii and j == jj) or (individual[i][j] == 0) or
               (individual[ii][jj] == 0)):
            i = np.random.randint(len(individual))
            j = np.random.randint(len(individual[i])-1)
            ii = np.random.randint(len(individual))
            jj = np.random.randint(len(individual[ii])-1)
        individual[i][j], individual[ii][jj] = individual[ii][jj], individual[i][j]
    
    return individual

def trayFactible(path):
    """ Determine if a path does not violate any rule """
    time = 0
    cap = 0
    for i in range(1, len(path)):
        time += dist_sites(path[i], path[i-1]) * travelTime
        cap += demand[path[i]]
        if ((time > tws[path[i]][1]) or (cap > hcapacity)):
            return False
        time = max(time, tws[path[i]][0]) + st[path[i]]
    return True

def repair(individual):
    """ Repair strategy to comply with time window constraints """
    tw_violation = []
    for i in range(len(individual)):
        helicopter = individual[i]
        time = 0
        cap = 0
        prev = 0
        for site in helicopter:
            t = dist_sites(prev, site) * travelTime
            if ((time+t > tws[site][1]) or (cap+demand[site] > hcapacity)):
                # helicopter arrives after time window's end or
                # is not able to finish service before time window ends
                tw_violation.append(site)
            else:
                time = max(time + t, tws[site][0]) + st[site]
                cap += demand[site]
        individual[i] = [h for h in helicopter if h not in tw_violation]
    
    # Insert deleted sites
    for site in tw_violation:

        # Find feasible positions to insert site
        pos = []
        for i in range(len(individual)):
            helicopter = individual[i]
            time = 0
            prev = 0
            for j in range(1, len(helicopter)):
                if (trayFactible(helicopter[:j] + [site] + helicopter[j:])):
                    pos.append([i, j])

        # Insert site
        if (len(pos) == 0): # New helicopter path
            individual.append([0, site, 0])
        else: # Select best position to insert site (minimizing distance)
            bestPos = 0
            bestDist = (dist_sites(individual[pos[0][0]][pos[0][1]], site) +
                        dist_sites(individual[pos[0][0]][pos[0][1]-1], site))
            for i in range(1, len(pos)):
                x, y = pos[i]
                dist = dist_sites(individual[x][y], site) + dist_sites(individual[x][y-1], site)
                if (dist < bestDist):
                    bestDist = dist
                    bestPos = i
            individual[pos[bestPos][0]].insert(pos[bestPos][1], site)

    return individual

def localSearch(individual):
    """ Local search strategy for expoitation """
    # Copia del ind og
    new_ind = [list(route) for route in individual]

    Rs = []
    for route in new_ind:
        for site in route:
            if site != 0 and site not in Rs:
                Rs.append(site)
    n = len(Rs)
    if n == 0:
        return new_ind
    Rr = [np.random.choice(Rs)]
    Rs.remove(Rr[0])
    n_reps = max(1, n // 10)
    for _ in range(n_reps):
        if len(Rs) == 0:
            break
        i = np.random.choice(Rr)
        # Calcular distancias a sitios restantes
        distancias = [(j, dist_sites(i, j)) for j in Rs]
        # Ordenar por distancia creciente
        distancias.sort(key=lambda x: x[1])
        # Tomar el más cercano
        cercano = distancias[0][0]
        Rr.append(cercano)
        Rs.remove(cercano)

    for route in new_ind:
        for site in Rr:
            if site in route:
                route.remove(site)

    for site in Rr:
        mejorRuta = None
        mejorPos = None
        mejorCosto = float('inf')

        for r, route in enumerate(new_ind):
            for pos in range(1, len(route)):
                if (trayFactible(route[:pos] + [site] + route[pos:])):
                    c1, c2 = route[pos - 1], route[pos]
                    costo = (dist_sites(c1, site) + dist_sites(site, c2) - dist_sites(c1, c2))
                    if costo < mejorCosto:
                        mejorCosto = costo
                        mejorRuta = r
                        mejorPos = pos

        # Insertar en la mejor posición encontrada
        if mejorRuta is not None:
            new_ind[mejorRuta].insert(mejorPos, site)
        else:
            # Si no hay lugar factible, crear nueva ruta
            new_ind.append([0, site, 0])

    return new_ind

def globalSearch(individual):
    """ Global search strategy for exploration """

    new_ind = [list(route) for route in individual if len(route) > 2]

    num_routes = len(new_ind)
    if num_routes == 0:
        return new_ind

    z = max(1, num_routes // 2)
    selected_routes = random.sample(range(num_routes), z)

    Rd = []
    for idx in selected_routes:
        for site in new_ind[idx]:
            if site != 0:
                Rd.append(site)
    new_ind = [route for i, route in enumerate(new_ind) if i not in selected_routes]
    for site in Rd:
        mejorRuta = None
        mejorPos = None
        mejorCosto = float('inf')

        for r, route in enumerate(new_ind):
            for pos in range(1, len(route)):  # evita 0
                if (trayFactible(route[:pos] + [site] + route[pos:])):
                    c1, c2 = route[pos - 1], route[pos]
                    costo = (dist_sites(c1, site) + dist_sites(site, c2) - dist_sites(c1, c2))
                    if costo < mejorCosto:
                        mejorCosto = costo
                        mejorRuta = r
                        mejorPos = pos

        # Insertar en la mejor posición encontrada
        if mejorRuta is not None:
            new_ind[mejorRuta].insert(mejorPos, site)
        else:
            # Crear una nueva
            new_ind.append([0, site, 0])

    return new_ind

def geneticAlgorithm(populationSize, cRate, mRate, Lm, generations):
    """ Genetic algorithm framework """

    # Initial population
    population = initialization(populationSize)

    evals = [0] * populationSize
    lastImprov = [0] * populationSize

    # Set a best individual
    evals[0] = evaluation(population[0])
    bestEval = evals[0]
    bestInd = population[0]

    # Evaluate initial population
    for i in range(1, populationSize):
        evals[i] = evaluation(population[i])
        if (evals[i] < bestEval):
            bestEval = evals[i]
            bestInd = population[i]

    for _ in range(generations):

        print(f"Processing generation {_+1}...")
        
        # Exploitation
        for i in range(populationSize):

            individual = population[i]

            # Generate new individual
            parent = selection(population, i)
            newIndividual = crossover(individual, parent, cRate)
            newIndividual = mutation(newIndividual, mRate)
            newIndividual = repair(newIndividual)
            newIndividual = localSearch(newIndividual)

            eval = evaluation(newIndividual)
            if (eval < evals[i]):
                population[i] = newIndividual
                evals[i] = eval
                lastImprov[i] = 0

                # Check if newIndividual improves bestInd
                if (eval < bestEval):
                    bestEval = eval
                    bestInd = newIndividual

            else:
                lastImprov[i] += 1

        # Exploration
        for i in range(populationSize):
            if (lastImprov[i] > Lm):
                newIndividual = globalSearch(population[i])
                population[i] = newIndividual
                lastImprov[i] = 0

                eval = evaluation(newIndividual)
                evals[i] = eval
                if (eval < bestEval):
                    bestEval = eval
                    bestInd = newIndividual
        
        print(f"Best evaluation iteration {_+1}: {bestEval}")

    return bestInd, bestEval

def drawSol(individual):

    x = [i for i,j in coord]
    y = [j for i,j in coord]

    plt.scatter(x, y)
    plt.scatter(x[0], y[0], color='r')
    for path in individual:
        plt.plot([x[p] for p in path], [y[p] for p in path])

    plt.show()

def solution(data_path, hcap):
    global coord, tws, st, demand, hcapacity, travelTime

    data = pd.read_csv(data_path)

    coord = [[data['XCOORD.'][i], data['YCOORD.'][i]] for i in range(len(data))]
    tws = [[data['READY TIME'][i], data['DUE DATE'][i]] for i in range(len(data))]
    st = data['SERVICE TIME']
    demand = data['DEMAND']
    hcapacity = hcap
    travelTime = 0

    bestInd, bestEval = geneticAlgorithm(populationSize=100, cRate=0.7, mRate=0.3, Lm=10, generations=25)

    print(f"Best evaluation: {bestEval}")
    print(bestInd)

    drawSol(bestInd)

########## CASO 1: Clustered Customers, Short Schedule Horizon ###########

solution('data/solomon_dataset/C1/C101.csv', 200)

