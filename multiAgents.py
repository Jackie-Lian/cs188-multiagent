# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        pac_x, pac_y = newPos
        #move towards food
        eval = 0
        foodList = newFood.asList()
        numFoodLeft = len(foodList)

        #move towards nearest food, the smaller the better
        #totalFoodDist the smaller the better
        nearestFoodPos = (0,0)
        nearestFoodDist = 1000000
        totalFoodDist = 0
        for food_x, food_y in foodList:
            if abs(pac_x - food_x) + abs(pac_y - food_y) < nearestFoodDist:
                nearestFoodDist = abs(pac_x - food_x) + abs(pac_y - food_y)
                nearestFoodPos = (food_x, food_y)
            totalFoodDist = totalFoodDist + abs(pac_x - food_x) + abs(pac_y - food_y)

        eval = eval + 20/(nearestFoodDist + 1) + 10/(totalFoodDist+1) #make sure no division by 0
    
        #avoid ghost, the greater the better
        nearestGhostPos = (0,0)
        nearestGhostDist = 100000
        ghostWeightedFactor = 1
        
        for ghostState in newGhostStates:
            ghost_x, ghost_y = ghostState.getPosition()
            if abs(pac_x - ghost_x) + abs(pac_y - ghost_y) < nearestGhostDist:
                 nearestGhostDist = abs(pac_x - ghost_x) + abs(pac_y - ghost_y)
                 nearestGhostPos = (ghost_x, ghost_y)
        if nearestGhostDist <= 2:
            ghostWeightedFactor = 0
        if nearestGhostDist <= 3:
            ghostWeightedFactor = 0.3
        if nearestFoodPos == nearestGhostPos and nearestGhostDist <= 1:
            eval = -500
        if newPos == nearestGhostPos:
            eval = -500
        
        eval = eval + (ghostWeightedFactor * nearestGhostDist)/5

        return successorGameState.getScore() + eval

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        numAgents = gameState.getNumAgents()
        def terminalTest(gameState, depth):
            #stop when the depth is reached or the game has already ended
            return depth == self.depth or gameState.isWin() or gameState.isLose()


        #returns the value for this maximizer node
        def maximizer(currState, depth):
            if terminalTest(currState, depth):
                return self.evaluationFunction(currState)
            actions  = currState.getLegalActions(0)
            maxScore = -100000
            for action in actions:
                maxScore = max(maxScore, minimizer(currState.generateSuccessor(0, action), depth, 1))
            return maxScore

        #returns the value at this minizer node
        def minimizer(currState, depth, ghostIndex):
            if terminalTest(currState, depth):
                return self.evaluationFunction(currState)
            actions = currState.getLegalActions(ghostIndex)
            minScore = 100000
            if ghostIndex == numAgents - 1:
                for action in actions:
                    minScore = min(minScore,
                               maximizer(currState.generateSuccessor(ghostIndex, action), depth + 1))
            else:
                for action in actions:
                     minScore = min(minScore,
                               minimizer(currState.generateSuccessor(ghostIndex, action), depth, ghostIndex + 1))
            return minScore
                
            
        legalActions = gameState.getLegalActions(0)
        bestAction = legalActions[0]
        bestScore = -10000
        for legalAction in legalActions:
            evalScore = minimizer(gameState.generateSuccessor(0,legalAction), 0, 1)
            if evalScore > bestScore:
                bestAction = legalAction
                bestScore = evalScore
        return bestAction

    
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        
        def terminalTest(gameState, depth):
            #stop when the depth is reached or the game has already ended
            return depth == self.depth or gameState.isWin() or gameState.isLose()


        #return the value of the max node using alpha-beta pruning
        def max_value(currState, alpha, beta, depth):
            if terminalTest(currState, depth):
                return self.evaluationFunction(currState)
            actions = currState.getLegalActions(0)
            maxScore = -1e9
            for action in actions:
                maxScore = max(maxScore, min_value(currState.generateSuccessor(0, action), alpha, beta, depth, 1))
                if maxScore > beta:
                    return maxScore
                alpha = max(alpha, maxScore)
            return maxScore
       

        def min_value(currState, alpha, beta, depth, ghostIndex):
            if terminalTest(currState, depth):
                return self.evaluationFunction(currState)
            actions = currState.getLegalActions(ghostIndex)
            minScore = 1e9
            if ghostIndex == currState.getNumAgents() - 1:
                for action in actions:
                    minScore = min(minScore, max_value(currState.generateSuccessor(ghostIndex, action), alpha, beta, depth + 1))
                    if minScore < alpha:
                        return minScore
                    beta = min(beta, minScore)
                return minScore
            else:
                for action in actions:
                    minScore = min(minScore, min_value(currState.generateSuccessor(ghostIndex, action), alpha, beta, depth, ghostIndex+1))
                    if minScore < alpha:
                        return minScore
                    beta = min(beta, minScore)
                return minScore
        
        legalActions = gameState.getLegalActions(0)
        bestAction = legalActions[0]
        bestScore = -1e9
        alpha = -1e9
        beta = 1e9
        for legalAction in legalActions:
            evalScore = min_value(gameState.generateSuccessor(0, legalAction), alpha, beta, 0, 1)
            if evalScore > bestScore:
                bestScore = evalScore
                bestAction = legalAction
            if bestScore > beta:
                return bestAction
            alpha = max(alpha, bestScore)
        return bestAction
    

        

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        def terminalTest(gameState, depth):
            #stop when the depth is reached or the game has already ended
            return depth == self.depth or gameState.isWin() or gameState.isLose()
        
        def expectiMin(currState, depth, ghostIndex):
            if terminalTest(currState, depth):
                return self.evaluationFunction(currState)
            actions = currState.getLegalActions(ghostIndex)
            totalScore = 0
            if ghostIndex == currState.getNumAgents() - 1:
                for action in actions:
                    totalScore = totalScore + expectiMax(currState.generateSuccessor(ghostIndex, action), depth + 1)
                return totalScore / len(actions)
            else:
                for action in actions:
                     totalScore = totalScore + expectiMin(currState.generateSuccessor(ghostIndex, action), depth, ghostIndex + 1)
                return totalScore / len(actions)

        def expectiMax(currState, depth):
            if terminalTest(currState, depth):
                return self.evaluationFunction(currState)
            actions  = currState.getLegalActions(0)
            maxScore = -100000
            for action in actions:
                maxScore = max(maxScore, expectiMin(currState.generateSuccessor(0, action), depth, 1))
            return maxScore
            

        legalActions = gameState.getLegalActions(0)
        bestAction = legalActions[0]
        bestVal = -1e9
        for action in legalActions:
            val = expectiMin(gameState.generateSuccessor(0, action), 0, 1)
            if val > bestVal:
                bestVal = val
                bestAction = action
        return bestAction

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Prioritize food eating, try to eat pellet if that brings a higher score. Avoid the ghost. 
    """

    pac_x, pac_y = currentGameState.getPacmanPosition()
    eval = 0

    #the closer the pacman is to food, the better --> add reciprocal
    #the less food there is left, the better --> add reciprocal
    #weighted so that pacman is more motivated to move towards the closestFood
    foodList = currentGameState.getFood().asList()
    pellets = currentGameState.getCapsules()
    foodList.extend(pellets)
    #find the closestFood and store its position
    totalFoodDist = 0
    closestFoodDist = 10000
    closestFoodPos = (0,0)
    for food_x, food_y in foodList:
        dist = abs(pac_x - food_x) + abs(pac_y - food_y)
        if pac_y == food_y:
            i, j = min(pac_x, food_x), max(pac_x, food_x)
            for k in range(i + 1, j):
                if currentGameState.hasWall(k, pac_y):
                    dist = dist + 4
        elif pac_x == food_x:
            i, j = min(pac_y, food_y),max(pac_y, food_y)
            for k in range(i+1, j):
                if currentGameState.hasWall(pac_x, k):
                    dist = dist + 4
        if dist < closestFoodDist:
            closestFoodDist = dist
            closestFoodPos = (food_x, food_y)
        totalFoodDist = totalFoodDist + dist
    eval = eval + 10 / (closestFoodDist + 1)
    eval = eval + 20 / (totalFoodDist + 1)
    numFood = currentGameState.getNumFood() + len(currentGameState.getCapsules())
    eval = eval + 30 / (numFood + 1)

    #avoid ghost
    ghostState = currentGameState.getGhostStates()
    ghostPos = ghostState[0].getPosition()
    ghostWeight = 1
    

    ghost_x, ghost_y = ghostPos
    ghostDist = abs(pac_x - ghost_x) + abs(pac_y - ghost_y)
    if ghostDist <= 1:
        ghostWeight = 0.1
    elif ghostDist <= 2:
        ghostWeight = 0.2
    elif ghostDist <= 4:
        ghostWeight = 0.5

    eval = eval - (5 / (ghostDist * ghostWeight + 1))
    if closestFoodPos == ghostPos and ghostDist <= 1:
        eval = -500
    foodToGhost = abs(closestFoodPos[0] - ghost_x) + abs(closestFoodPos[1] - ghost_y)
    if foodToGhost <= 2:
        eval = eval - 5
    elif foodToGhost >= 4:
        eval = eval + 5
    return eval + currentGameState.getScore()

# Abbreviation
better = betterEvaluationFunction
