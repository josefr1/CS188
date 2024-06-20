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
import math

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

        "*** YOUR CODE HERE ***"
        """
        1) Get distance from Pacman to the closest pellet.
            a) An edge case is that you need to check when 
            a successor state eats a pellet. If you don't check, your successor state will eat the pellet and the distance to the nearest pellet will actually increase in successorGameState. You can check if the len of the newFoodList is less than oldFoodList (current state).
        2) Get distance from Pacman to the closest ghost.
        3) Check if Pacman is in the same column/row as a ghost. Penalize Pacman if he is.
        4) If a ghost is scared, you want to ignore all ghost states in the calculation. You can do this by checking if sum(newScaredTimes) > 0.
        """
        total = 0
        minFood = 1000
        for food in newFood.asList():
            #ghostDist = min([util.manhattanDistance(newPos, i.getPosition()) for i in newGhostStates])
            closestFood = util.manhattanDistance(newPos, food)
            if closestFood < minFood:
                minFood = closestFood
        distToGhost = min([util.manhattanDistance(
            newPos, ghost.getPosition()) for ghost in newGhostStates])
        if distToGhost <= 2:
            total += -3
        if minFood<distToGhost:
            total += 10
        if len(newFood.asList()) < len(currentGameState.getFood().asList()):
            total += 12
        ghostStatePositionsCol = [ghost.getPosition()[0] for ghost in newGhostStates]
        ghostStatePositionsRow = [ghost.getPosition()[1] for ghost in newGhostStates]
        if newPos[0] in ghostStatePositionsCol or newPos[1] == ghostStatePositionsRow:
            total -= 1
        if sum(newScaredTimes) > 0:
            total -= 4
        return total

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
        "*** YOUR CODE HERE ***"
        return self.value(gameState, 0, self.depth)[1]

    def value(self, GameState, agentIndex, depth):
        if depth == 0 or GameState.isWin() or GameState.isLose():
            return self.evaluationFunction(GameState), None
        if agentIndex == 0:
            return self.maxValue(GameState, agentIndex, depth)
        if agentIndex > 0:
            return self.minValue(GameState, agentIndex, depth)

    def minValue(self, gameState, agentIndex, depth):
        actions = gameState.getLegalActions(agentIndex)
        v = math.inf
        returnedAction = None
        for action in actions:
            successors = gameState.generateSuccessor(agentIndex, action)
            if agentIndex == gameState.getNumAgents() - 1:
                score = self.value(successors, 0, depth-1)[0]
            else:
                score = self.value(successors, agentIndex+1, depth)[0]
            if score < v:
                v = score
                returnedAction = action
        return (v, returnedAction)

    def maxValue(self, gameState, agentIndex, depth):
        actions = gameState.getLegalActions(agentIndex)
        v = -math.inf
        returnedAction = None
        for action in actions:
            successors = gameState.generateSuccessor(agentIndex, action)
            if agentIndex == gameState.getNumAgents() - 1:
                score = max(self.value(successors, 0, depth-1)[0], v)
            else:
                score = max(self.value(successors, agentIndex+1, depth)[0], v)
            if score > v:
                v = score
                returnedAction = action
        return (v, returnedAction)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #same code as above just with alpha, beta conditional search
        return self.value(gameState, 0, self.depth, -math.inf, math.inf)[1]
    def value(self, GameState, agentIndex, depth, alpha, beta):
        if depth == 0 or GameState.isWin() or GameState.isLose():
            return self.evaluationFunction(GameState), None
        if agentIndex == 0:
            return self.maxValue(GameState, agentIndex, depth, alpha, beta)
        if agentIndex > 0:
            return self.minValue(GameState, agentIndex, depth, alpha, beta)

    def maxValue(self, gameState, agentIndex, depth, alpha, beta):
        actions = gameState.getLegalActions(agentIndex)
        v = -math.inf
        returnedAction = None
        for action in actions:
            successor_game_state = gameState.generateSuccessor(agentIndex, action)
            if agentIndex == gameState.getNumAgents() - 1:
                score = self.value(successor_game_state, 0, depth-1, alpha, beta)[0]
            else:
                score = self.value(successor_game_state, agentIndex+1, depth, alpha, beta)[0]
            if score > v:
                v = score
                returnedAction = action
            if score > beta:
                return (score, action)
            alpha = max(alpha, v)
        return (v, returnedAction)

    def minValue(self, gameState, agentIndex, depth, alpha, beta):
        actions = gameState.getLegalActions(agentIndex)
        v = math.inf
        returnedAction = None
        for action in actions:
            successors = gameState.generateSuccessor(agentIndex, action)
            if agentIndex == gameState.getNumAgents() - 1:
                score = (self.value(successors, 0, depth-1, alpha, beta)[0])
            else:
                score = (self.value(successors, agentIndex + 1, depth, alpha, beta)[0])
            if score < v:
                v = score
                returnedAction = action
            if score < alpha:
                return (score, action)
            beta = min(beta, v)
        return (v, returnedAction)

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
        "*** YOUR CODE HERE ***"
        return self.value(gameState, 0, self.depth)[1]

    def value(self, gameState, agentIndex, depth):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), None
        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth)
        if agentIndex > 0:
            return self.expValue(gameState, agentIndex, depth)

    def maxValue(self, gameState, agentIndex, depth):
        actions = gameState.getLegalActions(agentIndex)
        v = -math.inf
        returnedAction = None
        for action in actions:
            successors = gameState.generateSuccessor(agentIndex, action)
            if agentIndex == gameState.getNumAgents() - 1:
                score = self.value(successors, 0, depth-1)[0]
            else:
                score = self.value(successors, agentIndex + 1, depth)[0]
            if score > v:
                v = score
                returnedAction = action
        return v, returnedAction

    def expValue(self, gameState, agentIndex, depth):
        actions = gameState.getLegalActions(agentIndex)
        score = 0
        returnedAction = None
        for action in actions:
            successor_game_state = gameState.generateSuccessor(agentIndex, action)
            if agentIndex == gameState.getNumAgents() - 1:
                score = score +self.value(successor_game_state, 0, depth-1)[0]
            else:
                score = score + self.value(successor_game_state, agentIndex + 1, depth)[0]
        score = score / len(actions)
        return (score, returnedAction)

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: I looked for the closest food to the pacman. I also looked for the closest ghost.
    My eval function looks at the ratio between current score and distance to ghost.
    If the ghost is scared or there is food closer to pacman then a ghost, pacman will
    prioritize going for the food. Else, pacman will prioritze staying alive. The distance to
    ghost is added with 1 to negate a divide by 0 error.
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    minFood = 1000
    for food in newFood.asList():
        #ghostDist = min([util.manhattanDistance(newPos, i.getPosition()) for i in newGhostStates])
        closestFood = util.manhattanDistance(newPos, food)
        if closestFood < minFood:
            minFood = closestFood
    distToGhost = min([util.manhattanDistance(
        newPos, ghost.getPosition()) for ghost in newGhostStates])
    if newScaredTimes[0] > 0 or minFood < distToGhost:
        eval = currentGameState.getScore() + 6 / (distToGhost + 1)
    else:
        eval = currentGameState.getScore() - 6 / (distToGhost + 1)
    return eval

# Abbreviation
better = betterEvaluationFunction
