# analysis.py
# -----------
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


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    answerDiscount = 0
    answerNoise = 0.2
    return answerDiscount, answerNoise

def question3a():
    """
    1. Prefer the close exit (+1), risking the cliff (-10)
    python3 gridworld.py -g DiscountGrid -a value --discount 0.1 --noise 0 --livingReward 0.01
    2. Prefer the close exit (+1), but avoiding the cliff (-10)
    python3 gridworld.py -g DiscountGrid -a value --discount 0.1 --noise 0.1 --livingReward 0.01
    3. Prefer the distant exit (+10), risking the cliff (-10)
    python3 gridworld.py -g DiscountGrid -a value --discount 0.9 --noise 0 --livingReward 0.01
    4. Prefer the distant exit (+10), avoiding the cliff (-10)
    python3 gridworld.py -g DiscountGrid -a value --discount 0.8 --noise 0.5 --livingReward 0.1
    5. Avoid both exits and the cliff (so an episode should never terminate)
    python3 gridworld.py -g DiscountGrid -a value --discount 0 --noise 0 --livingReward 0.1
    """
    answerDiscount = 0.1
    answerNoise = 0
    answerLivingReward = 0.01
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3b():
    answerDiscount = 0.1
    answerNoise = 0.1
    answerLivingReward = 0.01
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3c():
    answerDiscount = 0.9
    answerNoise = 0
    answerLivingReward = 0.1
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3d():
    answerDiscount = 0.9
    answerNoise = 0.5
    answerLivingReward = 0.01
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3e():
    answerDiscount = 0
    answerNoise = 0
    answerLivingReward = 0.1
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question8():
    """answerEpsilon = None
    answerLearningRate = None
    return answerEpsilon, answerLearningRate"""
    return 'NOT POSSIBLE'
    # If not possible, return 'NOT POSSIBLE'

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
