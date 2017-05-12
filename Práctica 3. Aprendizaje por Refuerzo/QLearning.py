# coding=utf-8
import random
import numpy as np


# Environment size
width = 5
height = 16

# Actions
num_actions = 4

actions_list = {"UP": 0,
                "RIGHT": 1,
                "DOWN": 2,
                "LEFT": 3
                }

actions_vectors = {"UP": (-1, 0),
                   "RIGHT": (0, 1),
                   "DOWN": (1, 0),
                   "LEFT": (0, -1)
                   }

# Discount factor
discount = 0.8

Q = np.zeros((height * width, num_actions))  # Q matrix
Rewards = np.zeros(height * width)  # Reward matrix, it is stored in one dimension


def getState(y, x):
    return y * width + x


def getStateCoord(state):
    return int(state / width), int(state % width)


def getActions(state):
    y, x = getStateCoord(state)
    actions = []
    if x < width - 1:
        actions.append("RIGHT")
    if x > 0:
        actions.append("LEFT")
    if y < height - 1:
        actions.append("DOWN")
    if y > 0:
        actions.append("UP")
    return actions


def getRndAction(state):
    return random.choice(getActions(state))


def getRndState():
    return random.randint(0, height * width - 1)


Rewards[4 * width + 3] = -10000
Rewards[4 * width + 2] = -10000
Rewards[4 * width + 1] = -10000
Rewards[4 * width + 0] = -10000

Rewards[9 * width + 4] = -10000
Rewards[9 * width + 3] = -10000
Rewards[9 * width + 2] = -10000
Rewards[9 * width + 1] = -10000

Rewards[3 * width + 3] = 100
final_state = getState(3, 3)

print np.reshape(Rewards, (height, width))


def qlearning(s1, a, s2):
    Q[s1][a] = Rewards[s2] + discount * max(Q[s2])
    return

def greedy(state):
    index = np.argmax(Q[state])
    if(Q[state][index] == 0): return getRndAction(state)
    return actions_list.keys()[actions_list.values().index(index)]

def egreedy(state, epsilon):
    chance = random.random()
    if(chance >= epsilon): return greedy(state)
    return getRndAction(state)

# Aleatorio
def getDefaultAverage(episodes):
    global Q
    Q = np.zeros((height * width, num_actions))  # Q matrix
    actions = 0
    for i in xrange(episodes):
        state = getRndState()
        while state != final_state:
            action = getRndAction(state)
            actions += 1
            y = getStateCoord(state)[0] + actions_vectors[action][0]
            x = getStateCoord(state)[1] + actions_vectors[action][1]
            new_state = getState(y, x)
            qlearning(state, actions_list[action], new_state)
            state = new_state
    return float(actions) / float(episodes)

# Greedy
def getGreedyAverage(episodes):
    global Q
    Q = np.zeros((height * width, num_actions))  # Q matrix
    actions = 0
    for i in xrange(episodes):
        state = getRndState()
        while state != final_state:
            action = greedy(state)
            actions += 1
            y = getStateCoord(state)[0] + actions_vectors[action][0]
            x = getStateCoord(state)[1] + actions_vectors[action][1]
            new_state = getState(y, x)
            qlearning(state, actions_list[action], new_state)
            state = new_state
    return float(actions) / float(episodes)

# E-Greedy
def getEgreedyAverage(episodes, epsilon):
    global Q
    Q = np.zeros((height * width, num_actions))  # Q matrix
    actions = 0
    for i in xrange(episodes):
        state = getRndState()
        while state != final_state:
            action = egreedy(state, epsilon)
            actions += 1
            y = getStateCoord(state)[0] + actions_vectors[action][0]
            x = getStateCoord(state)[1] + actions_vectors[action][1]
            new_state = getState(y, x)
            qlearning(state, actions_list[action], new_state)
            state = new_state
    return float(actions) / float(episodes)

epoch = 100
print "Promedio de acciones por 100 episodios:"
print "Aleatorio = ", getDefaultAverage(epoch)
print "Greedy = ", getGreedyAverage(epoch)
print "E-Greedy(20%) = ", getEgreedyAverage(epoch, .2)
print "E-Greedy(40%) = ", getEgreedyAverage(epoch, .4)
print "E-Greedy(60%) = ", getEgreedyAverage(epoch, .6)
print "E-Greedy(80%) = ", getEgreedyAverage(epoch, .8)

"""
# Q matrix plot
s = 0
ax = plt.axes()
ax.axis([-1, width + 1, -1, height + 1])

for j in xrange(height):

    plt.plot([0, width], [j, j], 'b')
    for i in xrange(width):
        plt.plot([i, i], [0, height], 'b')

        direction = np.argmax(Q[s])
        if s != final_state:
            if direction == 0:
                ax.arrow(i + 0.5, 0.75 + j, 0, -0.35, head_width=0.08, head_length=0.08, fc='k', ec='k')
            if direction == 1:
                ax.arrow(0.25 + i, j + 0.5, 0.35, 0., head_width=0.08, head_length=0.08, fc='k', ec='k')
            if direction == 2:
                ax.arrow(i + 0.5, 0.25 + j, 0, 0.35, head_width=0.08, head_length=0.08, fc='k', ec='k')
            if direction == 3:
                ax.arrow(0.75 + i, j + 0.5, -0.35, 0., head_width=0.08, head_length=0.08, fc='k', ec='k')
        s += 1

    plt.plot([i+1, i+1], [0, height], 'b')
    plt.plot([0, width], [j+1, j+1], 'b')

#plt.show()
"""