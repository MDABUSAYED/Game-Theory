#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 08:45:04 2024

@author: msayed
"""


from gurobipy import *

import numpy as np

np.set_printoptions(suppress=True)

# Initialize the model
model = Model("Stackelberg Game")

f = open('input4_5.txt', "r")

Number_of_defender_action = int(f.readline())

Number_of_attacker_action = int(f.readline())



# Defender Variables
Defender_action_space = []
for i in range(Number_of_defender_action):
    n = "x-" + str(i)
    Defender_action_space.append(model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=n))
model.update()



# Attacker Variables
Attacker_action_space = []
for i in range(Number_of_attacker_action):
    n = "attack-" + str(i)
    Attacker_action_space.append(model.addVar(lb=0, ub=1, vtype=GRB.INTEGER, name=n))
model.update()



# Add defender stategy constraints
con = LinExpr()
for i in range(Number_of_defender_action):
    con.add(Defender_action_space[i])
model.addConstr(con == 1)

# Add attacker stategy constraints
con = LinExpr()
for i in range(Number_of_attacker_action):
    con.add(Attacker_action_space[i])
model.addConstr(con == 1)



# Get reward for attacker and defender
R = []
C = []
for i in range(Number_of_defender_action):
    rewards = f.readline().split()
    r = []
    c = []
    for j in range(Number_of_attacker_action):
        r_and_c = rewards[j].split(",")
        r.append(int(r_and_c[0]))
        c.append(int(r_and_c[1]))
    R.append(r)
    C.append(c)



# Attacker best response constraints
Utility_attacker = []

for j in range(Number_of_attacker_action):
    con = LinExpr()
    for i in range(Number_of_defender_action):
        con.add(Defender_action_space[i]*C[i][j])
    Utility_attacker.append(con)

for i in range(Number_of_attacker_action):
    for j in range(Number_of_defender_action):
        if i != j:
            val1 = QuadExpr()
            val1.add(Utility_attacker[i] * Attacker_action_space[i])
            val2 = QuadExpr()
            val2.add(Utility_attacker[j] * Attacker_action_space[i])
            model.addConstr(val1 >= val2, name="attacker_best_response_B" +str(i)+'_vs_B'+str(j))




# Update objective function

obj = QuadExpr()
for i in range(Number_of_defender_action):
    for j in range(Number_of_attacker_action):
        r = float(R[i][j])
        obj.add(r * Defender_action_space[i] * Attacker_action_space[j])
        

# Set the objective: Maximize the defender's expected utility
model.setObjective(obj, GRB.MAXIMIZE)

# Optimize the model
model.optimize()

# Print out values
def printSeperator():
    print("---------------")

# Print results
if model.status == GRB.OPTIMAL:
    
    '''
    print("\nOptimal Strategy:")
    print(f"Probability of choosing A1 (p1): {p1.X}")
    print(f"Probability of choosing A2 (p2): {p2.X}")
    print(f"Attacker chooses B1: {q1.X}")
    print(f"Attacker chooses B2: {q2.X}")
    print(f"Defender's expected utility: {model.objVal}")
    '''
    
    printSeperator()
    for v in model.getVars():
        print("%s -> %g" % (v.varName, v.x))

    printSeperator()
    print("Obj -> %g" % model.objVal)
    printSeperator()

# Example payoffs for the attacker (U_a), assuming a 4x5 game matrix
U_a = np.array(C)

# Suppose p is the optimal strategy for the defender (from the Stackelberg solver)
p = np.array([var.X for var in Defender_action_space])

# Assume the attacker's best response is action B2 (index 1 in 0-based indexing)
q = np.array([var.X for var in Attacker_action_space])

# Calculate the attacker's expected utility
expected_utility = np.sum(p[:, None] * q[None, :] * U_a)

print(f"Defender's Expected Utility: {model.objVal}")

print(f"Attacker's Expected Utility: {expected_utility}")
