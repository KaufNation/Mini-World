'''
Developed by: Joshua Kaufman
License: Apache License 2.0
Note: 
	Built on python 3.6.1
	Uses EvoNN algorithm which can be found in my other repo
	It is a small simulation which, over time, the creatures master. As with
	all evolutionary algorithms, outcome is random and not guaranteed.  A few thousand
	generation may have to be run before reaching an evolutionary optimized creature. The 
	world is set up where food is always at least x amount of food in the world.  Creatures 
	can eat food (green) or dead creature (red). Living, moving, thinking, all use energy 
	so creatures that dont scavenge for food die out.  A host of other variables effect how the 
	creatures react.  Feel free to mess around with the global variables below which woll effect 
	the outcome of the game.
	To see an optimized creature, check out this link: https://youtu.be/GaS_d8Q2y1o
Enjoy
'''
import sys, pygame
import random
import math
import time
import copy
import io
from EvoNN import NeuralNet


sintable = [
    0.00000, 0.01745, 0.03490, 0.05234, 0.06976, 0.08716, 0.10453,
    0.12187, 0.13917, 0.15643, 0.17365, 0.19081, 0.20791, 0.22495, 0.24192,
    0.25882, 0.27564, 0.29237, 0.30902, 0.32557, 0.34202, 0.35837, 0.37461,
    0.39073, 0.40674, 0.42262, 0.43837, 0.45399, 0.46947, 0.48481, 0.50000,
    0.51504, 0.52992, 0.54464, 0.55919, 0.57358, 0.58779, 0.60182, 0.61566,
    0.62932, 0.64279, 0.65606, 0.66913, 0.68200, 0.69466, 0.70711, 0.71934,
    0.73135, 0.74314, 0.75471, 0.76604, 0.77715, 0.78801, 0.79864, 0.80902,
    0.81915, 0.82904, 0.83867, 0.84805, 0.85717, 0.86603, 0.87462, 0.88295,
    0.89101, 0.89879, 0.90631, 0.91355, 0.92050, 0.92718, 0.93358, 0.93969,
    0.94552, 0.95106, 0.95630, 0.96126, 0.96593, 0.97030, 0.97437, 0.97815,
    0.98163, 0.98481, 0.98769, 0.99027, 0.99255, 0.99452, 0.99619, 0.99756,
    0.99863, 0.99939, 0.99985, 1.00000, 0.99985, 0.99939, 0.99863, 0.99756,
    0.99619, 0.99452, 0.99255, 0.99027, 0.98769, 0.98481, 0.98163, 0.97815,
    0.97437, 0.97030, 0.96593, 0.96126, 0.95630, 0.95106, 0.94552, 0.93969,
    0.93358, 0.92718, 0.92050, 0.91355, 0.90631, 0.89879, 0.89101, 0.88295,
    0.87462, 0.86603, 0.85717, 0.84805, 0.83867, 0.82904, 0.81915, 0.80902,
    0.79864, 0.78801, 0.77715, 0.76604, 0.75471, 0.74314, 0.73135, 0.71934,
    0.70711, 0.69466, 0.68200, 0.66913, 0.65606, 0.64279, 0.62932, 0.61566,
    0.60182, 0.58779, 0.57358, 0.55919, 0.54464, 0.52992, 0.51504, 0.50000,
    0.48481, 0.46947, 0.45399, 0.43837, 0.42262, 0.40674, 0.39073, 0.37461,
    0.35837, 0.34202, 0.32557, 0.30902, 0.29237, 0.27564, 0.25882, 0.24192,
    0.22495, 0.20791, 0.19081, 0.17365, 0.15643, 0.13917, 0.12187, 0.10453,
    0.08716, 0.06976, 0.05234, 0.03490, 0.01745, 0.00000, -0.01745, -0.03490,
    -0.05234, -0.06976, -0.08716, -0.10453, -0.12187, -0.13917, -0.15643,
    -0.17365, -0.19081, -0.20791, -0.22495, -0.24192, -0.25882, -0.27564,
    -0.29237, -0.30902, -0.32557, -0.34202, -0.35837, -0.37461, -0.39073,
    -0.40674, -0.42262, -0.43837, -0.45399, -0.46947, -0.48481, -0.50000,
    -0.51504, -0.52992, -0.54464, -0.55919, -0.57358, -0.58779, -0.60182,
    -0.61566, -0.62932, -0.64279, -0.65606, -0.66913, -0.68200, -0.69466,
    -0.70711, -0.71934, -0.73135, -0.74314, -0.75471, -0.76604, -0.77715,
    -0.78801, -0.79864, -0.80902, -0.81915, -0.82904, -0.83867, -0.84805,
    -0.85717, -0.86603, -0.87462, -0.88295, -0.89101, -0.89879, -0.90631,
    -0.91355, -0.92050, -0.92718, -0.93358, -0.93969, -0.94552, -0.95106,
    -0.95630, -0.96126, -0.96593, -0.97030, -0.97437, -0.97815, -0.98163,
    -0.98481, -0.98769, -0.99027, -0.99255, -0.99452, -0.99619, -0.99756,
    -0.99863, -0.99939, -0.99985, -1.00000, -0.99985, -0.99939, -0.99863,
    -0.99756, -0.99619, -0.99452, -0.99255, -0.99027, -0.98769, -0.98481,
    -0.98163, -0.97815, -0.97437, -0.97030, -0.96593, -0.96126, -0.95630,
    -0.95106, -0.94552, -0.93969, -0.93358, -0.92718, -0.92050, -0.91355,
    -0.90631, -0.89879, -0.89101, -0.88295, -0.87462, -0.86603, -0.85717,
    -0.84805, -0.83867, -0.82904, -0.81915, -0.80902, -0.79864, -0.78801,
    -0.77715, -0.76604, -0.75471, -0.74314, -0.73135, -0.71934, -0.70711,
    -0.69466, -0.68200, -0.66913, -0.65606, -0.64279, -0.62932, -0.61566,
    -0.60182, -0.58779, -0.57358, -0.55919, -0.54464, -0.52992, -0.51504,
    -0.50000, -0.48481, -0.46947, -0.45399, -0.43837, -0.42262, -0.40674,
    -0.39073, -0.37461, -0.35837, -0.34202, -0.32557, -0.30902, -0.29237,
    -0.27564, -0.25882, -0.24192, -0.22495, -0.20791, -0.19081, -0.17365,
    -0.15643, -0.13917, -0.12187, -0.10453, -0.08716, -0.06976, -0.05234,
    -0.03490, -0.01745, -0.00000
]
 
costable = [
    1.00000, 0.99985, 0.99939, 0.99863, 0.99756, 0.99619, 0.99452,
    0.99255, 0.99027, 0.98769, 0.98481, 0.98163, 0.97815, 0.97437, 0.97030,
    0.96593, 0.96126, 0.95630, 0.95106, 0.94552, 0.93969, 0.93358, 0.92718,
    0.92050, 0.91355, 0.90631, 0.89879, 0.89101, 0.88295, 0.87462, 0.86603,
    0.85717, 0.84805, 0.83867, 0.82904, 0.81915, 0.80902, 0.79864, 0.78801,
    0.77715, 0.76604, 0.75471, 0.74314, 0.73135, 0.71934, 0.70711, 0.69466,
    0.68200, 0.66913, 0.65606, 0.64279, 0.62932, 0.61566, 0.60182, 0.58779,
    0.57358, 0.55919, 0.54464, 0.52992, 0.51504, 0.50000, 0.48481, 0.46947,
    0.45399, 0.43837, 0.42262, 0.40674, 0.39073, 0.37461, 0.35837, 0.34202,
    0.32557, 0.30902, 0.29237, 0.27564, 0.25882, 0.24192, 0.22495, 0.20791,
    0.19081, 0.17365, 0.15643, 0.13917, 0.12187, 0.10453, 0.08716, 0.06976,
    0.05234, 0.03490, 0.01745, 0.00000, -0.01745, -0.03490, -0.05234, -0.06976,
    -0.08716, -0.10453, -0.12187, -0.13917, -0.15643, -0.17365, -0.19081,
    -0.20791, -0.22495, -0.24192, -0.25882, -0.27564, -0.29237, -0.30902,
    -0.32557, -0.34202, -0.35837, -0.37461, -0.39073, -0.40674, -0.42262,
    -0.43837, -0.45399, -0.46947, -0.48481, -0.50000, -0.51504, -0.52992,
    -0.54464, -0.55919, -0.57358, -0.58779, -0.60182, -0.61566, -0.62932,
    -0.64279, -0.65606, -0.66913, -0.68200, -0.69466, -0.70711, -0.71934,
    -0.73135, -0.74314, -0.75471, -0.76604, -0.77715, -0.78801, -0.79864,
    -0.80902, -0.81915, -0.82904, -0.83867, -0.84805, -0.85717, -0.86603, 
    -0.87462, -0.88295, -0.89101, -0.89879, -0.90631, -0.91355, -0.92050,
    -0.92718, -0.93358, -0.93969, -0.94552, -0.95106, -0.95630, -0.96126,
    -0.96593, -0.97030, -0.97437, -0.97815, -0.98163, -0.98481, -0.98769,
    -0.99027, -0.99255, -0.99452, -0.99619, -0.99756, -0.99863, -0.99939,
    -0.99985, -1.00000, -0.99985, -0.99939, -0.99863, -0.99756, -0.99619,
    -0.99452, -0.99255, -0.99027, -0.98769, -0.98481, -0.98163, -0.97815,
    -0.97437, -0.97030, -0.96593, -0.96126, -0.95630, -0.95106, -0.94552,
    -0.93969, -0.93358, -0.92718, -0.92050, -0.91355, -0.90631, -0.89879,
    -0.89101, -0.88295, -0.87462, -0.86603, -0.85717, -0.84805, -0.83867,
    -0.82904, -0.81915, -0.80902, -0.79864, -0.78801, -0.77715, -0.76604,
    -0.75471, -0.74314, -0.73135, -0.71934, -0.70711, -0.69466, -0.68200,
    -0.66913, -0.65606, -0.64279, -0.62932, -0.61566, -0.60182, -0.58779,
    -0.57358, -0.55919, -0.54464, -0.52992, -0.51504, -0.50000, -0.48481,
    -0.46947, -0.45399, -0.43837, -0.42262, -0.40674, -0.39073, -0.37461,
    -0.35837, -0.34202, -0.32557, -0.30902, -0.29237, -0.27564, -0.25882,
    -0.24192, -0.22495, -0.20791, -0.19081, -0.17365, -0.15643, -0.13917,
    -0.12187, -0.10453, -0.08716, -0.06976, -0.05234, -0.03490, -0.01745,
    -0.00000, 0.01745, 0.03490, 0.05234, 0.06976, 0.08716, 0.10453, 0.12187,
    0.13917, 0.15643, 0.17365, 0.19081, 0.20791, 0.22495, 0.24192, 0.25882,
    0.27564, 0.29237, 0.30902, 0.32557, 0.34202, 0.35837, 0.37461, 0.39073,
    0.40674, 0.42262, 0.43837, 0.45399, 0.46947, 0.48481, 0.50000, 0.51504,
    0.52992, 0.54464, 0.55919, 0.57358, 0.58779, 0.60182, 0.61566, 0.62932,
    0.64279, 0.65606, 0.66913, 0.68200, 0.69466, 0.70711, 0.71934, 0.73135,
    0.74314, 0.75471, 0.76604, 0.77715, 0.78801, 0.79864, 0.80902, 0.81915,
    0.82904, 0.83867, 0.84805, 0.85717, 0.86603, 0.87462, 0.88295, 0.89101,
    0.89879, 0.90631, 0.91355, 0.92050, 0.92718, 0.93358, 0.93969, 0.94552,
    0.95106, 0.95630, 0.96126, 0.96593, 0.97030, 0.97437, 0.97815, 0.98163,
    0.98481, 0.98769, 0.99027, 0.99255, 0.99452, 0.99619, 0.99756, 0.99863,
    0.99939, 0.99985, 1.00000
]

pygame.init()

#Colors
BLACK = 0, 0, 0
WHITE = 255,255,255
RED = 255,0,0
GREEN = 0,255,0
BLUE = 0,0,255
GREY = 127,127,127
LIGHT_GREY = 191,191,191
VERY_LIGHT_GREY = 223,223,223
DARK_GREY = 63,63,63
VERY_DARK_GREY = 31,31,31

TOTAL_PLAYBACK_MULTIPLYER = 3
#Global Creature Variables
MAX_HIDDEN_NEURONS = 20
MIN_HIDDEN_NEURONS = 4
MAX_CREATURE_SIZE = 20
MIN_CREATURE_SIZE = 3
MAX_NUM_EYES = 10
MUTATION_CHANCE = .1 # probability of mutiation
MUTATION_VARIABILITY = .05 #max mutation is this time max of that charachteristic
NEURON_MULTIPLYER = 20 #Increases neuron growth chance by 20
MAX_CREATURE_SPEED = 5
BASAL_ENERGY_LOSS_RATE = .05*TOTAL_PLAYBACK_MULTIPLYER #cost of just staying alive
ENERGY_DEATH_THRESHOLD = 10 #higher the value, the less energy a creature can have
ENERGY_COST_OF_NEURON = .005*TOTAL_PLAYBACK_MULTIPLYER
ENERGY_COST_OF_NEURAL_PATH = .001*TOTAL_PLAYBACK_MULTIPLYER
MOVEMENT_ENERGY_MULT = .01*TOTAL_PLAYBACK_MULTIPLYER #higher, the more it costs to move
NUM_THINK_CYCLES = 1 #number of cycles the brain will think for before rading neurons

#Mutation Variables
VARIABILITY = .5 #no max, should be <= 1 #larger the variability, the large weights change during mutations
WEIGHT_CHANGE_PROB = .05 #max 1 #larger the weight_change_prob the higher the chance is of any given weight being altered
CONNECTION_CHANGE_PROB = .02 #max 1 #larger the connection_change_prob the higher the chance of a connection being deleted or added
NEURON_CHANGE_PROB = .02 #max 1 #larger the neuron_change_prob the higher the chance a neuron can be added or removed


#Global Screen Variable
MULT = 3
SIZE = WIDTH, HEIGHT = 320*MULT, 240*MULT

#Global Sim Variables
MIN_NUM_CREATURES = 12 #must be divisible by 3
MIN_FOOD_AVAILABLE = 10000
MAX_FOOD_SIZE = 7
MIN_FOOD_SIZE = 2
MOVEMENT_MULT = 10*TOTAL_PLAYBACK_MULTIPLYER
FOOD_ENERGY_MULTIPLE = 25
CREATURE_ENERGY_MULTIPLE = 3

def sigmoid(x):
	if x > 20:
		x = 20
	if x < -20:
		x = -20
	return 1 / (1 + math.exp(-x))

#####Give creatures a score based on how much energy they eat in their life

'''
DNA stores the information necessary to build the creature

VARIABLES:
size
number of hidden neurons
hidden neuron connections

Input neurons:
sight xrgb
speed (1)
rotate speed (1)
energy	(1)
attack and health (2) - not yet

Output Neurons:
speed (1)
rotate speed (1)
procreate (1)
'''

class DNA:
	def __init__(self):
		#defining DNA charachteristics
		self.max_size = MIN_CREATURE_SIZE + random.random()*MAX_CREATURE_SIZE

		#Create NN / brain	
		self.brain = NeuralNet()

		#Add inputs to the brain
		self.brain.inputs.append(['eyes', 3, [1,MAX_NUM_EYES]])
		self.brain.inputs.append(['speed', 1, []])
		self.brain.inputs.append(['rotation speed', 1, []])
		self.brain.inputs.append(['energy', 1, []])

		#Add outputs to the brain
		self.brain.outputs.append(['speed', 1, []])
		self.brain.outputs.append(['rotation speed', 1, []])
		self.brain.outputs.append(['procreate', 1, []])

		#define hidden layers
		self.brain.hidden = [[MIN_HIDDEN_NEURONS,MAX_HIDDEN_NEURONS],0]

		self.brain.build()

		#updating legacy variables
		self.num_i_neurons = self.brain.num_active_i_neurons
		self.num_o_neurons = self.brain.num_active_o_neurons
		self.num_io_neurons = self.num_i_neurons+self.num_o_neurons
		self.num_hidden_neurons = sum(self.brain.num_active_h_neurons)
		self.num_neurons = len(self.brain.neuron_list)
		self.num_neural_pathways = self.brain.num_connections
		self.num_eyes = int(self.brain.inputs[0][4][0]/3)
		self.eye_offset = random.random()*2*math.pi/self.num_eyes

		#self.brain.show()

		return


def degrees(rad):
	while rad < 0:
		rad += 2*math.pi
	while rad > 2*math.pi:
		rad -= 2*math.pi
	return int(rad/math.pi*180)

class Creature:
	def __init__(self):
		#init dna
		self.dna = DNA()
		self.speed = 0.0
		self.color = BLUE
		self.sight = []
		for i in range(0,self.dna.num_eyes):
			self.sight.append([0.0,0.0,0.0])
		self.eye_offset = self.dna.eye_offset
		self.rotate_speed = 0.0
		self.max_energy = math.pi*self.dna.max_size**2 #energy based off area
		self.energy = self.max_energy/2
		self.size = (self.energy/math.pi)**.5 #size coresponds to energy
		self.max_speed = MAX_CREATURE_SPEED/(self.size**.5) 
		self.move_energy_loss = self.speed*MOVEMENT_ENERGY_MULT
		self.brain_energy_cost = ENERGY_COST_OF_NEURAL_PATH*self.dna.num_neural_pathways + self.dna.num_hidden_neurons*ENERGY_COST_OF_NEURON
		self.basal_energy_cost = self.size*BASAL_ENERGY_LOSS_RATE
		self.death_threshold = self.max_energy/ENERGY_DEATH_THRESHOLD
		self.direction = (random.random()-.5)*2*math.pi
		self.pos = [int(random.random()*WIDTH),int(random.random()*HEIGHT)]
		self.procreate = 0.0
		self.new_energy = 0.0
		self.dead = False
		self.life_span = 0
		self.gen_life_span = 0
		self.score = 0.0

		return

	def update(self):
		if not self.dead:
			#death check
			if self.energy < self.death_threshold:
				self.dead = True
				dead_creature_list.append(copy.deepcopy(self))
				return
			else:
				self.life_span += 1
				self.gen_life_span +=1
			#update brain
			self.think()
			out = self.dna.brain.get_outputs()
			if self.size < MIN_CREATURE_SIZE:
				self.size = MIN_CREATURE_SIZE
			#use the output neurons to modify current situation
			#neuron arangement is as follows: speed[x], speed[y], rotate_speed, procreate
			range_start = self.dna.num_i_neurons+self.dna.num_hidden_neurons
			self.max_speed = MAX_CREATURE_SPEED/(self.size)
			self.speed = out[0][1]*self.max_speed
			self.rotate_speed = (out[2][1]-.5)
			self.procreate = out[2][1]

			#apply movement to creature
			self.pos[0] += self.speed*MOVEMENT_MULT*costable[degrees(self.direction)]
			self.pos[1] += self.speed*MOVEMENT_MULT*sintable[degrees(self.direction)]
			####print(self.speed[0]*MOVEMENT_MULT,self.speed[1]*MOVEMENT_MULT)
			self.pos[0], self.pos[1] = check_in_bounds(self.pos[0], self.pos[1])
			####print(self.pos)
			self.direction += self.rotate_speed*MOVEMENT_MULT/20
			
			#update score
			self.score += 1.0

			#update size
			self.size = math.ceil((self.energy/math.pi)**.5)
			#modify energy
			self.move_energy_loss = self.speed*(math.pi*(self.size**2))*MOVEMENT_ENERGY_MULT
			self.brain_energy_cost = ENERGY_COST_OF_NEURAL_PATH*self.dna.num_neural_pathways + self.dna.num_hidden_neurons*ENERGY_COST_OF_NEURON
			self.basal_energy_cost = self.size*BASAL_ENERGY_LOSS_RATE
			self.energy -= (self.move_energy_loss + self.brain_energy_cost + self.basal_energy_cost)
			#procreate check
			if self.procreate > 0.5:
				self.split()
			self.color = BLUE
		else:
			self.color = RED

		return

	def trace_eyes(self):
		#get the color the eye beam sees
		direction = self.direction - (self.dna.num_eyes*self.dna.eye_offset)/2
		i = 0
		x = 0
		flop = 0
		while x/3 != self.dna.num_eyes:
			start = self.pos[0]+costable[degrees(direction)]*self.size, self.pos[1]+sintable[degrees(direction)]*self.size
			final_pos, self.sight[int(x/3)] = extrapolate_line(self, start,direction,creature_list,food)
			if self.dna.brain.neuron_list[i] != None:
				self.dna.brain.neuron_list[i].cur_value = float(self.sight[int(x/3)][flop])/float(255)
				flop += 1
				x += 1
				if x%3 == 0:
					flop = 0
					direction += self.dna.eye_offset
			i += 1

		x = 0
		while i < self.dna.num_eyes:
			tot = 0
			if self.dna.brain.neuron_list[x] != None:
				tot+=self.dna.brain.neuron_list[x*3].cur_value+self.dna.brain.neuron_list[x*3+1].cur_value+self.dna.brain.neuron_list[x*3+2].cur_value
				if tot > 1.0:
					print("Creature eye saw more than 1 color!!!!????")
				i += 1
		return 

	def think(self):
		#update input neurons
		#neuron arangement is as follows: sight xrgb(3), speed[x], speed[y], rotate speed, energy
		#updates eyes
		self.trace_eyes()
		#update rest
		next_neuron = MAX_NUM_EYES*3
		self.dna.brain.neuron_list[next_neuron].cur_value = self.speed/self.max_speed
		self.dna.brain.neuron_list[next_neuron+1].cur_value = self.rotate_speed + .5
		self.dna.brain.neuron_list[next_neuron+2].cur_value = self.energy/self.max_energy
		
		self.dna.brain.think(NUM_THINK_CYCLES)
		return

	def split(self):
		#return
		self.energy = self.energy/3
		self.score += self.energy/6
		creature_list.append(copy.deepcopy(self))
		last = len(creature_list) - 1
		creature_list[last].score = 0.0
		creature_list[last].mutate()

		return

	def mutate(self):
		#DNA
		#change pos
		self.pos[0] += random.randint(0,100) - 50
		self.pos[1] += random.randint(0,100) - 50
		#defining DNA charachteristics
		#change size of creature
		if random.random() < MUTATION_CHANCE:
			self.dna.max_size += (random.random()-.5)*MAX_CREATURE_SIZE*MUTATION_VARIABILITY
			if self.dna.max_size > MAX_CREATURE_SIZE:
				self.dna.max_size = MAX_CREATURE_SIZE
		#change num eyes
		if random.random() < MUTATION_CHANCE:
			eye_change = [-1,1][random.randint(0,1)]
			if self.dna.num_eyes + eye_change >= 1 and self.dna.num_eyes + eye_change <= MAX_NUM_EYES:
				self.dna.brain.change_input('eyes',eye_change)
				if eye_change > 0:
					self.sight.append([0.0,0.0,0.0])
				else:
					del self.sight[0]
				self.dna.num_eyes += eye_change
		#update eye_offset if value too large
		if self.eye_offset*self.dna.num_eyes > 2*math.pi:
			self.eye_offset = 2*math.pi/self.dna.num_eyes
		#mutate eye_offset
		if random.random() < MUTATION_CHANCE:
			self.eye_offset += math.pi*2*MUTATION_VARIABILITY*(random.random()-.5)
			if self.eye_offset*self.dna.num_eyes > 2*math.pi:
				self.eye_offset = 2*math.pi / self.dna.num_eyes
			if self.eye_offset < .01:
				self.eye_offset = .01
		#update energy stuff
		self.max_energy = math.pi*self.dna.max_size**2 #energy based off area
		#update neuron information
		self.num_i_neurons = self.dna.brain.num_active_i_neurons
		self.num_o_neurons = self.dna.brain.num_active_o_neurons
		self.num_io_neurons = self.num_i_neurons+self.num_o_neurons
		self.num_hidden_neurons = sum(self.dna.brain.num_active_h_neurons)
		self.num_neurons = self.num_io_neurons+self.num_hidden_neurons

		self.dna.brain.mutate(VARIABILITY, WEIGHT_CHANGE_PROB, CONNECTION_CHANGE_PROB, NEURON_CHANGE_PROB)

		return

def init_sim(screen, food, creature_list):
	screen.fill(BLACK)
	creature_list = generate_creatures(MIN_NUM_CREATURES)
	place_food(screen, food)
	place_creature(screen, creature_list, food)
	return creature_list


def generate_creatures(num):
	creature_list = []
	for i in range(0,num):
		creature_list.append(Creature())
	return creature_list

def place_creature(screen, creature_list, food):
	for creature in creature_list:
		pygame.draw.circle(screen, creature.color, (int(creature.pos[0]),int(creature.pos[1])), int(creature.size), 0)
		direction = creature.direction - (creature.dna.num_eyes*creature.dna.eye_offset)/2
		for i in range(0,creature.dna.num_eyes):
			start = creature.pos[0]+costable[degrees(direction)]*creature.size, creature.pos[1]+sintable[degrees(direction)]*creature.size
			final_pos = start
			color = (0,0,0)
			if creature.dead != True:
				final_pos, color = extrapolate_line(creature, start,direction,creature_list,food)
			pygame.draw.line(screen, VERY_LIGHT_GREY, start, final_pos, 1)
			direction += creature.dna.eye_offset
	return

def get_angle(c1,c2):
	x = c1[0] - c2[0]
	y = c1[1] - c2[1]
	angle = math.atan2(y,x)
	return scrub_angle(angle)

def scrub_angle(angle):
	twopi = 2*math.pi
	while angle < 0:
		angle += twopi
	while angle > twopi:
		angle -= twopi
	return angle


def extrapolate_line(creature, start, direction, creature_list, food):
	x = start[0]
	y = start[1]
	new_energy = 0.0
	direct = scrub_angle(direction+math.pi)
	for i in range(1,len(food)):
		f_x = food[i][0][0]
		f_y = food[i][0][1]
		angle = get_angle([x,y],[f_x,f_y])
		if angle > direct:
			angle = angle - direct
		else:
			angle = direct - angle
		length = ((f_x - x)**2 + (f_y - y)**2)**.5
		radius = food[i][2]
		color = food[i][3]
		if (creature.size + radius >= length):
			energy = food[i][1]	
			if energy < (creature.max_energy - creature.energy):
				creature.energy += energy
				creature.score += energy
				food[i][2] = 0
				food[i][1] = 0
			else:
				energy = creature.max_energy - creature.energy
				creature.energy += energy
				creature.score += energy
				food[i][1] -= energy
				food[i][2] = math.ceil(food[i][1]/FOOD_ENERGY_MULTIPLE)
			food[0] -= energy
		dist = length*math.atan(angle)
		if radius > dist:
			return [f_x,f_y],color
	count = 0
	for item in food[1:]:
		if item[1] == 0:
			food.remove(item)
		count += 1
	
	for i in range(0,len(creature_list)):
		f_x = creature_list[i].pos[0]
		f_y = creature_list[i].pos[1]
		angle = get_angle([x,y],[f_x,f_y])
		if angle > direct:
			angle = angle - direct
		else:
			angle = direct - angle
		length = ((f_x - x)**2 + (f_y - y)**2)**.5
		radius = creature_list[i].size
		color = creature_list[i].color
		if (creature.size + radius >= length and creature_list[i].energy > 0 and color == RED):
			energy = creature_list[i].energy*CREATURE_ENERGY_MULTIPLE
			if energy < (creature.max_energy - creature.energy):
				creature.energy += energy
				creature.score += energy
				creature_list[i].energy = 0
				creature_list[i].size = 0
			else:
				energy = creature.max_energy - creature.energy
				creature.energy += energy
				creature.score += energy
				creature_list[i].energy -= energy/CREATURE_ENERGY_MULTIPLE
				if creature_list[i].energy < 0:
					print("Energy in negatives")
					exit()
				creature_list[i].size = (creature_list[i].energy/math.pi)**.5
		dist = length*math.atan(angle)
		if radius > dist:
			return [f_x,f_y],color
	for creat in creature_list:
		if creat.energy == 0 and creat not in remove_creature:
			remove_creature.append(creat)
	
	x,y = check_in_bounds(x,y)	
	return [x,y], BLACK

def check_in_bounds(x,y):
	if x > WIDTH:
		x = WIDTH
	if x < 0:
		x = 0
	if y > HEIGHT:
		y = HEIGHT
	if y < 0:
		y = 0
	return x,y

def place_food(screen, food):
	while food[0] < MIN_FOOD_AVAILABLE:
		pos = [int(WIDTH*random.random()), int(HEIGHT*random.random())]
		radius = random.randint(MIN_FOOD_SIZE,MAX_FOOD_SIZE)
		amount = radius*FOOD_ENERGY_MULTIPLE
		color = GREEN
		food.append([pos,amount,radius,color])
		food[0] += amount
	for i in range(1, len(food)):
		pygame.draw.circle(screen, GREEN, food[i][0], int(food[i][1]/FOOD_ENERGY_MULTIPLE), 0)
	return

def main():
	#add in seeing the brain while it plays, and highlighitng the creatures whos brain it is
	#save out top preformers
	creature = Creature()
	global screen
	screen = pygame.display.set_mode(SIZE)
	screen.set_alpha(None)
	#food list needs to be global
	global food
	food = []
	food.append(0.0)
	#creature list needs to be global
	global creature_list
	global dead_creature_list
	#global remove_food
	global remove_creature
	remove_creature = []
	creature_list = []
	creature_list = init_sim(screen, food, creature_list)
	pygame.display.flip()
	gen = 1
	while 1:
		dead_creature_list = []
		in_game_time = 0
		while 1:
			screen.fill(BLACK)
			for event in pygame.event.get():
				if event.type == pygame.QUIT: 
					sys.exit()
			live_count = 0
			for i in range(0,len(creature_list)):
				creature_list[i].update()
				creature_count = len(creature_list)
				if not creature_list[i].dead:
					live_count += 1

			if len(remove_creature) > 0:
				for creature in remove_creature:
					dead_creature_list.append(copy.deepcopy(creature))
					creature_list.remove(creature)
			remove_creature = []
			place_food(screen, food)
			place_creature(screen, creature_list, food)
			pygame.display.flip()
			in_game_time += 1
			if live_count == 0:
				break
		dead_creature_list.sort(key=lambda x: x.score, reverse=True)
		creature_list = []
		#add top creatures from last gen
		for i in range(0,int(MIN_NUM_CREATURES/3)):
			creature_list.append(copy.deepcopy(dead_creature_list[i]))
			creature_list[i].dead = False
			creature_list[i].energy = creature_list[i].max_energy/2
			creature_list[i].pos = [int(random.random()*WIDTH),int(random.random()*HEIGHT)]
		#add a mutated copy of top creatures from last gen
		for i in range(0,int(MIN_NUM_CREATURES/3)):
			creature_list.append(copy.deepcopy(creature_list[i]))
			size = len(creature_list)
			creature_list[size-1].dna.brain.mutate(VARIABILITY, WEIGHT_CHANGE_PROB, CONNECTION_CHANGE_PROB, NEURON_CHANGE_PROB)
			creature_list[size-1].pos = [int(random.random()*WIDTH),int(random.random()*HEIGHT)]
		#add random other creatures
		for i in range(0,int(MIN_NUM_CREATURES/3)):
			creature_list.append(Creature())
		print("gen " + str(gen) + " all died :(")
		gen += 1

	return

main()


