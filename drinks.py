#!/bin/python3

import json
import os
import random

TITLE = '''
███╗   ███╗██╗██╗  ██╗      ██████╗       ███╗   ███╗███████╗███╗   ██╗██╗   ██╗
████╗ ████║██║╚██╗██╔╝     ██╔═══██╗      ████╗ ████║██╔════╝████╗  ██║██║   ██║
██╔████╔██║██║ ╚███╔╝█████╗██║   ██║█████╗██╔████╔██║█████╗  ██╔██╗ ██║██║   ██║
██║╚██╔╝██║██║ ██╔██╗╚════╝██║   ██║╚════╝██║╚██╔╝██║██╔══╝  ██║╚██╗██║██║   ██║
██║ ╚═╝ ██║██║██╔╝ ██╗     ╚██████╔╝      ██║ ╚═╝ ██║███████╗██║ ╚████║╚██████╔╝
╚═╝     ╚═╝╚═╝╚═╝  ╚═╝      ╚═════╝       ╚═╝     ╚═╝╚══════╝╚═╝  ╚═══╝ ╚═════╝
'''

# Force drinks to include this ingredient
# Note: set this to None when not in use
BASE_INGREDIENT = None
GENERATE_MENU = ['tequila', 'bourbon', 'white rum', 'vodka', 'gin']
GENERATE_MENU_UNIQUE = True

# Whether to restrict ingredients to those listed above
LIMIT_INGREDIENT_LIST = 'manual'

# Whether to allow updating the same edge >1 times during a training session
TRAIN_WITH_DUPLICATES = False

# Whether to print training status
VERBOSE = True

u = 0.1 # minimum weight
Y = 0.9 # maximum weight

w = 0.6 # default weight. higher = more experimental recipes
d = 0.1 # scale factor for updating edge weights
G = 8   # training iterations

W = 1   # in range (0, 1], lower number means higher bar for drink quality
z = 0.6 # in range (0, 1], lower number means more ingredients
q = 2   # > 1 means cocktail generation is more random, < 1 is less random

L = 5   # max number of total ingredients in a drink (not unique ingredients)
l = 3   # min number of total ingredients in a drink (not unique ingredients)

N = 4   # number of cocktails to generate
m = 0.6 # minimum acceptable average weight when generating cocktails
M = 0.8 # maximum acceptable average weight when generating cocktails

O = 5   # number of nuclei for automatic ingredient limitation
X = 6   # number of edges per nucleus for automatic ingredient limitation

# Data from files
ingredient_measures = None
with open('./data/measures.json') as file:
  ingredient_measures = json.load(file)

nouns = []
with open('./data/nouns.txt') as file:
  for line in file:
    nouns.append(line.strip())

adjs = []
with open('./data/adjs.txt') as file:
  for line in file:
    adjs.append(line.strip())

def key(a, b):
  if a > b:
    a, b = b, a

  return a + '.' + b

def construct_graph(ingredient_list):
  graph = {}
  sz = len(ingredient_list)

  # Initialize all weights to the default
  for i in range(sz):
    for j in range(i+1, sz):
      graph[key(ingredient_list[i], ingredient_list[j])] = w

  return graph

# Used during training, update weights for all combinations of ingredients in
#  the recipe
def update_recipe(graph, recipe, rating):
  if not TRAIN_WITH_DUPLICATES:
    recipe = list(set(recipe))

  sz = len(recipe)
  for i in range(sz):
    for j in range(i+1, sz):
      a, b = recipe[i], recipe[j]
      if a == b:
        continue
      update_edge(graph, a, b, rating)

def update_edge(graph, a, b, rating):
  # Step edges slightly toward the average of the rating + the previous weight
  k = key(a, b)

  avg = 0.5*(graph[k] + rating)
  delta = d*(graph[k] - avg)
  graph[k] = max(u, min(graph[k] - delta, Y))

def random_recipe(graph, ingredient_list, start=None):
  if start is None:
    start = random.choice(ingredient_list)

  recipe = [start]

  next_ingredient = start
  while True:
    max_rand_so_far = -1

    for ingredient in ingredient_list:
      if ingredient == start:
        # TODO: check if ingredient is in recipe so far?
        continue

      # Sometimes you want a drink with just 1 kind of hard liquor
      if GENERATE_MENU_UNIQUE and ingredient in GENERATE_MENU:
        continue

      biased_rand = random.random()*q*graph[key(start, ingredient)]

      if biased_rand > max_rand_so_far:
        max_rand_so_far = biased_rand
        next_ingredient = ingredient

    recipe.append(next_ingredient)

    # Check to make sure the recipe isn't too short (want to have interesting
    #  drinks)
    if len(recipe) < l:
      continue
    # This is so recipes don't get arbitrarily long by scaling up portions
    #  e.g. 2 parts tequila, 2 parts mango juice is the same as 1 of each, but
    #  one can imagine the walk going back and forth between two high-synergy
    #  ingredients many times
    elif len(recipe) >= L:
      break

    # Check if you're done making the drink
    # This fitness function has a higher standard for ingredient synergy the
    #  fewer unique ingredients there are in the recipe
    avg_weight = get_average_weight(graph, recipe)
    if avg_weight*W >= (1-z*len(set(recipe))/L):
      break

    else:
      start = next_ingredient

  return recipe

# Arrange the recipe in a more compact way
def recipe_to_dict(recipe):
  out = {}

  for ingredient in recipe:
    if ingredient in out:
      out[ingredient] += 1
    else:
      out[ingredient] = 1

  return out

# Arrange the recipe in a more human-friendly way
def recipe_to_string(quality, recipe):
  recipe = recipe_to_dict(recipe)

  out = f'The {random.choice(adjs)} {random.choice(nouns)} ({int(quality*100)}/100)\n'

  for ingredient, measure in recipe.items():
    unit = ingredient_measures[ingredient]

    if unit == 'tsp':
      measure *= 1.5

    if unit == 'half-oz':
      measure /= 2
      unit = 'oz'

    if unit == '2oz':
      measure *= 2
      unit = 'oz'

    if int(measure) == measure:
      measure = int(measure)

    out += f'{str(measure).rjust(3)} {unit.ljust(3)}  {ingredient}\n'

  return out

# Measure how synergistic the ingredients of this recipe are
def get_average_weight(graph, recipe):
  avg = 0
  num_edges = 0

  sz = len(recipe)
  for i in range(sz):
    for j in range(i+1, sz):
      a, b = recipe[i], recipe[j]
      if a == b:
        continue
      num_edges += 1
      avg += graph[key(a, b)]

  return avg/num_edges

# Loop through the known recipes and update ingredient weights accordingly
def train(training_file):
  ingredient_list = []
  recipes = []

  with open(training_file) as file:
    recipes = json.load(file)

  for recipe in recipes:
    for ingredient in recipe['ingredients']:
      if ingredient not in ingredient_list:
        ingredient_list.append(ingredient)

  weights = construct_graph(ingredient_list)

  for i in range(G):
    # Using this training method, the order in which training data is traversed
    #  affects the final weights. randomize to avoid bias when training for
    #  multiple generations
    random.shuffle(recipes)
    for recipe in recipes:
      update_recipe(weights, recipe['ingredients'], recipe['rating']/10.0)

  return weights, ingredient_list

# Useful for only creating a drink that uses a limited subset of the full
#  ingredient list
def get_subgraph(full_graph, ingredient_list):
  subgraph = {}
  sz = len(ingredient_list)

  for i in range(sz):
    for j in range(i+1, sz):
      k = key(ingredient_list[i], ingredient_list[j])
      subgraph[k] = full_graph[k]

  return subgraph

def identify_nuclei(graph, ingredient_list, n=20):
  scores = []

  for a in ingredient_list:
    acc = 0
    for b in ingredient_list:
      if a == b:
        continue
      else:
        acc += graph[key(a, b)]
    scores.append({
      'name': a,
      'score': acc,
    })

  scores.sort(reverse=True, key=lambda x: x['score'])
  return [s['name'] for s in scores[:n]]

def cluster_from_nuclei(graph, ingredient_list, nuclei, n=5):
  cool_kids = [nucleus for nucleus in nuclei]
  for nucleus in nuclei:
    edges = []
    for b in ingredient_list:
      if nucleus == b:
        continue
      else:
        edges.append({
          'edge': b,
          'weight': graph[key(nucleus, b)]
        })
    edges.sort(reverse=True, key=lambda x: x['weight'])
    cool_kids += [e['edge'] for e in edges[:n]]

  return list(set(cool_kids))

def print_cluster(weights, ingredients):
  nuclei = identify_nuclei(weights, ingredients, O)
  cluster = c5luster_from_nuclei(weights, ingredients, nuclei, X)

  cluster.sort()
  for i, ingredient in enumerate(cluster):
    print(f'{i+1}. {ingredient}')

# Load training data, train a graph, generate 10 recipes
def main():
  os.system('cls||clear')

  _weights, all_ingredients = train('./data/training_set_db.json')
  if LIMIT_INGREDIENT_LIST == 'manual':
    ingredients = []
    with open('./data/stock.json') as file:
      stock = json.load(file)
      for k, v in stock.items():
        if v:
          ingredients.append(k)
    weights = get_subgraph(_weights, ingredients)
  elif LIMIT_INGREDIENT_LIST == 'auto':
    nuclei = GENERATE_MENU
    if nuclei is None:
      nuclei = identify_nuclei(_weights, all_ingredients, O)
    ingredients = cluster_from_nuclei(_weights, all_ingredients, nuclei, X)
    weights = get_subgraph(_weights, ingredients)
    with open('./data/.auto-ingredients.txt', 'w') as file:
      file.write('\n'.join([f'{i+1}. {g}' for i, g in enumerate(sorted(ingredients))]))
  else:
    weights, ingredients = _weights, all_ingredients

  print(TITLE)

  sanity = 10000 # as time goes on, lower your standards

  # Special option to generate 1 drink for each of several ingredients
  if GENERATE_MENU is not None:
    for base_ingredient in GENERATE_MENU:
      sanity = 10000

      not_unique = False
      while True:
        recipe = random_recipe(weights, ingredients, start=base_ingredient)
        avg_weight = get_average_weight(weights, recipe)

        if avg_weight > m*sanity/10000 and avg_weight < M + (1-M)*(1-sanity/10000):
          print(recipe_to_string(avg_weight, recipe))
          break
        sanity = max(sanity-1, 0)
  else:
    n_recipes = 0  # number of recipes to try and make
    while n_recipes < N:
      recipe = random_recipe(weights, ingredients, start=BASE_INGREDIENT)
      avg_weight = get_average_weight(weights, recipe)

      # Check that the quality recipe is within the specified range of values
      #  (scaled by the sanity factor, so the more failures there are to find a
      #  recipe that is good enough, the lower the bar is set for "good enough")
      # Note: at sanity == 0, any recipe will be accepted as good enough
      if avg_weight > m*sanity/10000 and avg_weight < M + (1-M)*(1-sanity/10000):
        print(recipe_to_string(avg_weight, recipe))
        n_recipes += 1
      sanity = max(sanity-1, 0)

if __name__ == '__main__':
  main()
