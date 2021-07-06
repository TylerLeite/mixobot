import json
import random

ingredients = [
  'tequila',
  'chambord',
  'sour mix',
  'simple syrup',
  'coca cola',
  'soda water',
  'blue curacao',
  'orange juice',
]

w = 0.1 # default weight. currently 0.1 for large data sets, 0.5 for small ones

u = 0.1 # minimum weight
Y = 0.9 # maximum weight

W = 1.5  # in range (0, 1], lower number means higher bar for drink quality
n = 1  # in range (0, 1] lower number means more ingredients
L = 5 # max number of total ingredients in a drink (not unique ingredients)
l = 3 # min number of total ingredients in a drink (not unique ingredients)

G = 1 # training iterations

def key(a, b):
  if a > b:
    a, b = b, a

  return a + '.' + b

def construct_graph(ingredient_list):
  graph = {}
  sz = len(ingredient_list)

  for i in range(sz):
    for j in range(i+1, sz):
      graph[key(ingredient_list[i], ingredient_list[j])] = w

  return graph

# Used during training, update weights for all combinations of ingredients in
#  the recipe
def update_recipe(graph, recipe, rating):
  sz = len(recipe)
  for i in range(sz):
    for j in range(i+1, sz):
      a, b = recipe[i], recipe[j]
      if a == b:
        continue
      update_edge(graph, a, b, rating)

def update_edge(graph, a, b, rating):
  # Average the previous weight and the rating, clamp to the specified range
  # TODO: step by a smaller amount? averaging is probably unstable
  k = key(a, b)
  graph[k] = max(u, min(0.5*(graph[k] + rating), Y))

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

      biased_rand = random.random()*graph[key(start, ingredient)]
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
    if avg_weight*W >= (1-n*len(set(recipe))/len(ingredients)):
      break

    else:
      start = next_ingredient

  return recipe

# Arrange the recipe in a more human-friendly way
def recipe_to_dict(recipe):
  out = {}

  for ingredient in recipe:
    if ingredient in out:
      out[ingredient] += 1
    else:
      out[ingredient] = 1

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
  ingredients = []
  recipes = []

  with open(training_file) as file:
    recipes = json.load(file)

  for i in range(G):
    for recipe in recipes:
      for ingredient in recipe['ingredients']:
        if ingredient not in ingredients:
          ingredients.append(ingredient)

  weights = construct_graph(ingredients)

  for recipe in recipes:
    update_recipe(weights, recipe['ingredients'], recipe['rating']/10.0)

  return weights, ingredients

# Useful for only creating a drink that uses a limited subset of the full
#  ingredient list
def get_subgraph(full_graph, ingredient_list):
  subgraph = {}
  sz = len(ingredient_list)

  # all weights are initialized to 0.5
  for i in range(sz):
    for j in range(i+1, sz):
      k = key(ingredient_list[i], ingredient_list[j])
      subgraph[k] = full_graph[k]

  return subgraph

# Load training data, train a graph, generate 10 recipes
def main():
  _weights, all_ingredients = train('./data/training_set_db.json')
  weights = get_subgraph(_weights, ingredients)

  recipes = []
  while len(recipes) < 10:
    recipe = random_recipe(weights, ingredients, 'tequila')
    if get_average_weight(weights, recipe) > 0.8:
      recipes.append(recipe_to_dict(recipe))

  [print(recipe) for recipe in recipes]

if __name__ == "__main__":
  main()
