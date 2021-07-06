import json
import sys

training_data = None

with open('./training_set_db.json') as file:
  training_data = json.load(file)

ratings = []
with open('ratings.txt', 'r') as file:
  for line in file:
    ratings.append(int(line.strip()))

if len(sys.argv) > 1:
  if sys.argv[1] == '--apply':
    # write ratings into training set
    for i, recipe in enumerate(training_data):
      recipe['rating'] = ratings[i]
      with open('./training_set_db.json', 'w') as file:
        json.dump(training_data, file, indent=2)
  else:
    print('--apply: overwrite ratings in the data set (cannot be undone)')
    print('--reset: delete all rating progress (cannot be undone)')
  sys.exit(0)

def to_string(ingredients):
  dct = {}

  for ingredient in recipe['ingredients']:
    if ingredient in dct:
      dct[ingredient] += 1
    else:
      dct[ingredient] = 1

  out = ''
  for k in dct:
    out += str(dct[k]) + ' ' + k + ', '

  return out[:-2]

def save(fname, ratings):
  with open(fname, 'w') as file:
    file.write('\n'.join([str(rating) for rating in ratings]))

end = len(training_data)
prev = len(ratings)

if prev == end:
  print('all drinks have been rated. call with --pass or manually clear ratings.txt to start over')

for i in range(prev, end):
  recipe = training_data[i]
  try:
    new_rating = input(str(i+1) + '/' + str(end) + ') ' + to_string(recipe['ingredients']) + ': ')

    if new_rating in ['q', 'e', 's', 'quit', 'exit', 'save']:
      raise Exception('Exited')
    else:
      ratings.append(new_rating)
      save('ratings.txt', ratings)
  except Exception as e:
    print('\nbyee')
    # save('ratings.txt', ratings)
    break

print('you did it!')
