ACTION_SPACE_SIZE = 14
NUM_OF_PLAYERS = 2


def convert_num_to_action_array(unique_action_index):
    # TODO: CHECK THIS OUT
    action_array = []
    # Left to right?
    exponent = NUM_OF_PLAYERS-1
    index = unique_action_index
    while index != 0:
        temporary_number = 0
        while ACTION_SPACE_SIZE**exponent <= index:
            print(ACTION_SPACE_SIZE**exponent, index)
            temporary_number += 1
            index -= ACTION_SPACE_SIZE**exponent
        exponent -= 1
        action_array.append(temporary_number)
    while len(action_array) < NUM_OF_PLAYERS:
        action_array = [0] + action_array
    return action_array


actions = [13]
unique_action_index = 0
for i in range(len(actions)):
  print(i)
  unique_action_index += actions[-i-1]*ACTION_SPACE_SIZE**i
print(unique_action_index)
print(convert_num_to_action_array(unique_action_index))
