import random


def state_generator(number_of_blocks):
    # A function which generates a random state for the blocks world problem
    # number_of_blocks - the number of blocks of the problem
    random.seed()  # initialize random seed
    no_of_stacks = random.randint(1, number_of_blocks)  # generate random number of stacks
    state_list = []
    iterator = no_of_stacks

    while iterator >= 0:
        state_list.append([-1, ])  # Add no_of_stacks "non-empty" lists to state_list
        iterator = iterator - 1

    for iterator in range(number_of_blocks):
        stack_number = random.randint(0, no_of_stacks - 1)  # Add a random block to a random stack
        state_list[stack_number].append(iterator)
        if -1 in state_list[stack_number]:
            state_list[stack_number].remove(-1)

    generated_state = []
    for stack in state_list:
        if -1 not in stack:  # Tuples that remain "empty" (contain -1) are not added to the generated_state list
            generated_state.append(tuple(stack))

    state_tuple = tuple(generated_state)  # Transform the list into a tuple
    return state_tuple
