from search import *


class BlocksWorld(Problem) :
    def __init__(self, initial, goal):
        Problem.__init__(self, initial, goal)
        self.initial = initial
        self.goal = goal

    def result(self, state, action):
        # Method used to compute the resulting state based on a certain action and the current state and return it
        # state - the current state
        # action - action to be taken
        state_list = list(state)
        source_stack = action[0]
        destination_stack = action[1]
        moved_block = 0
        for block in state_list[source_stack]:  # moved_block is the last block in stack
            moved_block = block
        if len(state[source_stack]) != 1:  # If moved_block is not the only block in its stack
            new_stack = []
            for iterator in range(len(state[source_stack])-1):
                new_stack.append(state[source_stack][iterator])  # Copy the other blocks to another stack and
            state_list.append(tuple(new_stack))  # Add them to the final list

        if destination_stack != ' ':   # If the block is moved from one stack to another
            state_list.remove(state[destination_stack])  # delete the old stack
            state_list.append(state[destination_stack] + (moved_block, ))  # and then add the block to it
        else:
            state_list.append((moved_block, ))  # Moving the block down means creating a new stack
            # with only the block in it

        state_list.remove(state[source_stack])  # Delete the old source stack

        state_list.sort(key=lambda stack: len(stack))
        return tuple(state_list)

    def actions(self, state):
        # A method which computes the possible actions to be taken in the current state and returns their list
        # state - the current state
        actions_list = []
        for stack in state:

            for other_stack in state:
                if other_stack != stack:
                    actions_list.append((state.index(stack), state.index(other_stack)))  # An action is represented as a
                    # pair - first element represents the source stack index and the second the destination
                    # stack index for the block movement

            if len(stack) != 1:
                actions_list.append((state.index(stack), ' '))  # A block can be moved down only if it's not
                #  the only block in its stack, then the action would be redundant

        return actions_list

    def goal_test(self, state):
        # A method used to check whether the goal state of the problem has been reached based on the current state
        # state - the current state
        for stack in state:
            if stack not in self.goal:
                return False
        return True
