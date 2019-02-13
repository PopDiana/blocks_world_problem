from problem import *

class BlocksWorldH1(BlocksWorld):
    def h(self, node):
        # This heuristic checks whether a block is in the right place
        sum = 0  # The number of blocks that are out of place relative to the goal state.
        for stack in node.state:
            for block in stack:
                for other_stack in self.goal:
                    if block in other_stack:
                        other_position = other_stack.index(block)
                        block_position = stack.index(block)
                        if block_position == 0 or other_position == 0 and \
                                          block_position != other_position or \
                                         stack[block_position-1] != other_stack[other_position-1]:
                                        # blocks under the current block in both states are different
                            sum = sum + 1
                            break
        return sum


class BlocksWorldH2(BlocksWorld):
    def h(self, node):
        # This heuristic counts the number of moves that need to be done
        # in order for every block to reach it correct place
        sum = 0
        for stack in node.state:
            for other_stack in self.goal:
                if stack[0] in other_stack:
                    goal_stack = other_stack
                    break
            for block in stack:
                block_position = stack.index(block)
                if block in goal_stack:
                    if block_position == goal_stack.index(block):
                        continue
                # All blocks above the current block must be moved
                # so that it'll reach its correct place
                sum = sum + len(stack) - block_position
                for iterator in range(block_position, len(stack)):
                    stack_block = stack[iterator]
                    stack_position = stack.index(stack_block)
                    # All blocks below stack_block must be moved
                    # in their correct place, but for every block
                    # stack_block must be moved on the table so we add
                    # each move to the final sum
                    if stack_position != 0:  # If there are blocks below stack_block in the current state
                        for other_stack in self.goal:
                            if stack_block in other_stack:
                                other_position = other_stack.index(stack_block)
                                if other_position != 0:  # If there are blocks below stack_block in the goal state
                                    for iterator_2 in range(0, stack_position):
                                        other_block = stack[iterator_2]
                                        if other_block in other_stack:
                                            if other_stack.index(other_block) < other_position:
                                                sum = sum + 1
                                                break
        return sum
