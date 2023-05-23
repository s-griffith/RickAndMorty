import numpy as np

from FrozenLakeEnv import FrozenLakeEnv
from typing import List, Tuple
import heapdict

class Node():
    def __init__(self, id, cost, h_cost = 0):
        self.id = id
        self.cost = cost
        self.h_cost = h_cost
        self.predecessor = None
        self.action = None

    def set_predecessor(self, predecessor, action):
        self.predecessor = predecessor
        self.action = action


class Agent():
    def __init__(self):
        self.env = None
        self.cost = 0
        self.expanded = 0
        self.actions = []

    def solution(self, node) -> list():
        self.cost = 1
        while node.predecessor is not None:
            self.actions.append(node.action)
            node = node.predecessor
            self.cost += node.cost
        self.actions.reverse()
        return self.actions
    
    def add_expansion(self, num = 1):
        self.expanded += num
    
    def get_cost(self, node) -> int:
        cost = 0
        if node is not None:
            temp = node
            while node.predecessor is not None:
                node = node.predecessor
                cost += node.cost
            node = temp
        return cost
    def h_cost(self, node) -> int:
        goals = self.env.get_goal_states()
        dist = 100
        for goal in goals:
            row = abs(self.env.to_row_col(goal)[0] - self.env.to_row_col(node)[0])
            column = abs(self.env.to_row_col(goal)[1] - self.env.to_row_col(node)[1])
            if (row + column) < dist:
                dist = row + column
        return dist


class BFSAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        return None
    
    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], float, int]:
        self.env = env
        self.env.reset()
        state = Node(self.env.get_initial_state(), 0)
        self.env.set_state(state.id)
        open = []
        close = []
        openNodes = []
        open.append(state.id)
        openNodes.append(state)
        while open:
            state = openNodes.pop(0)
            open.pop(0)
            self.env.set_state(state.id)
            close.append(state.id)
            if state.id != None:
                self.add_expansion()
                for action, successor in self.env.succ(self.env.get_state()).items():
                    if action == None:
                        break
                    child = Node(successor[0], successor[1])
                    if (child.id not in open) and (child.id not in close):
                        child.set_predecessor(state, action)
                        if self.env.is_final_state(child.id):
                            return [self.solution(child), self.cost, self.expanded]
                        open.append(child.id)
                        openNodes.append(child)
        return [None, -1, -1]


class DFSAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        return None
    
    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], float, int]:
        self.env = env
        self.env.reset()
        state = Node(self.env.get_initial_state(), 0)
        self.env.set_state(state.id)
        open = []
        close = []
        openNodes = []
        open.append(state.id)
        openNodes.append(state)
        return self.recursiveDFS(open, openNodes, close)

    def recursiveDFS(self, open, openNodes, close) -> Tuple[List[int], float, int]:
        state = openNodes.pop(0)
        open.pop(0)
        self.env.set_state(state.id)
        close.append(state.id)
        if self.env.is_final_state(state.id):
            return [self.solution(state), self.cost, self.expanded]
        if state.id != None:
            self.add_expansion()
            for action, successor in (self.env.succ(self.env.get_state())).items():
                if action == None:
                    break
                child = Node(successor[0], successor[1])
                if (child.id not in open) and (child.id not in close):
                    child.set_predecessor(state, action)
                    open.append(child.id)
                    openNodes.append(child)
                    result = self.recursiveDFS(open, openNodes, close)
                    if result[0] is not None:
                        return result
        return [None, -1, -1]


class UCSAgent(Agent):
  
    def __init__(self) -> None:
        super().__init__()
        return None

    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], float, int]:
        self.env = env
        self.env.reset()
        state = Node(self.env.get_initial_state(), 0)
        self.env.set_state(state.id)
        open = heapdict.heapdict()
        close = []
        openNodes = heapdict.heapdict()
        openIDs = {}
        open[state.id] = (state.h_cost, state.id)
        openNodes[state] = (state.h_cost, state.id)
        openIDs[state.id] = state
        while open:
            open.popitem()
            state = (openNodes.popitem())[0]
            openIDs.pop(state.id)
            self.env.set_state(state.id)
            close.append(state.id)
            if state.id != None:
                if self.env.is_final_state(state.id):
                    return [self.solution(state), self.cost, self.expanded]
                self.add_expansion()
                for action, successor in (self.env.succ(self.env.get_state())).items():
                    if action == None:
                        break
                    child = Node(successor[0], successor[1], successor[1] + state.h_cost)
                    if child.id not in close and child.id not in open.keys():
                        child.set_predecessor(state, action)
                        open[child.id] = (child.h_cost, child.id)
                        openNodes[child] = (child.h_cost, child.id)
                        openIDs[child.id] = child
                    elif child.id in open.keys() and open[child.id][0] > child.h_cost:
                        child.set_predecessor(state, action)
                        open[child.id] = (child.h_cost, child.id)
                        openNodes.pop(openIDs[child.id])
                        openNodes[child] = (child.h_cost, child.id)
                        openIDs[child.id] = child
        return [None, -1, -1]


class GreedyAgent(Agent):
  
    def __init__(self) -> None:
        super().__init__()
        return None

    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], float, int]:
        self.env = env
        self.env.reset()
        state = Node(self.env.get_initial_state(), 0, self.h_cost(self.env.get_initial_state()))
        self.env.set_state(state.id)
        open = heapdict.heapdict()
        close = []
        openNodes = heapdict.heapdict()
        open[state.id] = (state.h_cost, state.id)
        openNodes[state] = (state.h_cost, state.id)
        while open:
            open.popitem()
            state = (openNodes.popitem())[0]
            self.env.set_state(state.id)
            close.append(state.id)
            if self.env.is_final_state(state.id):
                return [self.solution(state), self.cost, self.expanded]
            if state.id != None:
                self.add_expansion()
                for action, successor in (self.env.succ(self.env.get_state())).items():
                    if action == None or successor[0] == None:
                        break
                    child = Node(successor[0], successor[1], self.h_cost(successor[0]))
                    if child.id not in close and child.id not in open.keys():
                        child.set_predecessor(state, action)
                        open[child.id] = (child.h_cost, child.id)
                        openNodes[child] = (child.h_cost, child.id)
        return [None, -1, -1]

class WeightedAStarAgent(Agent):
    
    def __init__(self):
        super().__init__()
        self.h_weight = None
        return None

    def h_cost(self, node, previous = None, cost = 0) -> int:
        h = super().h_cost(node)
        g = self.get_cost(previous) + cost
        return ((1-self.h_weight)*g + self.h_weight*h)

    def search(self, env: FrozenLakeEnv, h_weight) -> Tuple[List[int], float, int]:
        self.env = env
        self.h_weight = h_weight
        self.env.reset()
        state = Node(self.env.get_initial_state(), 0, self.h_cost(self.env.get_initial_state()))
        open = heapdict.heapdict()
        close = {}
        openIDs = {}
        open[state.id] = (state.h_cost, state.id)
        openIDs[state.id] = state
        while open:
            key = open.popitem()[0]
            state = openIDs.pop(key)
            self.env.set_state(state.id)
            close[state.id] = state.h_cost
            if self.env.is_final_state(state.id):
                return [self.solution(state), self.cost, self.expanded]
            if state.id != None:
                self.add_expansion()
                for action, successor in (self.env.succ(self.env.get_state())).items():
                    if action == None:
                        break
                    child = Node(successor[0], successor[1], self.h_cost(successor[0], state, successor[1]))
                    if child.id not in close.keys() and child.id not in openIDs.keys():
                        child.set_predecessor(state, action)
                        open[child.id] = (child.h_cost, child.id)
                        openIDs[child.id] = child
                    elif child.id in openIDs.keys():
                        if open[child.id][0] > child.h_cost:
                            child.set_predecessor(state, action)
                            open[child.id] = (child.h_cost, child.id)
                            openIDs[child.id] = child
                    elif child.h_cost < close[child.id]:
                        child.set_predecessor(state, action)
                        open[child.id] = (child.h_cost, child.id)
                        openIDs[child.id] = child
                        close.pop(child.id)
                        self.add_expansion(-1) #to avoid double-counting expanding the same node
        return [None, -1, -1]


class IDAStarAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.path = {}
        self.limit = 0
        return None
        
    def DFSf(self, state, cost, f_limit) -> Tuple[List[int], float, int]:
        new_f = cost + state.h_cost
        if new_f > f_limit:
            self.limit = min(self.limit, new_f)
            return [None, -1, -1]
        if self.env.is_final_state(state.id):
            return [self.path.values(), cost, -1]
        for action, successor in (self.env.succ(state.id)).items():
            if action == None:
                break
            child = Node(successor[0], successor[1], self.h_cost(successor[0]))
            if child.id not in self.path.keys():
                self.path[child.id] = action
                result = self.DFSf(child, cost + child.cost, f_limit)
                if result != [None, -1, -1]:
                    return result
                self.path.pop(child.id)
        return [None, -1, -1]


    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], float, int]:
        self.env = env
        self.env.reset()
        state = Node(self.env.get_initial_state(), 0, self.h_cost(self.env.get_initial_state()))
        cost = 0
        self.limit = state.h_cost
        flag = True
        while flag:
            f_limit = self.limit
            self.limit = 1000
            result = self.DFSf(state, cost, f_limit)
            if result != [None, -1, -1]:
                return result
        return [None, -1, -1]