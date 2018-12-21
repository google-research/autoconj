# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility functions to recognize and extract useful information  from graphs
(e.g. tree/chain orderings, if they exist).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Queue import Queue
from collections import defaultdict


def make_factor_graph(factors):
  """Construct an adjacency list representation of a factor graph.

  Arguments:
    factors: list of tuples of nodes where each tuple represents a factor
  Returns:
    a dictionary where nodes map to their set of neighbors
  """
  graph = defaultdict(set)
  for factor in factors:
    for node in factor:
      graph[node].add(factor)
      graph[factor].add(node)
  return graph


def find_chain(graph):
  """Perform depth-first search to check if graph graph is a chain.

  Arguments:
    graph: dictionary mapping from node to set of node's neighbors
  Returns:
    a chain-ordering of the graph's nodes if one exists or False
  """
  start_node = graph.keys()[0]
  if len(graph[start_node]) > 2:
    return False

  chain_list = [start_node]
  node_stack = list(enumerate(list(graph[start_node])))
  while node_stack:
    i, curr_node = node_stack.pop()
    chain_list = i*[curr_node] + chain_list + (1-i)*[curr_node]
    visited_nghbrs = graph[curr_node].intersection(set(chain_list))
    unvisited_nghbrs = graph[curr_node].difference(set(chain_list))

    if len(visited_nghbrs) > 1 or len(unvisited_nghbrs) > 1:
      return False
    if len(unvisited_nghbrs) == 1:
      node_stack.append((i, unvisited_nghbrs.pop()))

  return len(chain_list) == len(graph.keys()) and chain_list


def find_tree(graph):
  """Perform breadth-first search to check if graph is a tree.

  Arguments:
    graph: dictionary mapping from node to set of node's neighbors
  Returns:
    a tree-ordering of the graph's nodes if one exists or False
  """
  root = graph.keys()[0]
  depths = {root: 0}
  node_queue = Queue()
  node_queue.put((root, 0))

  while not node_queue.empty():
    curr_node, curr_depth = node_queue.get()
    visited_nghbrs = graph[curr_node].intersection(set(depths.keys()))
    unvisited_nghbrs = graph[curr_node].difference(set(depths.keys()))

    # Check if there's a cycle in the graph.
    if len(visited_nghbrs) > 1:
      return False

    # Add unvisited neighbors to the queue.
    for nghbr in unvisited_nghbrs:
      depths[nghbr] = curr_depth + 1
      node_queue.put((nghbr, curr_depth+1))

  # Sort nodes by distance from the root and check if tree contains all nodes.
  visited_nodes = depths.keys()
  elimination_ordering = sorted(visited_nodes, key=lambda node: depths[node],
                                reverse=True)
  return len(elimination_ordering) == len(graph.keys()) and elimination_ordering
