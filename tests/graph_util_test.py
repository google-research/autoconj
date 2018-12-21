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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest

from autoconj import graph_util
from collections import defaultdict, OrderedDict


def check_elimination_ordering(ordering, depths):
  depth_list = [depths[node] for node in ordering]
  for i in range(len(depth_list)):
    if not all([depth_list[i] >= depth for depth in depth_list[i+1:]]):
      return False
  return True


def add_edges(pgm, edge_list):
  for v1, v2 in edge_list:
    if v1 not in pgm:
      pgm[v1] = set()
    if v2 not in pgm:
      pgm[v2] = set()
    pgm[v1].add(v2)
    pgm[v2].add(v1)

T = 10


class GraphUtilTest(absltest.TestCase):

  def testFindChainFromEnd(self):
    # Starting at the end of a chain.
    pgm = OrderedDict([(0, set([1]))] +
                      [(t, set([t-1, t+1])) for t in range(1, T-1)] +
                      [(T-1, set([T-2]))])
    chain_list = graph_util.find_chain(pgm)
    self.assertEqual(chain_list, range(T))

  def testFindChainFromMiddle(self):
    # Starting in the middle of a chain.
    mid = T // 2
    pgm = OrderedDict([(t, set([t-1, t+1])) for t in range(mid, T-1)] +
                      [(0, set([1])), (T-1, set([T-2]))] +
                      [(t, set([t-1, t+1])) for t in range(1, mid)])
    chain_list = graph_util.find_chain(pgm)
    self.assertEqual(chain_list, list(reversed(range(T))))

  def testFindChainCycle(self):
    # Cycle graph.
    pgm = OrderedDict([(0, set([1, T-1]))] +
                      [(t, set([t-1, t+1])) for t in range(1, T-1)] +
                      [(T-1, set([T-2, 0]))])
    self.assertFalse(graph_util.find_chain(pgm))

  def testFindChainDisconnectedGraph(self):
    # Disconnected graph which includes chain.
    pgm = OrderedDict([(0, set([1, T-1]))] +
                      [(t, set([t-1, t+1])) for t in range(1, T-1)] +
                      [(T-1, set([T-2, 0]))] +
                      [(T, set([]))])
    self.assertFalse(graph_util.find_chain(pgm))

  def testFindChainConnectedGraph(self):
    # Connected graph which is not a chain.
    pgm = OrderedDict([(0, set([1])), (1, set([0, 2, 3])),
                       (2, set([1])), (3, set([1]))])
    self.assertFalse(graph_util.find_chain(pgm))

  def testFindTreeChainFromEnd(self):
    # Starting at the end of a chain.
    pgm = OrderedDict([(0, set([1]))] +
                      [(t, set([t-1, t+1])) for t in range(1, T-1)] +
                      [(T-1, set([T-2]))])
    depths = dict(zip(range(T), range(T)))
    elimination_order = graph_util.find_tree(pgm)
    self.assertTrue(check_elimination_ordering(elimination_order, depths))

  def testFindTreeChainFromMiddle(self):
    mid = T // 2
    pgm = OrderedDict([(t, set([t-1, t+1])) for t in range(mid, T-1)] +
                      [(0, set([1])), (T-1, set([T-2]))] +
                      [(t, set([t-1, t+1])) for t in range(1, mid)])
    depths = {t: abs(mid-t) for t in range(T)}
    elimination_order = graph_util.find_tree(pgm)
    self.assertTrue(check_elimination_ordering(elimination_order, depths))

  def testFindTreeWheel(self):
    T = 3
    pgm = OrderedDict()
    add_edges(pgm, [(0, t) for t in range(1, T)])
    depths = dict([(0, 0)] + [(t, 1) for t in range(1, T)])
    elimination_order = graph_util.find_tree(pgm)
    self.assertTrue(check_elimination_ordering(elimination_order, depths))

  def testFindTreeGeneric(self):
    pgm = OrderedDict()
    # Children:
    # (root) 0: 1, 2
    # 1: 3, 4, 5
    # 2: 7
    # 3: 6
    # 4: 9, 10
    # 5:
    # 6:
    # 7: 8
    # 8:
    # 9:
    # 10:
    add_edges(pgm, [(0, 1), (0, 2), (1, 3), (1, 4), (1, 5), (3, 6),
                    (4, 9), (4, 10), (1, 5), (2, 7), (7, 8)])
    depths = {0:0, 1:1, 2:1, 3:2, 4:2, 5:2, 6:3, 7:2, 8:3, 9:3, 10:3}
    elimination_order = graph_util.find_tree(pgm)
    self.assertTrue(check_elimination_ordering(elimination_order, depths))

  def testFindTreeLoop(self):
    pgm = OrderedDict()
    add_edges(pgm, [(0, 1), (0, 2), (1, 3), (1, 4), (1, 5), (3, 6),
                    (4, 9), (4, 10), (1, 5), (2, 7), (7, 8)])
    add_edges(pgm, [(8, 0)])
    elimination_order = graph_util.find_tree(pgm)
    self.assertFalse(elimination_order)

if __name__ == '__main__':
  absltest.main()
