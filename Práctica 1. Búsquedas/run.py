# Search methods

import search

ab = search.GPSProblem('O', 'E', search.romania)
#af = search.GPSProblem('A', 'F', search.romania)
#ae = search.GPSProblem('A', 'E', search.romania)

# Alredy implemented
'''
print "Breadth First: ", search.breadth_first_graph_search(ab).path()
print "Depth First: ", search.depth_first_graph_search(ab).path()
print "Depth limited:", search.iterative_deepening_search(ab).path()
print search.depth_limited_search(ab).path()

print search.astar_search(ab).path()
'''

# Student methods
print "Branch and Bound: ", search.branch_and_bound(ab).path()
print "\nB&B Subestimate: ", search.branch_and_bound_subestimate(ab).path()

# Result:
# [<Node B>, <Node P>, <Node R>, <Node S>, <Node A>] : 101 + 97 + 80 + 140 = 418
# [<Node B>, <Node F>, <Node S>, <Node A>] : 211 + 99 + 140 = 450