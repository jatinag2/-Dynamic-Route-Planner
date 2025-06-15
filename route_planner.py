import heapq
import math
import matplotlib.pyplot as plt
import networkx as nx

# Node class to represent each cell in the grid
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.position = (x, y)

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __lt__(self, other):
        return False

# Heuristic function for A* (Euclidean distance)
def heuristic(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

# Generate valid neighbors of a node
def get_neighbors(node, grid_size, obstacles):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    neighbors = []
    for dx, dy in directions:
        nx, ny = node.x + dx, node.y + dy
        if 0 <= nx < grid_size[0] and 0 <= ny < grid_size[1] and (nx, ny) not in obstacles:
            neighbors.append(Node(nx, ny))
    return neighbors

# Dijkstra's algorithm
def dijkstra(start, goal, grid_size, obstacles):
    dist = {start: 0}
    prev = {}
    visited = set()
    pq = [(0, start)]

    while pq:
        current_dist, current = heapq.heappop(pq)
        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            break

        for neighbor in get_neighbors(current, grid_size, obstacles):
            alt = dist[current] + 1
            if neighbor not in dist or alt < dist[neighbor]:
                dist[neighbor] = alt
                prev[neighbor] = current
                heapq.heappush(pq, (alt, neighbor))

    return reconstruct_path(prev, start, goal)

# A* algorithm
def astar(start, goal, grid_size, obstacles):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            return reconstruct_path(came_from, start, goal)

        for neighbor in get_neighbors(current, grid_size, obstacles):
            tentative_g = g_score[current] + 1
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return []

# Reconstruct the path from start to goal
def reconstruct_path(came_from, start, goal):
    current = goal
    path = []
    while current != start:
        path.append(current.position)
        current = came_from.get(current)
        if current is None:
            return []  # No path
    path.append(start.position)
    path.reverse()
    return path

# Visualize the grid and path
def plot_grid(grid_size, path, obstacles):
    G = nx.grid_2d_graph(grid_size[0], grid_size[1])
    pos = dict((n, n) for n in G.nodes())
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos=pos, node_color='lightgrey', with_labels=False, node_size=200)

    if path:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='green')
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='green', width=2)

    nx.draw_networkx_nodes(G, pos, nodelist=list(obstacles), node_color='red')

    plt.title("Dynamic Route Planner")
    plt.show()

# Main function to test both algorithms
if __name__ == '__main__':
    grid_size = (10, 10)
    start = Node(0, 0)
    goal = Node(9, 9)
    obstacles = {(3, 3), (3, 4), (4, 3), (5, 5), (6, 6), (7, 6)}

    print("Dijkstra Path:")
    path1 = dijkstra(start, goal, grid_size, obstacles)
    print(path1)
    plot_grid(grid_size, path1, obstacles)

    print("A* Path:")
    path2 = astar(start, goal, grid_size, obstacles)
    print(path2)
    plot_grid(grid_size, path2, obstacles)
