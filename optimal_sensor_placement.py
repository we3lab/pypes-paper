from pype_schema.parse_json import JSONParser
from define import color_map

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from mpl_toolkits.mplot3d import Axes3D

class SensorOptimizer:
    def __init__(self, cost_weights, observability_weights, redundancy_weights, json_path=None):
        # Create graph
        self.G = nx.MultiDiGraph()
        if json_path:
            self.load_json(json_path)
            self.nodes = list(self.G.nodes)
            self.edges = list(self.G.edges)
            self.n = len(self.edges)
        else:
            self.nodes = []
            self.edges = []
            self.n = 0

        self.cost_weights = np.array(cost_weights)
        self.observability_weights = np.array(observability_weights)
        self.redundancy_weights = np.array(redundancy_weights)
        
        self.pareto_front = []

    def __repr__(self):
        return f"SensorOptimizer with {len(self.nodes)} nodes and {len(self.edges)} edges"
    
    def load_nodes(self, nodes):
        self.nodes = nodes
        self.G.add_nodes_from(nodes)
    
    def load_edges(self, edges):
        self.edges = edges
        self.G.add_edges_from(edges)
        self.n = len(self.edges)

    def load_json(self, json_path):
        parser = JSONParser(json_path)
        network = parser.initialize_network()

        self.G.add_nodes_from(network.nodes.__iter__())
        for id, connection in network.connections.items():
            try:
                color = color_map[connection.contents.name]
            except KeyError:
                color = "red"
            self.G.add_edge(
                connection.source.id, connection.destination.id, color=color, label=id
            )

            if connection.bidirectional:
                self.G.add_edge(
                    connection.destination.id, connection.source.id, color=color, label=id
                )

    def print_graph(self):
        print("Nodes:", self.G.nodes)
        print("Edges:", self.G.edges)

    def cost_function(self, x):
        return np.dot(self.cost_weights, x)

    def observability_function(self, x, verbose=False):
        reduced_graph = self.G.copy()
        y = np.zeros_like(x)

        # Label all edges with measurements (sensors) as observable and remove them from the graph
        for i, edge in enumerate(self.edges):
            if x[i] == 1:  # If sensor is placed
                y[i] = 1  # Mark as observable
                reduced_graph.remove_edge(*edge) 

        if verbose:
            print("Reduced graph nodes:", reduced_graph.nodes)
            print("Reduced graph edges:", reduced_graph.edges)

        # In the reduced graph, label the edges not on a cycle as observable
        for i, edge in enumerate(self.edges):
            if x[i] == 0:  # Only consider edges without sensors
                # check if there is a 2 nodes cycle in the original graph
                u, v = edge
                if reduced_graph.has_edge(u, v) and reduced_graph.has_edge(v, u):
                    y[i] = 0
                    continue
                undirected_graph = reduced_graph.to_undirected()
                undirected_graph.remove_edge(*edge)
                if not nx.has_path(undirected_graph, edge[0], edge[1]):
                    y[i] = 1  
                else:
                    y[i] = 0
                undirected_graph.add_edge(*edge)
        return y

    def redundancy_function(self, x, verbose=False):
        """Calculates redundancy of the system based on the sensor placement"""
        reduced_graph = self.G.copy()
        z = np.zeros_like(x)

        for i, edge in enumerate(self.edges):
            if x[i] == 1:  # If sensor is placed
                reduced_graph.remove_edge(*edge)

        # Find the connected components of the reduced graph
        components = list(nx.connected_components(reduced_graph.to_undirected()))
        component_map = {node: comp_idx for comp_idx, component in enumerate(components) for node in component}

        if verbose:
            print("Reduced graph nodes:", reduced_graph.nodes)
            print("Reduced graph edges:", reduced_graph.edges)
            print("Connected components:", components)
            print("Component map:", component_map)

        for i, edge in enumerate(self.edges):
            if x[i] == 1:  # If sensor was placed on this edge
                if component_map[edge[0]] != component_map[edge[1]]:
                    z[i] = 1  
                else:
                    z[i] = 0 
        return z

    
    def compute_bounds(self, L):
        """Compute the upper and lower bounds for a set of sensor layouts L."""
        X_L_lower = np.ones(self.n, dtype=int) 
        X_L_upper = np.zeros(self.n, dtype=int)

        for layout in L:
            X_L_lower &= layout
            X_L_upper |= layout 

        # Upper bound
        cost_upper = self.cost_function(X_L_upper)
        y_upper = self.observability_function(X_L_upper)
        obs_upper = np.dot(self.observability_weights, (1 - y_upper))
        z_upper = self.redundancy_function(X_L_upper)
        red_upper = np.dot(self.redundancy_weights, (1 - z_upper))

        # Lower bound 
        cost_lower = self.cost_function(X_L_lower)
        y_lower = self.observability_function(X_L_lower)
        obs_lower = np.dot(self.observability_weights, (1 - y_lower))
        z_lower = self.redundancy_function(X_L_lower)
        red_lower = np.dot(self.redundancy_weights, (1 - z_lower))

        return ((cost_lower, obs_lower, red_lower), (cost_upper, obs_upper, red_upper))

    def fathom(self, upper_bounds_A, lower_bounds_B):
        """Check if set B can be fathomed based on bounds comparison with set A."""
        if (upper_bounds_A[0] < lower_bounds_B[0] and
            upper_bounds_A[1] < lower_bounds_B[1] and
            upper_bounds_A[2] < lower_bounds_B[2]):
            return True 
        return False

    def branch_only(self):
        layouts = []
        def recursive_branching(current_layout, current_index):
            if current_index == self.n:
                cost = self.cost_function(current_layout)
                y = self.observability_function(current_layout)
                observability = np.dot(self.observability_weights, (1 - y))
                z = self.redundancy_function(current_layout)
                redundancy = np.dot(self.redundancy_weights, (1 - z))
                layouts.append((current_layout.copy(), cost, observability, redundancy))
                return

            current_layout[current_index] = 0
            recursive_branching(current_layout, current_index + 1)

            current_layout[current_index] = 1
            recursive_branching(current_layout, current_index + 1)

        recursive_branching(np.zeros(self.n, dtype=int), 0)
    
        self.pareto_front = self.get_pareto_optimal_solutions_nobound(layouts)
        return self.pareto_front
   
    def get_pareto_optimal_solutions_nobound(self, layouts):
        print("[INFO] Computing Pareto-optimal solutions...")
        print("[INFO] Number of layouts:", len(layouts))

        pareto_front = []

        for i, current in enumerate(layouts):
            is_dominated = False
            for j, other in enumerate(layouts):
                if i != j:
                    # Check if 'other' dominates 'current'
                    # If all objectives are better or equal and at least one is better, 'other' dominates 'current'
                    if (np.array(other[1:]) <= np.array(current[1:])).all() and \
                    (np.array(other[1:]) < np.array(current[1:])).any():
                        is_dominated = True
                        break

            if not is_dominated:
                pareto_front.append(current)
        print("[INFO] Number of Pareto-optimal solutions:", len(pareto_front))
        return pareto_front

    def visualize_graph(self):
        nx.draw(self.G, with_labels=True, node_color='lightblue', font_weight='bold')
        plt.show()

    def display_pareto_solutions(self, print_layout=False):
        if print_layout:
            for layout, cost, obs, red in self.pareto_front:
                print(f"Layout: {layout}, Cost: {cost}, Observability: {obs}, Redundancy: {red}")
        print(f"Number of Pareto-optimal solutions: {len(self.pareto_front)}")

    def test_objectives(self, x):
        print("x:", np.array(x), "Cost:\t\t", self.cost_function(x))
        y = self.observability_function(x, verbose=True)
        obs = np.dot(self.observability_weights, (1 - y))
        print("y:", y, "Observability:\t", self.n-obs)
        z = self.redundancy_function(x, verbose=False)
        red = np.dot(self.redundancy_weights, (1 - z))
        print("z:", z, "Redundancy:\t\t", self.n-red)

    def count_types(self):
        tcounts = defaultdict(int)
        for layout, cost, obs, red in self.pareto_front:
            tcounts[(cost, self.n-obs, self.n-red)] += 1
        sorted_tcounts = dict(sorted(tcounts.items(), key=lambda item: item[0]))
        print("Number of solutions by type:")
        for t, count in sorted_tcounts.items():
            cost, obs, red = t
            print(f"[{cost:.0f}, {obs:.0f}, {red:.0f}]: {count}")
        return tcounts
    
    def count_objectives(self, objectives):
        cost, obs, red = objectives
        count = 0
        for solution in self.pareto_front:
            layout, c, o, r = solution
            if c == cost and self.n-o == obs and self.n-r == red:
                count += 1
        print(f"Number of solutions with [{cost:.0f}, {obs:.0f}, {red:.0f}]: {count}")
        return count

    def visualize_pareto_solutions(self):
        pareto_solutions = [(layout[1], layout[2], layout[3]) for layout in self.pareto_front]
        solution_counts = Counter(pareto_solutions)
        costs, observabilities, redundancies, sizes = [], [], [], []

        for (cost, obs, red), count in solution_counts.items():
            costs.append(cost)
            observabilities.append(self.n - obs)
            redundancies.append(self.n - red)
            sizes.append(count)

        max_size = max(sizes)
        sizes = [size / max_size * 1000 for size in sizes]

        # Create a 3D scatter plot
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(costs, observabilities, redundancies, s=sizes, c='b', alpha=0.6)

        ax.set_xlabel('Cost')
        ax.set_ylabel('Observability')
        ax.set_zlabel('Redundancy')
        ax.set_title('Pareto-Optimal Solutions (Cost, Observability, Redundancy)')
        plt.show()


if __name__ == '__main__':
    WWTP_ID = 2

    if WWTP_ID == 1:
        # WWTP1 graph
        n = 7
        nodes = ['a', 'b', 'c', 'd', 'e']
        edges = [('e', 'a'), ('a', 'b'), ('b', 'c'), ('c', 'e'), ('c', 'd'), ('d', 'a'), ('d', 'e')]

    elif WWTP_ID == 2:
        # WWTP2 graph
        n = 11
        nodes = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
        edges = [('g', 'a'), ('a', 'b'), ('b', 'c'), 
                ('c', 'd'), ('d', 'e'), ('e', 'g'), 
                ('e', 'f'), ('g', 'b'), ('f', 'g'), 
                ('f', 'a'), ('d', 'c')]

    elif WWTP_ID == 3:
        # WWTP3 graph
        n = 12
        nodes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        edges = [('h', 'a'), ('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'e'), ('e', 'f'), 
                 ('f', 'h'), ('f', 'g'), ('g', 'h'), ('g', 'b'), ('e', 'c'), ('d', 'a')]

    cost_weights = np.ones(n)
    observability_weights = np.ones(n)
    redundancy_weights = np.ones(n)

    optimizer = SensorOptimizer(cost_weights, observability_weights, redundancy_weights)
    optimizer.load_nodes(nodes)
    optimizer.load_edges(edges)

    # layout = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    # optimizer.test_objectives(layout)
    
    # optimizer = SensorOptimizer(cost_weights, observability_weights, redundancy_weights, json_path="json/others/WWTP3.json")

    # optimizer.visualize_graph()

    # print(optimizer)
    pareto_front = optimizer.branch_only()
    optimizer.display_pareto_solutions(print_layout=False)
    optimizer.count_types()
    # optimizer.count_objectives((6, 7, 6))
    # optimizer.visualize_pareto_solutions()
