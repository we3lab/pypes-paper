from pype_schema.parse_json import JSONParser
from define import color_map

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from mpl_toolkits.mplot3d import Axes3D
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--mode", "-m", type=int, default=1, help="Which WWTP to use (1, 2, 3)")
parser.add_argument("--update", "-u", action="store_true", help="update mode")
parser.add_argument("--log_file", "-l", type=str, help="Path to log file")
parser.add_argument("--original_log_file", "-o", type=str, help="Path to original log file")
args = parser.parse_args()

plt.rcParams['axes.labelsize'] = 14  # Font size for axis labels
plt.rcParams['xtick.labelsize'] = 14  # Font size for x-axis tick labels
plt.rcParams['ytick.labelsize'] = 14  # Font size for y-axis tick labels
# plt.rcParams['figure.dpi'] = 300 # Sets default DPI to 300
plt.rcParams['savefig.dpi'] = 300
# plt.rc('text', usetex=True)

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
        """Check if set B can be fathomed based on bounds comparison with set A."""
        if (upper_bounds_A[0] < lower_bounds_B[0] and
            upper_bounds_A[1] < lower_bounds_B[1] and
            upper_bounds_A[2] < lower_bounds_B[2]):
            return True 
        return False

    def branch(self, constraints=None):
        '''Branch and bound algorithm to find the Pareto-optimal solutions
            constraints: a list of index that are fixed to 1
        '''
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
            if constraints and (current_index+1) in constraints:
                current_layout[current_index] = 1
                recursive_branching(current_layout, current_index + 1)
            else:
                current_layout[current_index] = 0
                recursive_branching(current_layout, current_index + 1)

                current_layout[current_index] = 1
                recursive_branching(current_layout, current_index + 1)


        recursive_branching([0] * self.n, 0)

        self.pareto_front = self.get_pareto_optimal_solutions(layouts)
        return self.pareto_front
   
    def get_pareto_optimal_solutions(self, layouts):
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
        # sort solutions by 1. cost, 2. observability, 3. redundancy 4. layout
        pareto_front.sort(key=lambda x: (x[1], x[2], x[3], x[0]))
        print("[INFO] Number of Pareto-optimal solutions:", len(pareto_front))
        return pareto_front

    def visualize_graph(self):
        nx.draw(self.G, with_labels=True, node_color='lightblue', font_weight='bold')
        plt.show()

    def save_pareto_solutions(self, path=None):
        if path:
            with open(path, 'w') as f:
                f.write("Cost\t\tObservability\tRedundancy\tLayout\n")
                # save in integer format
                for layout, cost, obs, red in self.pareto_front:
                    f.write(f"({cost:.0f}, {self.n-obs:.0f}, {self.n-red:.0f})\t{layout}\n")
        print(f"Number of Pareto-optimal solutions: {len(self.pareto_front)}")

    def compare_pareto_solutions(self, other_pareto_front_layouts):
        count = defaultdict(int)
        layouts = list(''.join(map(str, layout[0])) for layout in self.pareto_front)
        new_layouts = []
        for i, layout in enumerate(layouts):
            if layout not in other_pareto_front_layouts:
                count['new'] += 1
                new_layouts.append(i+1)
            else:
                count['both'] += 1
            
        for i, layout in enumerate(other_pareto_front_layouts):
            if layout not in layouts:
                count['original'] += 1
            
        print(count)
        print("New layouts:", new_layouts)

    def compare_pareto_solutions_wraper(self, other_path):
        with open(other_path, 'r') as f:
            # (4, 6, 3)	[0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0]
            other_pareto_front_layouts = list()
            for line in f.readlines():
                if line.startswith("Cost"):
                    continue
                data = line.split("\t")
                cost, obs, red = data[0][1:-1].split(',')
                cost, obs, red = int(cost), int(obs), int(red)
                layout = list(map(int, data[1].strip()[1:-1].split(',')))
                other_pareto_front_layouts.append(''.join(map(str, layout)))
        self.compare_pareto_solutions(other_pareto_front_layouts)

    def test_objectives(self, x):
        print("x:", np.array(x), "Cost:\t\t", self.cost_function(x))
        y = self.observability_function(x, verbose=list)
        obs = np.dot(self.observability_weights, (1 - y))
        print("y:", y, "Observability:\t", self.n-obs)
        z = self.redundancy_function(x, verbose=False)
        red = np.dot(self.redundancy_weights, (1 - z))
        print("z:", z, "Redundancy:\t\t", self.n-red)

    def count_types(self):
        tcounts = defaultdict(int)
        for layout, cost, obs, red in self.pareto_front:
            tcounts[(self.n-obs, cost, self.n-red)] += 1
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

    def visualize_pareto_solutions(self, save_path=None):
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
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(costs, observabilities, redundancies, s=sizes, c='b', alpha=0.6)
        # add dash lines from the points to the axes
        for i in range(len(costs)):
            ax.plot([costs[i], costs[i]], [observabilities[i], observabilities[i]], [0, redundancies[i]], c='gray', linestyle='--', linewidth=0.8)
            ax.plot([costs[i], costs[i]], [0, observabilities[i]], [redundancies[i], redundancies[i]], c='gray', linestyle='--', linewidth=0.8)
            ax.plot([0, costs[i]], [observabilities[i], observabilities[i]], [redundancies[i], redundancies[i]], c='gray', linestyle='--', linewidth=0.8)

        ax.set_xlabel('Cost')
        ax.set_ylabel('Observability')
        ax.set_zlabel('Redundancy')
        # ax.set_title('Pareto-Optimal Solutions (WWTP2-update)', fontsize=18)
        plt.show()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")


def plot_pareto_3d(file_path, plant_no=1, margin=0.73):
    """
    Plot a 3D Pareto front from a text file containing Cost, Observability, Redundancy, and Layout.
    
    Parameters:
    - file_path (str): Path to the input text file.
    - plant_no (int): Plant number for labeling (default=1).
    - margin (float): Margin for plot limits (default=0.73 for PlantNo=1, adjust as needed).
    """

    blue = "#1f78b4"
    green = "#33a02c"
    red_color = "#EA3323"
    orange = "#FFC400"


    # Load data from text file
    with open(file_path, 'r') as f:
        lines = f.readlines()[1:]  # Skip header

    # Parse data
    ParetoFront = []
    Layouts = []
    for line in lines:
        parts = line.strip().split('\t')
        cost, obs, red = eval(parts[0])  # Parse tuple like (0, 0, 0)
        layout = eval(parts[1])          # Parse list like [0, 0, 0, ...]
        ParetoFront.append([cost, obs, red])
        Layouts.append(layout)

    ParetoFront = np.array(ParetoFront)
    Layouts = np.array(Layouts)

    # Compute maximum values
    nC = np.max(ParetoFront[:, 0])  # Max cost (sensors)
    nO = np.max(ParetoFront[:, 1])  # Max observability
    nR = np.max(ParetoFront[:, 2])  # Max redundancy
    MaxScale = np.max([nC, nO, nR])
    MaxXY = np.max([nC, nO])

    # Get unique Pareto points and their counts
    unique_points, indices = np.unique(ParetoFront, axis=0, return_inverse=True)
    counts = np.bincount(indices)
    Matrix = np.hstack((counts.reshape(-1, 1), unique_points))

    # Debug: Print data to verify
    print(f"ParetoFront:\n{ParetoFront}")
    print(f"Unique Points:\n{unique_points}")
    print(f"Counts:\n{counts}")
    print(f"Matrix:\n{Matrix}")

    # Calculate marker sizes based on counts
    Sizes = counts ** 0.5
    Sizes = (0.27 + 0.73 * (Sizes - np.min(Sizes)) / (np.max(Sizes) - np.min(Sizes))) * 23
    print(f"Sizes:\n{Sizes}")

    # Create 3D plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-margin, MaxScale + margin])
    ax.set_ylim([-margin, MaxScale + margin])
    ax.set_zlim([0, MaxScale + margin])
    ax.set_xticks(np.arange(0, MaxScale + 1))
    ax.set_yticks(np.arange(0, MaxScale + 1))
    ax.set_zticks(np.arange(0, MaxScale + 1), minor=True)
    ax.grid(True)
    # ax.zaxis._axinfo['juggled'] = (2,2,2)

    for i in range(len(unique_points)):
        ax.plot3D([MaxScale + margin, unique_points[i, 0]], 
                  [unique_points[i, 1], unique_points[i, 1]], 
                  [unique_points[i, 2], unique_points[i, 2]], '-', 
                  color=[0.42]*3, linewidth=0.5)
        ax.plot3D([unique_points[i, 0], unique_points[i, 0]], 
                  [-margin, unique_points[i, 1]], 
                  [unique_points[i, 2], unique_points[i, 2]], '-', 
                  color=[0.42]*3, linewidth=0.5)
        ax.plot3D([unique_points[i, 0], unique_points[i, 0]], 
                  [unique_points[i, 1], unique_points[i, 1]], 
                  [-margin, unique_points[i, 2]], '-', 
                  color=[0.42]*3, linewidth=0.5)

    ax.plot3D([0, 0], [-margin, nO], [nR, nR], '--', 
              color=[0.42]*3, linewidth=0.5)  # y-plane projection at x=nC
    ax.plot3D([0, 0], [nO, nO], [-margin, nR+margin], '--', 
              color=[0.42]*3, linewidth=0.5)  # z-plane projection at x=nC
    ax.plot3D([-margin, nC], [nO, nO], [nR, nR], '--', 
              color=[0.42]*3, linewidth=0.5)  # x-plane projection ending at x=nC
    ax.plot3D([0], [nO], [nR], 'kx')
    
    # Define conditions and colors as per MATLAB code
    conditions = [
        (np.logical_and(np.logical_and(unique_points[:, 1] != nO, unique_points[:, 0] != 0), 
                        np.logical_and(unique_points[:, 2] != nR, unique_points[:, 2] != 0)), green),  # Green
        (np.logical_and(unique_points[:, 1] == nO, unique_points[:, 0] + np.sum(Layouts[0]) == unique_points[:, 2])[1:], red_color),  # Red
        (np.logical_and(unique_points[:, 1] == nO, unique_points[:, 0] + np.sum(Layouts[0]) == unique_points[:, 2])[0:1], orange),  # Orange
        (np.logical_and(unique_points[:, 1] == nO, unique_points[:, 2] == 0), blue),  # Blue
        (unique_points[:, 0] == 0, 'k'),  # Black
        (np.logical_and(np.logical_and(unique_points[:, 0] != 0, unique_points[:, 2] == 0), 
                        np.logical_and(unique_points[:, 1] != 0, unique_points[:, 1] != nO)), 'w')  # White
    ]

    # Plot points with specified colors and sizes
    for condition, color in conditions:
        select = np.where(condition)[0]
        for j in select:
            ax.plot3D([unique_points[j, 0]], [unique_points[j, 1]], [unique_points[j, 2]], 'ko', 
                      markerfacecolor=color, markersize=Sizes[j])
    ax.view_init(azim=-120, elev=30)
    ax.set_xlabel(f'# sensors (fC)')
    # ax.set_xlabel(r'$# sensors (f\underline{C})$')
    ax.set_ylabel(f'# observable ({int(nO)}-fO)')
    ax.set_zlabel(f'# redundant ({int(nR)}-fR)')
    # ax.text(MaxXY, MaxScale + margin, MaxScale + margin, f'# redundant ({int(nR)}-fR)', 
            # fontsize=10, ha='right', va='bottom')

    plt.tight_layout()
    plt.show()

def run():
    WWTP_ID = args.mode

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
                ('e', 'f'), ('g', 'c'), ('f', 'g'), 
                ('f', 'a'), ('d', 'b')]

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

    if args.update:
        layout_43 = [3, 4, 6, 7]
        pareto_front = optimizer.branch(constraints=layout_43)
        optimizer.compare_pareto_solutions_wraper(args.original_log_file)
    else:
        pareto_front = optimizer.branch()

    optimizer.visualize_pareto_solutions(save_path="results/opt_sensor/pareto_front_WWTP2.png")
    optimizer.save_pareto_solutions(path=args.log_file)
    optimizer.count_types()


if __name__ == '__main__':
    run()
    # plot_pareto_3d(args.log_file, plant_no=2, margin=0.73)