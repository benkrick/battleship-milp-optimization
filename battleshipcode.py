import gurobipy as gp
from gurobipy import GRB
import numpy as np

def solve_battleship():
    # Parameters
    board_size = 10
    ship_info = {
        'Carrier': {'size': 5, 'count': 1},
        'Battleship': {'size': 4, 'count': 1},
        'Cruiser': {'size': 3, 'count': 1},
        'Submarine': {'size': 3, 'count': 1},
        'Destroyer': {'size': 2, 'count': 1}
    }

    # Opponent's guessing strategy (checkerboard)
    guess_order = []
    for parity in [0, 1]:
        for i in range(board_size):
            for j in range(board_size):
                if (i + j) % 2 == parity:
                    guess_order.append((i, j))

    # Assign guess numbers based on the opponent's strategy
    guess_numbers = {}
    for idx, (i, j) in enumerate(guess_order):
        guess_numbers[(i, j)] = idx + 1  # Guesses start from 1

    # Model
    model = gp.Model("Battleship_Maximize_Guesses")

    # Decision variables
    # Ship placement variables
    s = model.addVars(board_size, board_size, vtype=GRB.BINARY)

    # Guessing variables
    g = model.addVars(board_size, board_size, vtype=GRB.BINARY)
    h = model.addVars(board_size, board_size, vtype=GRB.BINARY)
    m = model.addVars(board_size, board_size, vtype=GRB.BINARY)

    # Auxiliary variable for total guesses
    G = model.addVar(vtype=GRB.INTEGER, name="Total_Guesses")

    # Ship placement variables for positions
    ship_positions = {}
    position_vars = {}
    ship_count = 0
    for ship_name, info in ship_info.items():
        size = info['size']
        count = info['count']
        ship_positions[ship_name] = []
        position_vars[ship_name] = []

        # Generate all possible positions (horizontal and vertical)
        positions = []
        for orientation in ['H', 'V']:
            if orientation == 'H':
                for i in range(board_size):
                    for j in range(board_size - size + 1):
                        cells = [(i, j + k) for k in range(size)]
                        positions.append({'cells': cells, 'orientation': 'H'})
            else:
                for i in range(board_size - size + 1):
                    for j in range(board_size):
                        cells = [(i + k, j) for k in range(size)]
                        positions.append({'cells': cells, 'orientation': 'V'})

        # Create position variables
        vars = model.addVars(len(positions), vtype=GRB.BINARY)
        position_vars[ship_name] = vars
        ship_positions[ship_name] = positions

        # Each ship must be placed exactly once
        model.addConstr(vars.sum() == count)

    # Link ship placement to s[i,j]
    for i in range(board_size):
        for j in range(board_size):
            # Sum over all positions of all ships that cover cell (i,j)
            occupancy_vars = []
            for ship_name in ship_info.keys():
                positions = ship_positions[ship_name]
                vars = position_vars[ship_name]
                for idx, pos in enumerate(positions):
                    if (i, j) in pos['cells']:
                        occupancy_vars.append(vars[idx])
            # Set s[i,j] = sum of occupancy_vars
            model.addConstr(s[i, j] == gp.quicksum(occupancy_vars))
            # No overlapping ships
            model.addConstr(s[i, j] <= 1)

    # Guessing strategy constraints
    for i in range(board_size):
        for j in range(board_size):
            # Guessing strategy
            if (i + j) % 2 == 0:
                model.addConstr(g[i, j] == 1)
            else:
                # For the second parity, we assume they will guess eventually
                model.addConstr(g[i, j] >= 0)

            # Hit and miss relationships
            model.addConstr(h[i, j] <= s[i, j])
            model.addConstr(h[i, j] <= g[i, j])
            model.addConstr(h[i, j] >= s[i, j] + g[i, j] - 1)

            model.addConstr(m[i, j] >= g[i, j] - s[i, j])
            model.addConstr(m[i, j] <= g[i, j])
            model.addConstr(m[i, j] <= 1 - s[i, j])

    # Total hits must equal total ship cells
    model.addConstr(h.sum() == 17, name="Total_Hits")

    # Total misses is less than or equal to 83
    model.addConstr(m.sum() <= 83, name="Total_Misses")

    # Total guesses
    model.addConstr(G == g.sum(), name="Total_Guesses_Def")

    # Objective function: Maximize total guesses
    model.setObjective(G, GRB.MAXIMIZE)

    # Optimize
    model.optimize()

    # Output results
    if model.status == GRB.OPTIMAL:
        print(f"Optimal total guesses required: {int(G.X)}\n")
        ship_grid = np.zeros((board_size, board_size), dtype=int)
        for i in range(board_size):
            for j in range(board_size):
                if s[i, j].X > 0.5:
                    ship_grid[i, j] = 1
        print("Optimal Ship Placement (1 indicates ship presence):")
        print(ship_grid)
    else:
        print("No optimal solution found.")

if __name__ == "__main__":
    solve_battleship()
