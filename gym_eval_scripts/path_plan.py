import heapq
import numpy as np
import math

class AStarPlanner:

    def __init__(self, xy_max,xy_min,obs, resolution, rr):
        """
        Initialize grid map for a star planning

        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m]
        rr: robot radius[m]
        """

        self.resolution = resolution
        self.rr = rr
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        self.obstacle_map = None
        self.x_width, self.y_width = 0, 0
        self.motion = self.get_motion_model()
        self.calc_obstacle_map(xy_max,xy_min,obs)

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    def planning(self,st,ed):
        """
        A star path search

        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        start_node = self.Node(self.calc_xy_index(st[0], self.min_x),
                               self.calc_xy_index(st[1], self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(ed[0], self.min_x),
                              self.calc_xy_index(ed[1], self.min_y), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        while True:
            if len(open_set) == 0:
                print("Open set is empty..")
                break

            c_id = min(
                open_set,
                key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node,
                                                                     open_set[
                                                                         o]))
            current = open_set[c_id]

            if current.x == goal_node.x and current.y == goal_node.y:
                print("Find goal")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2], c_id)
                n_id = self.calc_grid_index(node)

                # If the node is not safe, do nothing
                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)
        return np.array([rx,ry]).T

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [
            self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx[::-1], ry[::-1]

    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_position(self, index, min_position):
        """
        calc grid position

        :param index:
        :param min_position:
        :return:
        """
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        # collision check
        if self.obstacle_map[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(self, xy_max,xy_min,obs):

        self.min_x,self.min_y = np.floor(xy_min)
        self.max_x,self.max_y = np.ceil(xy_max)

        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)

        # obstacle map generation
        self.obstacle_map = np.zeros((self.x_width,self.y_width))

        for t in range(obs.shape[1]):
        #     grid[obs_int[0,i,t,0]:obs_int[1,i,t,0],obs_int[0,i,t,1]:obs_int[1,i,t,1]] = 1
            self.obstacle_map[self.calc_xy_index(obs[0,t,0], self.min_x):self.calc_xy_index(obs[1,t,0], self.min_x),
                            self.calc_xy_index(obs[0,t,1], self.min_y):self.calc_xy_index(obs[1,t,1], self.min_y)] = 1
        # self.obstacle_map = [[False for _ in range(self.y_width)]
        #                      for _ in range(self.x_width)]
        # for ix in range(self.x_width):
        #     x = self.calc_grid_position(ix, self.min_x)
        #     for iy in range(self.y_width):
        #         y = self.calc_grid_position(iy, self.min_y)
        #         if x 
        #         for iox, ioy in zip(ox, oy):
        #             d = math.hypot(iox - x, ioy - y)
        #             if d <= self.rr:
        #                 self.obstacle_map[ix][iy] = True
        #                 break

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]

        return motion

# ##############################################################
class Node:
    def __init__(self, pos, g=0, h=0, parent=None):
        self.row, self.col = pos
        self.g = g
        self.h = h
        self.parent = parent

    def __lt__(self, other):
        return self.g + self.h < other.g + other.h

def is_valid(row, col, grid):
    num_rows,num_cols = grid.shape
    return 0 <= row < num_rows and 0 <= col < num_cols and not grid[row,col]

def calculate_h_value(row, col, destination):
    return abs(row - destination[0]) + abs(col - destination[1])

def get_neighbours(row, col, grid):
    neighbours = [(row+1, col), (row-1, col), (row, col+1), (row, col-1),
                  (row+1, col+1), (row-1, col+1), (row+1, col-1), (row-1, col-1)]
    valid_neighbours = []
    for neighbour in neighbours:
        n_row, n_col = neighbour
        if is_valid(n_row, n_col, grid):
            valid_neighbours.append(neighbour)
    return valid_neighbours

def reconstruct_path(current_node):
    path = []
    while current_node is not None:
        path.append((current_node.row, current_node.col))
        current_node = current_node.parent
    path.reverse()
    return np.array(path)

def d_star_algorithm(start, destination, grid):
    open_set = []
    closed_set = set()
    heapq.heapify(open_set)
    heapq.heappush(open_set, start)
    cnt = 0
    while open_set:
        
        print(cnt)
        cnt+=1
        current_node = heapq.heappop(open_set)
        closed_set.add((current_node.row, current_node.col))

        if current_node.row == destination[0] and current_node.col == destination[1]:
            return reconstruct_path(current_node)

        neighbours = get_neighbours(current_node.row, current_node.col, grid)

        for neighbour in neighbours:
            neighbour_row, neighbour_col = neighbour
            if (neighbour_row, neighbour_col) in closed_set:
                continue

            g_value = current_node.g + 1
            h_value = calculate_h_value(neighbour_row, neighbour_col, destination)
            neighbour_node = Node([neighbour_row, neighbour_col], g_value, h_value, current_node)

            if neighbour_node in open_set:
                for node in open_set:
                    if node == neighbour_node and node.g > neighbour_node.g:
                        open_set.remove(node)
                        heapq.heapify(open_set)
                        heapq.heappush(open_set, neighbour_node)
                        break
            else:
                heapq.heappush(open_set, neighbour_node)

    return None



# Parameters
k = 0.1  # look forward gain
Lfc = 0.8  # [m] look-ahead distance
Kp = 0.8  # speed proportional gain
dt = 0.2  # [s] time tick

class State:

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

    def update(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

    def calc_distance(self, point_x, point_y, i=None):
        if i == None:
            dx = self.x - point_x
            dy = self.y - point_y
        else:
            dx = self.x[i] - point_x
            dy = self.y[i] - point_y

        return np.hypot(dx, dy)


class States:

    def __init__(self):
        self.x = []
        self.y = []
        self.yaw = []
        self.v = []
        self.t = []

    def append(self, t, state):
        self.x.append(state.x)
        self.y.append(state.y)
        self.yaw.append(state.yaw)
        self.v.append(state.v)
        self.t.append(t)


def proportional_control(target, current):
    a = Kp * (target - current)

    return a


class TargetCourse:

    def __init__(self, cx, cy):
        self.cx = cx
        self.cy = cy
        self.old_nearest_point_index = None

    def search_target_index(self, state):

        # To speed up nearest point search, doing it at only first time.
        num = state.x.shape[0]
        if self.old_nearest_point_index is None:
            # search nearest point index
            dx = [state.x[i] - self.cx[i] for i in range(num)]
            dy = [state.y[i] - self.cy[i] for i in range(num)]
            d = [np.hypot(dx[i],dy[i]) for i in range(num)]
            ind = [np.argmin(d[i]) for i in range(num)]
            self.old_nearest_point_index = ind

            # dx = [state.x - icx for icx in self.cx]
            # dy = [state.y - icy for icy in self.cy]
            # d = np.hypot(dx, dy)
            # ind = np.argmin(d)
            # self.old_nearest_point_index = ind
        else:
            ind = self.old_nearest_point_index
            px = np.array([self.cx[i][ind[i]] for i in range(num)])
            py = np.array([self.cy[i][ind[i]] for i in range(num)])

            distance_this_index = state.calc_distance(px,py)

            path_ind = [i for i in range(num)]
            for i in path_ind:
                while True:
                    if (ind[i]+1)>=len(self.cx[i]):
                        break
                    px = np.array([self.cx[i][ind[i]+1]])
                    py = np.array([self.cy[i][ind[i]+1]])
                    distance_next_index = state.calc_distance(px,py,i)
                    if distance_this_index[i] < distance_next_index:
                        break
                    ind[i] = ind[i] + 1 if (ind[i] + 1) < len(self.cx[i]) else ind[i]
                    distance_this_index[i] = distance_next_index
            self.old_nearest_point_index = ind

        Lf = k * state.v + Lfc  # update look ahead distance

        # search look ahead target point index
        # px = np.array([self.cx[i][ind[i]] for i in range(num)])
        # py = np.array([self.cy[i][ind[i]] for i in range(num)])
        path_ind = [i for i in range(num)]
        for i in path_ind:
            while True:
                px = np.array([self.cx[i][ind[i]]])
                py = np.array([self.cy[i][ind[i]]])
                distance = state.calc_distance(px,py,i)
                if (ind[i] + 1) >= len(self.cx[i])\
                                or Lf[i] <= distance:
                    break
                ind[i] += 1


        # while Lf > state.calc_distance(px, py):
            # if (ind + 1) >= len(self.cx):
            #     break  # not exceed goal
            # ind += 1

        return ind, Lf


def pure_pursuit_steer_control(state, trajectory, pind):
    ind, Lf = trajectory.search_target_index(state)
    num = len(ind)
    deltas = []
    for i in range(num):
        if pind[i] >= ind[i]:
            ind[i] = pind[i]


        if ind[i] < len(trajectory.cx[i]):
            tx = trajectory.cx[i][ind[i]]
            ty = trajectory.cy[i][ind[i]]
        else:  # toward goal
            tx = trajectory.cx[i][-1]
            ty = trajectory.cy[i][-1]
            ind[i] = len(trajectory.cx[i]) - 1

        alpha = math.atan2(ty - state.y[i], tx - state.x[i]) - state.yaw[i]
        # delta = alpha
        delta = math.atan2(2.0*math.sin(alpha) / Lf[i], 1.0)
        deltas.append(delta)

    return np.array(deltas), ind


def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):
    """
    Plot arrow
    """
    from matplotlib import pyplot as plt

    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
                fc=fc, ec=ec, head_width=width, head_length=width)
    plt.plot(x, y)
