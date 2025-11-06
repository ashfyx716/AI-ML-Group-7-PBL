import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
from queue import PriorityQueue

st.set_page_config(page_title="Maze Solver", layout="centered")

st.title("🧩 Maze Solver - BFS | DFS | A*")

ROWS, COLS = 20, 20
grid = np.zeros((ROWS, COLS))

start = (0, 0)
end = (ROWS - 1, COLS - 1)
walls = st.session_state.get("walls", set())

# --- Grid Visualization ---
def draw_grid(path=[], visited=[]):
    img = np.ones((ROWS, COLS, 3))
    for (r, c) in walls:
        img[r, c] = [0, 0, 0]   # wall = black
    for (r, c) in visited:
        img[r, c] = [0.3, 0.6, 1]  # visited = blue
    for (r, c) in path:
        img[r, c] = [1, 0.5, 0]   # path = orange
    img[start] = [0, 1, 0]        # start = green
    img[end] = [1, 0, 0]          # end = red
    st.image(img, width=400, caption="Maze Grid")


def get_neighbors(pos):
    r, c = pos
    directions = [(1,0), (-1,0), (0,1), (0,-1)]
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < ROWS and 0 <= nc < COLS and (nr, nc) not in walls:
            yield (nr, nc)

def bfs(start, end):
    queue = [start]
    visited = {start: None}
    while queue:
        cur = queue.pop(0)
        if cur == end:
            break
        for n in get_neighbors(cur):
            if n not in visited:
                visited[n] = cur
                queue.append(n)
    path = []
    cur = end
    while cur in visited and cur is not None:
        path.append(cur)
        cur = visited[cur]
    return path[::-1], visited

def dfs(start, end):
    stack = [start]
    visited = {start: None}
    while stack:
        cur = stack.pop()
        if cur == end:
            break
        for n in get_neighbors(cur):
            if n not in visited:
                visited[n] = cur
                stack.append(n)
    path = []
    cur = end
    while cur in visited and cur is not None:
        path.append(cur)
        cur = visited[cur]
    return path[::-1], visited

def astar(start, end):
    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {start: None}
    g_score = {start: 0}
    while not open_set.empty():
        _, cur = open_set.get()
        if cur == end:
            break
        for n in get_neighbors(cur):
            temp_g = g_score[cur] + 1
            if n not in g_score or temp_g < g_score[n]:
                g_score[n] = temp_g
                f = temp_g + abs(n[0]-end[0]) + abs(n[1]-end[1])
                open_set.put((f, n))
                came_from[n] = cur
    path = []
    cur = end
    while cur in came_from and cur is not None:
        path.append(cur)
        cur = came_from[cur]
    return path[::-1], came_from

# --- UI Controls ---
algo = st.selectbox("Select Algorithm", ["BFS", "DFS", "A*"])
if st.button("Solve Maze"):
    if algo == "BFS":
        path, visited = bfs(start, end)
    elif algo == "DFS":
        path, visited = dfs(start, end)
    else:
        path, visited = astar(start, end)

    draw_grid(path, visited)
    st.success(f"{algo} found path of length {len(path)}")

else:
    draw_grid()
