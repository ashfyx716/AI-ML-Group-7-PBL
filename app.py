import streamlit as st
import numpy as np
from queue import PriorityQueue
from PIL import Image

# --- Page Config ---
st.set_page_config(page_title="Maze Solver", layout="centered")
st.title("🧩 Maze Solver - BFS | DFS | A*")

# --- Grid Setup ---
ROWS, COLS = 20, 20
start = (0, 0)
end = (ROWS - 1, COLS - 1)

# Use Streamlit session state for persistence
if "walls" not in st.session_state:
    st.session_state.walls = set()

walls = st.session_state.walls

# --- Functions ---
def draw_grid(path=[], visited=[]):
    """Draws the maze grid with proper sharp pixels."""
    img = np.ones((ROWS, COLS, 3))

    # Walls
    for (r, c) in walls:
        img[r, c] = [0, 0, 0]  # black
    # Visited
    for (r, c) in visited:
        img[r, c] = [0.3, 0.6, 1]  # light blue
    # Path
    for (r, c) in path:
        img[r, c] = [1, 0.5, 0]  # orange
    # Start / End
    img[start] = [0, 1, 0]  # green
    img[end] = [1, 0, 0]    # red

    # Convert to PIL for sharp scaling
    img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img)
    img = img.resize((400, 400), resample=Image.NEAREST)

    st.image(img, caption="Maze Grid", use_container_width=False)

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

# --- Sidebar Controls ---
st.sidebar.header("🧱 Maze Controls")

# Add walls manually via coordinates
r = st.sidebar.number_input("Row (0–19):", min_value=0, max_value=19, value=0)
c = st.sidebar.number_input("Col (0–19):", min_value=0, max_value=19, value=0)

if st.sidebar.button("Add Wall"):
    if (r, c) != start and (r, c) != end:
        walls.add((r, c))
        st.session_state.walls = walls

if st.sidebar.button("Reset Maze"):
    st.session_state.walls = set()
    walls.clear()
    st.sidebar.success("Maze reset successfully!")

# --- Algorithm Selection ---
algo = st.selectbox("Select Algorithm", ["BFS", "DFS", "A*"])

# --- Run Algorithm ---
if st.button("Solve Maze"):
    if algo == "BFS":
        path, visited = bfs(start, end)
    elif algo == "DFS":
        path, visited = dfs(start, end)
    else:
        path, visited = astar(start, end)

    draw_grid(path, visited)
    st.success(f"{algo} found path of length {len(path)} ✅")

else:
    draw_grid()
