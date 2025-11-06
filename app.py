import streamlit as st
import numpy as np
import time
from queue import PriorityQueue
from PIL import Image

# --- Page Config ---
st.set_page_config(page_title="Maze Solver", layout="centered")
st.title("🧩 Maze Solver - BFS | DFS | A* (Animated + Compare)")

# --- Grid Setup ---
ROWS, COLS = 20, 20
start = (0, 0)
end = (ROWS - 1, COLS - 1)

# Persistent state
if "walls" not in st.session_state:
    st.session_state.walls = set()

walls = st.session_state.walls


def make_image(path=[], visited=[]):
    """Generates a PIL image for current maze state."""
    img = np.ones((ROWS, COLS, 3))
    for (r, c) in walls:
        img[r, c] = [0, 0, 0]
    for (r, c) in visited:
        img[r, c] = [0.3, 0.6, 1]
    for (r, c) in path:
        img[r, c] = [1, 0.5, 0]
    img[start] = [0, 1, 0]
    img[end] = [1, 0, 0]
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img).resize((400, 400), resample=Image.NEAREST)


def draw_grid(path=[], visited=[], placeholder=None):
    """Renders the maze grid in one placeholder (to prevent stacking)."""
    img = make_image(path, visited)
    if placeholder:
        placeholder.image(img, caption="Maze Grid", use_container_width=False)
    else:
        st.image(img, caption="Maze Grid", use_container_width=False)


def get_neighbors(pos):
    r, c = pos
    for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < ROWS and 0 <= nc < COLS and (nr, nc) not in walls:
            yield (nr, nc)


def bfs(start, end, placeholder=None, animate=False):
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
                if animate:
                    draw_grid([], visited, placeholder)
                    time.sleep(0.03)
    path = []
    cur = end
    while cur in visited and cur is not None:
        path.append(cur)
        cur = visited[cur]
    if animate:
        for i in range(len(path)):
            draw_grid(path[:i], visited, placeholder)
            time.sleep(0.03)
    return path[::-1], visited


def dfs(start, end, placeholder=None, animate=False):
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
                if animate:
                    draw_grid([], visited, placeholder)
                    time.sleep(0.03)
    path = []
    cur = end
    while cur in visited and cur is not None:
        path.append(cur)
        cur = visited[cur]
    if animate:
        for i in range(len(path)):
            draw_grid(path[:i], visited, placeholder)
            time.sleep(0.03)
    return path[::-1], visited


def astar(start, end, placeholder=None, animate=False):
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
                if animate:
                    draw_grid([], came_from, placeholder)
                    time.sleep(0.03)
    path = []
    cur = end
    while cur in came_from and cur is not None:
        path.append(cur)
        cur = came_from[cur]
    if animate:
        for i in range(len(path)):
            draw_grid(path[:i], came_from, placeholder)
            time.sleep(0.03)
    return path[::-1], came_from


def compare_algorithms():
    """Compare time taken by BFS, DFS, and A*"""
    results = {}
    algos = {"BFS": bfs, "DFS": dfs, "A*": astar}
    for name, func in algos.items():
        start_time = time.time()
        path, _ = func(start, end, animate=False)
        results[name] = round(time.time() - start_time, 4)
    st.subheader("📊 Comparison Results")
    for algo, t in results.items():
        st.write(f"**{algo}** → {t}s | Path length: {len(path)}")


# --- Sidebar ---
st.sidebar.header("🧱 Maze Controls")
st.sidebar.write("Click to toggle walls (basic version).")

if st.sidebar.button("Reset Maze"):
    st.session_state.walls = set()
    walls.clear()
    st.sidebar.success("Maze reset successfully!")

# --- Main UI ---
algo = st.selectbox("Select Algorithm", ["BFS", "DFS", "A*", "Compare"])
placeholder = st.empty()  # This is the animation container

# Initial grid render
draw_grid([], [], placeholder)

col1, col2 = st.columns(2)
with col1:
    if st.button("Solve Maze"):
        if algo == "Compare":
            compare_algorithms()
        else:
            func = {"BFS": bfs, "DFS": dfs, "A*": astar}[algo]
            path, visited = func(start, end, placeholder, animate=True)
            draw_grid(path, visited, placeholder)
            st.success(f"{algo} found path of length {len(path)} ✅")
with col2:
    if st.button("Show Grid"):
        draw_grid([], [], placeholder)
