# Project Title: 4x4 Puzzle Solver

This is a Python program that uses image processing and AI-based search algorithms to solve a 4x4 sliding tile puzzle. 

The code is designed to capture the state of a sliding puzzle from a screenshot, identify the tiles, and then use an IDA* (Iterative Deepening A*) algorithm to find the solution. 

The code also includes functions to simulate the moves, thus solving the puzzle in real-time. 

## Prerequisites

This program requires the following libraries:
- OpenCV (cv2)
- PyAutoGUI
- numpy
- time
- PyDirectInput
- PriorityQueue (from the queue library)
- ctypes
- PIL (Python Imaging Library)
- matplotlib

To install the prerequisites, use the following command:

```bash
pip install opencv-python pyautogui numpy pydirectinput queue ctypes Pillow matplotlib
```

## How to Run the Code

1. Run the script with Python.

2. Make sure the 4x4 sliding puzzle is visible on the screen, in the specified region (X: 700 to 1220, Y: 180 to 700).

3. The script will capture a screenshot of the specified region, then identify and label the tiles.

4. The IDA* algorithm will calculate the moves required to solve the puzzle.

5. The script will then simulate these moves, solving the puzzle in real-time.

6. You can observe the final state of the puzzle and the steps taken to solve it.

## Functions

- `combine_tiles(matrix, goal_tiles)`: Combines individual tiles to form a complete image.
- `capture_screenshot(region)`: Captures a screenshot of the specified region.
- `split_screenshot(screenshot, rows, cols)`: Splits the screenshot into individual tiles.
- `label_tiles(tiles, goal_tiles)`: Labels the tiles based on color similarity and comparison with the goal state.
- `store_labels_in_matrix(labels, rows, cols)`: Stores the labels in a numpy matrix.
- `test_accuracy(actual, generated)`: Tests the accuracy of the generated solution against the actual solution.

## Note

- Make sure the game window is in the same position every time you run this script, because the code uses fixed coordinates to interact with the game.

- You can change the region of the screenshot and the image paths in the code as per your needs.

- Make sure all goal tile images are present in the same directory as the script.

- This code is designed for 4x4 sliding puzzles. For puzzles of other sizes, you'll need to adjust the code accordingly.
