## :memo: Sudoku-Solver-using-Deep-Learning-Computer-Vision-and-Backtracking
Hey there, sudoku lovers! Ever encountered a grid you could not solve and faced the trouble of solving it using computer programs by typing in all the digits one by one.
Not anymore!
We have a program, a solution that will spare you from all that trouble.
**This program uses, various methods of Computer Vision, Neural Networks, and Backtracking.**

More details given below.
:rocket: 

### Structure of the Project:

It is divided on the basis of functions into four parts, named trivially as given underneath.

#### 1. Extracting the Sudoku Puzzle from the webcam (one.py)

The Video Feed is set live using the function cv2.VideoCapture() and each frame is captured from the feed. The frame is first converted into a grayscale image and the noise
in the frame is reduced by the use of cv2.GuassianBlur(). Then the frame is converted into a binary image with the help of Adaptive Threshold function with inverse adaptive type.
The user then displays the Sudoku puzzle in front of the webcam. As the puzzle is clear and readable, the user presses 'q' to end the video feed and the last frame is displayed
in a new window. This image allows the user to Mouse functionalities so that the user can tap on the corner points of the grid. After successfully tapping on the four points, the
window closes automatically. Next, the image is gone under Perspective Transform where the image is cropped and transformed. The final transformed image is displayed in a new 
window and the image is saved by the name 'puzzle_extacted.jpg'.

#### 2. Segregating the unit cells in sudoku (two.py)

==This part of the Project, apart from being an intermediate, is the launcher of the entire project, as it connects all the four parts.== Other than that, the function
of the program is to read the saved image from Part 1 (one.py) namely 'puzzle_extracted.jpg' and segregates all the 81 unit cells of the sudoku grid into individual images
of a particular size using slicing and indexing and sends it to the trained Neural Network so as to identify the number in each image. The numbers so returned get stored
in a numpy array, to be further sent to the solver program (four.py).
