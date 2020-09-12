## :memo: Sudoku-Solver-using-Deep-Learning-Computer-Vision-and-Backtracking
Hey there, sudoku lovers! Ever encountered a grid you could not solve and faced the trouble of solving it using computer programs by typing in all the digits one by one. Not anymore!
We have a program, a solution that will spare you from all that trouble.
**This program uses, various methods of Computer Vision, Neural Networks, and Backtracking.**

More details given below.
:rocket: 

### Structure of the program:

It is divided on the basis of functions into four parts, named trivially as given underneath.

#### 1. Extracting the Sudoku Puzzle from the webcam (one.py)

The Video Feed is set live using the function cv2.VideoCapture() and each frame is captured from the feed. The frame is first converted into a grayscale image and the noise in the frame is reduced by the use of cv2.GuassianBlur(). Then the frame is converted into a binary image with the help of Adaptive Threshold function with inverse adaptive type.


#### 2. Segregating the unit cells in sudoku (two.py)

== This part of the launcher of the entire project, as it connects all the four parts and acts an intermediate. == Other than that, the function of the program is to segregate all the 81 unit cells of the sudoku grid into individual images of a particular size using slicing and indexing and sends it to the trained Neural Network so as to identify the number in each image. The numbers so returned get stored in a numpy array, to be further sent to the solver program (four.py).

