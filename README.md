## :memo: Sudoku-Solver-using-Deep-Learning-Computer-Vision-and-Backtracking
Hey there, sudoku lovers! Ever encountered a grid you could not solve and faced the trouble of solving it using computer programs by typing in all the digits one by one.
Not anymore!
We have a program, a solution that will spare you from all that trouble.
**This program uses, various methods of Computer Vision, Neural Networks, and Backtracking.**

More details given below.
:rocket: 

---
### Packages and Tools used: 
**Packages:** openCV, numpy, matplotlib, os, pytorch

**Tools:** VS Code

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

**This part of the Project, apart from being an intermediate, is the launcher of the entire project, as it connects all the four parts.** Other than that, the function
of the program is to read the saved image from Part 1 (one.py) namely 'puzzle_extracted.jpg' and segregates all the 81 unit cells of the sudoku grid into individual images
of a particular size using slicing and indexing and sends it to the trained Neural Network so as to identify the number in each image.
The numbers so returned get stored
in a numpy array, to be further sent to the solver program (four.py). All this is done after running the first program (one.py) by importing it.

#### 3. Use of Artifcial Neural Network to identify digits (three.py)

All the other files except one.py, two.py and four.py, all belong to this part. So starting with three.py, the file contains all code for the neural network, including training and testing functions. To train the neural network, uncomment the '#main()' on the last line of the code and run it, the neural network will start training on the dataset given to it in 'Database/Dataset' directory. After testing please comment out the 'main()' function call at the end of the code again. To identify the number in an image, it is to be sent to the function 'real_deal(im)' with the a square image of dimensions 50px only, it will return the number in 'int' datatype. We have several other functions that help builf and optimize the neural network. The file X_Y_writer.py is used in three.py to get the training dataset in vectorized and standardized form, test_dev.py and test_train.py can be run to test the accuracy of the last trained neural network model on the dev set and train set respectively. The other files in the directory with '.npy' extension store the training and the dev set data as of the last trained neural network. The database for our project is taken by dividing unit cells of pre scanned sudoku grids as stored in 'Database' directory, using the divider.py function. Finally the parameters for the last trained neural network is stored in 'Parameters' directory, to files divided by layers into '.npz' extension files. 

#### 4. Solving the Sudoku Using Backtracking algorithm (four.py)

Finally, the scanned grid is structured and initialized as numpy array, with 0 in place of blank spaces, and sent to four.py. This part uses the recursive backtracking algorithm to solve the sudoku grid and prints the solved grid!

:arrow_right: Voila, your sudoku got solved and the tremendous frustration of an unsolved sudoku got avoided!

---

### Footnotes

- To run the program, run two.py file.
- Do not use handwritten sudoku.
- To train the neural network again, uncomment '#main()' in the last line of three.py and run. Comment the 'main()' on the last line again.
