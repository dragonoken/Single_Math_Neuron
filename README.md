# Description
This is a program with single neuron that takes a number as an input and outputs another number.
The neuron will first train on some data and adjust its weight to find the right operation.
Then, it will take an integer input from the user and output a number using the same operation.

The operation in this program is (ideally) 2x - 1


At first, I tried using a simple gradient descent, but I figured out that it was prone to exploding gradient when there is a huge value as an input in the training dataset. So, I added an extra step to normalize the gradients to unit scale (-1 <= grad <= 1) when the values are too big (or too small if they are negative).

Also, to make the learning process faster, I made the learning rate decay when there is not much progress being made.

# Dependency
just Python 3 :) (My Python version was 3.6.5)

# Usage
1. Run the python script (using cmd or any other program that can run it)
2. Wait until the training is done (it shouldn't take long... only few seconds, probably)
3. When prompted, enter any integer you want
4. Look at this good boy comming up with the answer
