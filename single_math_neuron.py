from math import sqrt

# For training the neuron
def train(train_data_pairs, init_learning_rate, epoch, init_w, init_b, round_digits=3):
    # Initial weight, bias, and learning rate
    w = init_w
    b = init_b
    learning_rate = init_learning_rate

    # Training for <epoch> iterations
    print("Training in progress...\n")
    last_MSE = 0 # This will be used to decay the learning rate. Initialized to 0
    for i in range(epoch):
        SSE = 0 # Sum of Squared Errors
        for x, y in train_xy_pairs:
            #print("w : {}\nb : {}".format(w, b))
            output = w * x + b
            error = y - output
            SSE += error ** 2

            # Calculate how much to adjust the weight and the bias through gradient descent
            # Squared Error is used as the loss function
            # SE = 0.5 * (y - (w * x + b)) ** 2
            grad_w = -x * (y - (output))
            grad_b = -(y - (output))
            # If the scale of the gradients are too big
            if sqrt(grad_w**2 + grad_b**2) > 1:
                delta_w = learning_rate * -grad_w / sqrt(grad_w**2 + grad_b**2)
                delta_b = learning_rate * -grad_b / sqrt(grad_w**2 + grad_b**2)
            else:
                delta_w = learning_rate * -grad_w
                delta_b = learning_rate * -grad_b

            # Adjust the weight and the bias
            w += delta_w
            b += delta_b

        # Decaying learning rate if the error does not change significantly
        MSE = SSE / len(train_data_pairs) # Mean Squared Error
        if abs(1 - last_MSE / MSE) < 0.00000000001:
            learning_rate = learning_rate / 10
        last_MSE = MSE

        # Report every 1000 iterations
        if (i + 1) % 1000 == 0:
            print('Iteration {}\nMean Squared Error : {}'.format(i + 1, MSE))

    w, b = round(w, round_digits), round(b, round_digits)
    print("\nTraining done!\n\nResult weight : {}\nResult bias : {}\n".format(w, b))
    return w, b


# For taking an integer input from the user
def take_input():
    number = input("Input an integer : ").strip().replace(' ', '')
    if number[0] == '-':
        negative = True
        number = number[1:]
    else:
        negative = False
    while len(number) > 1 and number[0] == '0':
        number = number[1:]

    while not number.isdigit():
        number = input("\nInput a valid integer : ").strip().replace(' ', '')
        if number[0] == '-':
            negative = True
            number = number[1:]
        else:
            negative = False
        while len(number) > 1 and number[0] == '0':
            number = number[1:]

    number = int(number)
    if negative:
        number = -number

    return number


def output_result(number, w, b):
    result = w * number + b
    print("Result is", result)
    input("\nPress Enter to quit...")


# Excecute everything!
def main(train_data_pairs, learning_rate=0.01, epoch=500000, init_w=0, init_b=0):
    w, b = train(train_data_pairs, learning_rate, epoch, init_w, init_b)
    number = take_input()
    output_result(number, w, b)


if __name__ == '__main__':
    # Pairs of training inputs and outputs (x, 2x - 1)
    train_xy_pairs = ((0, -1),
                      (1, 1),
                      (2, 3),
                      (3, 5),
                      (4, 7),
                      (5, 9),
                      (6, 11),
                      (7, 13),
                      (8, 15),
                      (9, 17),
                      (100, 199),
                      (1000, 1999),
                      (50000, 99999),
                      (-4000, -8001),
                      (-7777, -15555))
    main(train_xy_pairs)
