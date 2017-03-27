# Source: https://rstudio.github.io/tensorflow/tutorial_mnist_pros.html
# Comments in this file will reflect mostly the changes from "beginner.R",
# so see that file for a fuller explanation of what each step is doing.

library(tensorflow)

print("Loading dataset")
datasets <- tf$contrib$learn$datasets
mnist <- datasets$mnist$read_data_sets("MNIST-data", one_hot = TRUE)

# This tutorial uses an InteractiveSession, which lets us be more flexible
# about structuring the code. The tensorflow online docs recommend that if
# you're NOT using an InteractiveSession you should build the entire computation
# graph before starting a session and launching the graph.
print("Starting interactive session")
sess <- tf$InteractiveSession()

print("Building placeholders")
x <- tf$placeholder(tf$float32, shape(NULL, 784L))
y_ <- tf$placeholder(tf$float32, shape(NULL, 10L))

print("Defining variables")
W <- tf$Variable(tf$zeros(shape(784L, 10L)))
b <- tf$Variable(tf$zeros(shape(10L)))

# Note initialization difference between interactive and non-interactive
# sessions.
print("Initializing variables")
sess$run(tf$global_variables_initializer())

print("Defining model")
y <- tf$nn$softmax(tf$matmul(x, W) + b)

print("Defining cross-entropy loss function")
cross_entropy <- tf$reduce_mean(-tf$reduce_sum(y_ * tf$log(y), reduction_indices=1L))

print("Setting up optimization algorithm")
optimizer <- tf$train$GradientDescentOptimizer(0.5)
train_step <- optimizer$minimize(cross_entropy)

print("TRAINING!!!")
for (i in 1:1000) {
  batches <- mnist$train$next_batch(100L)
  batch_xs <- batches[[1]]
  batch_ys <- batches[[2]]
  sess$run(train_step,
           feed_dict = dict(x = batch_xs, y_ = batch_ys))
}

print("Evaluating!")
correct_prediction <- tf$equal(tf$argmax(y, 1L), tf$argmax(y_, 1L))
accuracy <- tf$reduce_mean(tf$cast(correct_prediction, tf$float32))
result <- accuracy$eval(feed_dict = dict(x = mnist$test$images, y_ = mnist$test$labels))
print(result)

# Now the good stuff! Bring the model's performance from "embarrassing" to
# "acceptable". To do that we'll build a small convolutional neural network (CNN).

# To make the model, we will need a bunch of weights and biases. So we'll make two
# functions to help. It is apparently a good idea when initializing weights to
# add a small amount of noise to break symmetry, and to prevent 0 gradients.
# And also apparently a good practice to initialize ReLU ('https://en.wikipedia.org/wiki/Rectifier_(neural_networks)')
# neurons with a slight positive bias to avoid dead neurons. Rather than doing
# all of this repeatedly we can do it inside the helper functions.
print("Defining CNN initialization helper functions")
weight_variable <- function(shape) {
  initial <- tf$truncated_normal(shape, stddev = 0.1)
  tf$Variable(initial)
}

bias_variable <- function(shape) {
  initial <- tf$constant(0.1, shape = shape)
  tf$Varialbe(initial)
}

# TensorFlow's convolution and pooling operations allow for great flexibility.
# But in this tutorial we'll choose a "vanilla" setting for every option we're
# given. Convolutions will use a stride of 1 and will be zero-padded so output
# and input are the same size. Pooling is "plain old" max pooling over 2x2
# blocks. To keep the code clean these are also abstracted into functions.
print("Defining convolution and pooling operations")
conv2d <- function(x, W) {
  tf$nn$conv2d(x, W, strides = c(1L, 1L, 1L, 1L), padding = 'SAME')
}

max_pool_2x2 <- function(x) {
  tf$nn$max_pool(
    x,
    ksize = c(1L, 2L, 2L, 1L),
    strides = c(1L, 2L, 2L, 1L),
    padding='SAME'
  )
}

# First convolutional layer! It consists of convolution, followed by max pooling.
# The convolutional will compute 32 features for each 5x5 patch. First two
# dimensions are the patch size, the next is the number of input channls, and
# the last is the number of output channels. There's also a bias vector with a
# component for each output channel.
print("Building first convolutional layer")
W_conv1 <- weight_variable(shape(5L, 5L, 1L, 32L))
b_conv1 <- bias_variable(shape(32L))

# To apply the first layer, reshape x to a 4D tensor. 2nd and 3rd dimensions
# correspond to image width and height, 4th dimension is number of color channels.
x_image <- tf$reshape(x, shape(-1L, 28L, 28L, 1L))

# Finally, convolve x_image with the weight tensor, add the bias, apply the
# ReLU function and max pool.
h_conv1 <- tf$nn$relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 <- max_pool_2x2(h_conv1)

# Second convolutional layer! To build a deep network we stack several layers
# of this type. The second layer has 64 features for each 5x5 patch.
print("Building second convolutional layer")
W_conv2 <- weight_variable(shape(5L, 5L, 32L, 64L))
b_conv2 <- bias_variable(shape(64L))

h_conv2 <- tf$nn$relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 <- max_pool_2x2(h_conv2)

# Densely connected layer! Image has been reduced to 7x7 (NB: wut?). Now add
# a fully-connected layer with 1024 neurons to allow processing on the entire
# image. Reshape the tensor from pooling layer to batch of vectors, and then
# multiply by weight matrix, add the bias, and apply a ReLU.
print("Building densely connected layer")
W_fc1 <- weight_variable(shape(7L * 7L * 64L, 1024L))
b_fc1 <- bias_variable(shape(1024L))

h_pool2_flat <- tf$reshape(h_pool2, shape(-1L, 7L * 7L * 64L))
h_fc1 <- tf$nn$relu(tf$matmul(h_pool2_flat, W_fc1) + b_vc1)

# Dropout! Apply dropout (https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)
# before the readout layer to reduce overfitting. Create a placeholder for the
# probability that a neuron's output is kept during dropout: this allows us to
# turn dropout on during training and turn it off for testing. tf$nn$dropout
# automatically handles scaling neuron outputs in addition to masking them, so
# it "just works" without additional scaling.
print("Building a dropout layer")
keep_prob <- tf$placeholder(tf$float32)
h_fc1_drop <- tf$nn$dropout(h_fc1, keep_prob)

# Readout layer! Finally, we add a softmax layer just like the one-layer
# softmax regression we tried originally.
W_fc2 <- weight_variable(shape(1024L, 10L))
b_fc2 <- weight_variable(shape(10L))

y_conv <- tf$nn$softmax(tf$matmul(h_fc1_drop, W_fc2) + b_fc2)

# Finally... Train and evaluate the model! Training is nearly the same as the
# softmax example, with the following key differences:
#   1. Replace gradient descent optimizer with more sophisticated ADAM optimizer
#   2. Include additional param "keep_prob" in "feed_dict" to control dropout rather
#   3. Add logging every 100th iteration
# Note that this training may take a while...
print("TRAINING CNN!!!")
cross_entropy <- tf$reduce_mean(-tf$reduce_sum(y_ * tf$log(y_conv), reduction_indices = 1L))
train_step <- tf$train$AdamOptimizer(1e-4)$minimize(cross_entropy)
correct_prediction <- tf$equal(tf$argmax(y_conv, 1L), tf$argmax(y_, 1L))
accuracy <- tf$reduce_mean(tf$cast(correct_prediction, float32))
sess$run(tf$global_variables_initializer())

for (i in 1:20000) {
  batch <- mnist$train$next_batch(50L)
  if (i %% 100 == 0) {
    train_accuracy <- accuracy$eval(feed_dict = dict(x = batch[[1]], y_ = batch[[2]], keep_prob = 1.0))
    cat(sprintf("step %d, training accuracy %g\n", i, train_accuracy))
  }
  train_step$run(feed_dict = dict(x = batch[[1]], y_ = batch[[2]], keep_prob = 0.5))
}

print("Evaluating model accuracy")
train_accuracy <- accuracy$eval(feed_dict = dict(x = mnist$test$images, y_ = mnist$test$labels, keep_prob = 1.0))
cat(sprintf("test accuracy %g\n", train_accuracy))
