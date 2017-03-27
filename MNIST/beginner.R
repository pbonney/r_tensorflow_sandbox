# Source: https://rstudio.github.io/tensorflow/tutorial_mnist_beginners.html
# Full code behind that tutorial lives here: https://github.com/rstudio/tensorflow/blob/master/inst/examples/mnist/mnist_softmax.R

library(tensorflow)

print("Loading dataset")
datasets <- tf$contrib$learn$datasets
mnist <- datasets$mnist$read_data_sets("MNIST-data", one_hot = TRUE)

# MNIST images are 28 pixels x 28 pixels, flattened to a 784-dimensional vector.
# We'll represent them as a 2-D tensor of floating point numbers. "NULL" in the
# first dimension means that it can be of any length.
print("Creating placeholder for x")
x <- tf$placeholder(tf$float32, shape(NULL, 784L))

# A "Variable" is a modifiable tensor that lives in the computational graph. It
# can be used and modified by the computation.
#
# We'll initialize two of them with zeros. "10" here is for the 10 digits we're
# trying to learn. "W" will multiply the 784-dimensional image vectors to produce
# 10-dimensional vectors of evidence for the different classes. "b" has shape 10
# so it can be added to the output.
print("Defining variables")
W <- tf$Variable(tf$zeros(shape(784L, 10L)))
b <- tf$Variable(tf$zeros(shape(10L)))

# Now we define the model - note how compact the definition is! We first multiply
# x by W, then add b, and finally apply tf$nn$softmax.
print("Defining model")
y <- tf$nn$softmax(tf$matmul(x, W) + b)

#
# TRAINING THE MODEL
#

# First, we need a loss function. One very common one is "cross-entropy". Learn
# more about it here: http://colah.github.io/posts/2015-09-Visual-Information/

# To implement cross-entropy first add a new placeholder to input the correct
# answers.
print("Placeholder for correct answers")
y_ <- tf$placeholder(tf$float32, shape(NULL, 10L))

# Now implement the cross-entropy function
print("Define cross-entropy function")
cross_entropy <- tf$reduce_mean(-tf$reduce_sum(y_ * tf$log(y), reduction_indices=1L))
# (What the heck is happening here? First tf$log computs the log of each element of
# y. Then we multiply each element of y_ by the corresponding element of tf$log(y).
# Then tf$reduce_sum adds elements in y's second dimension, courtesy of the
# "reduction_indices=1L" parameter. Finally tf$reduce_mean computes the mean over
# the whole batch.)

# NB: Tensor indices (like "reduction_indices=1L") are 0-based inside the TF API,
# NOT 1-based as usual in R.

# Now choose your optimization algorithm of choice for backpropagation.
print("Set up optimization algorithm")
optimizer <- tf$train$GradientDescentOptimizer(0.5)
train_step <- optimizer$minimize(cross_entropy)

# Before you start training, create an operation to initialize all the variables
# we created. Note that this step just defines the initialization, it doesn't
# execute it.
print("Set up session")
init <- tf$global_variables_initializer()

# Now create our session, and initialize our variables
sess <- tf$Session()
sess$run(init)

# And finally, train!
print("TRAIN!!!")
for (i in 1:1000) {
  batches <- mnist$train$next_batch(100L) # Stochastic training: take random slice of 100 datapoints on each training step, rather than the whole dataset - for the sake of performance
  batch_xs <- batches[[1]]
  batch_ys <- batches[[2]]
  sess$run(train_step,
           feed_dict = dict(x = batch_xs, y_ = batch_ys))
}

#
# EVALUATING THE MODEL
#

# First figure out where we predicted the correct label. tf$argmax gives the
# highest entry in a tensor along a given axis. tf$argmax(y, 1L) gives us the
# label or model thinks is most likely for each input, while tf$argmax(y_, 1L)
# is the actual correct label. We'll compare them and get a vector of booleans,
# which we'll then cast to floats and then compute the mean to get the overall
# accuracy.
print("Evaluate!")
correct_prediction <- tf$equal(tf$argmax(y, 1L), tf$argmax(y_, 1L))
accuracy <- tf$reduce_mean(tf$cast(correct_prediction, tf$float32))

# Finally, ask for the accuracy on our test data
result <- sess$run(accuracy, feed_dict = dict(x = mnist$test$images, y_ = mnist$test$labels))
