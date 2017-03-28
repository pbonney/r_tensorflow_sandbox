# Stepping out on my own...

library(tensorflow)
np <- import("numpy")

sess <- tf$InteractiveSession()

print("Load training set")
training_set <- tf$contrib$learn$datasets$base$load_csv_with_header(
  filename = "iris/iris_training.csv",
  target_dtype = np$int,
  features_dtype = np$float32
)

print("Load test set")
test_set <- tf$contrib$learn$datasets$base$load_csv_with_header(
  filename = "iris/iris_test.csv",
  target_dtype = np$int,
  features_dtype = np$float32
)

print("Define variables and placeholders")
x <- tf$placeholder(tf$float32, shape(NULL, 4L))
W <- tf$Variable(tf$zeros(shape(4L, 3L)))
b <- tf$Variable(tf$zeros(shape(3L)))

y <- tf$nn$softmax(tf$matmul(x, W) + b)

y_ <- tf$placeholder(tf$float32, shape(NULL, 3L))

print("Define optimizer")
cross_entropy <- tf$reduce_mean(-tf$reduce_sum(y_ * tf$log(y), reduction_indices=1L))

optimizer <- tf$train$GradientDescentOptimizer(0.5)
train_step <- optimizer$minimize(cross_entropy)

sess$run(tf$global_variables_initializer())

x_in <- training_set$data
y_in <- tf$one_hot(training_set$target, 3L)$eval()

print("Train!!!")
for (i in 1:1000) {
  sess$run(train_step,
           feed_dict = dict(x = x_in, y_ = y_in))
}

print("Evaluate!")
correct_prediction <- tf$equal(tf$argmax(y, 1L), tf$argmax(y_, 1L))
accuracy <- tf$reduce_mean(tf$cast(correct_prediction, tf$float32))

x_test <- test_set$data
y_test <- tf$one_hot(test_set$target, 3L)$eval()

result <- sess$run(accuracy, feed_dict = dict(x = x_test, y_ = y_test))
print(result)
