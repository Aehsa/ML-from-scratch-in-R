#Linear Regression

#forward pass
predict <- function(X, w, b){
  return (as.vector(b + X%*%w))
}

#MSE
cost = function(X, w, b, y, lambda){
  n <- length(y)
  mse <- sum((predict(X, w, b) - y)**2) / (2*n)
  regularizaton <- (lambda / (2 * n)) * sum(w^2)
  cost <- mse + regularizaton
  return(cost)
}

#gradient calculation
gradient <- function(X, w, b, y, lambda){
  n <- length(y)
  error <- (predict(X, w, b) - y) # nx1 array
  dw <- (t(X) %*% error) / n + (lambda * w) / n
  db <- sum(error) / n
  return(list(dw = dw, db = db))
}

#optimization
update_weights = function(X, w, b, y, alpha, lambda){
  gradients <- gradient(X, w, b, y, lambda)
  w <- w - alpha * gradients$dw 
  b <- b - alpha * gradients$db
  return(list(w = w, b = b))
}

# Approximating the Taylor expansion of sin(x) using linear regression

x <- -40:40
x <- x/10
y <- sin(x)

# Generating Polynomial Values of X for training
p <- 7 # degree of polynomial to train
X <- sapply(1:p, function(p) x^p)
X <- scale(X)
w <- rnorm(p, mean=0, sd=sqrt(1/p))  # Initial weights
b <- 0          # Initial bias
alpha <- 0.1    # Learning rate
lambda <- 0.1    # Regularization strength
iterations <- 1000  # Number of iterations

# Fitting the model using gradient descent
history <- c()
for (i in 1:iterations) {
  results <- update_weights(X, w, b, y, alpha, lambda)
  w <- results$w
  b <- results$b
  
  current_cost <- cost(X, w, b, y, lambda)
  history <- c(history, current_cost)
  # Optionally print cost every 100 iterations
  if (i %% 100 == 0) {
    cat("Iteration", i, "MSE:", current_cost, "\n")
  }
}

# Assessing the model
final_cost <- cost(X, w, b, y, lambda)
cat("Final MSE after", iterations, "iterations:", final_cost, "\n")

cat("final weights", w)
cat("final intercept", b)

# Plotting
par(mfrow=c(1,3))
plot(1:length(history), history)
plot(x, predict(X, w, b))
plot(x, y)