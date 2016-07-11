# Neuron class
Neuron class provides LNU (Linear Neural Unit), QNU (Quadratic Neural Unit), RBF (Radial Basis Function), MLP (Multi Layer Perceptron), MLP-ELM (Multi Layer Perceptron - Extreme Learning Machine) neurons learned with Gradient descent or LeLevenberg–Marquardt algorithm. This class is suitable for prediction on time series.

# Dependencies
Neuron class needs pandas and numpy to work propertly.

# Example of usage

Consider *Y* are targets and *X* are inputs.

## LNUGD
```python
neuron = LNUGD()
prediction = 1
yn, w, e, Wall, MSE = neuron.train(Y_train, X_train, epochs=2, prediction=prediction)
yn, w, Wall, MSE, e = neuron.countSerie(Y, X, logging=False, prediction=prediction)
```

## QNULM
```python
neuron = QNULM()
prediction = 0
yn, w, e, Wall, MSE = neuron.train(Y_train, X_train, epochs=10, prediction=prediction)
yn, w, MSE, e = neuron.countSerie(Y, X, logging=False, prediction=prediction)
```

## RBF
```python
neuron = RBF()
prediction = 1
neuron.train(Y_train, X_train, prediction=prediction)
yn = neuron.count(Y,X, logging=True, beta=0.01, prediction=prediction)
```
## MLPGD
```python
neuron = MLPGD()
prediction = 0
yn = neuron.count(Y_train, X_train, prediction=prediction, epochs=5)
yn = neuron.count(Y, X, prediction=prediction, epochs=1)
```
## MLPELM
```python
neuron = MLPELM()
prediction = 1
yn = neuron.count(Y_train, X_train, prediction = prediction, epochs = 10)
yn = neuron.count(Y, X, prediction = prediction)
```
## MLPLMWL
```python
neuron = MLPLMWL()
prediction = 1
yn = neuron.count(Y, X, learningWindow = 50, overLearn = 10,  prediction = prediction)
```
