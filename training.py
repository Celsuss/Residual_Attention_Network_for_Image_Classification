import tensorflow as tf
import trainLogging
import utils

# TODO: Uncomment this, only for debuging
# @tf.function
def trainStep(model, x, y, loss_op, optimizer, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        predictions = model(x)
        loss = loss_op(y, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(y, predictions)
    
# TODO: Uncomment this, only for debuging
# @tf.function
def testStep(model, x, y, loss_op, test_loss, test_accuracy):
    predictions = model(x)
    loss = loss_op(y, predictions)

    test_loss(loss)
    test_accuracy(y, predictions)

def getTrainAndTestMetrics():
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
    return train_loss, train_accuracy, test_loss, test_accuracy

def resetMetrics(train_loss, train_accuracy, test_loss, test_accuracy):
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()
    return train_loss, train_accuracy, test_loss, test_accuracy

def testModel(model, x, y, model_name='model'):
    train_loss, train_accuracy, test_loss, test_accuracy = getTrainAndTestMetrics()
    predictions = model(x)
    test_accuracy(y, predictions)
    trainLogging.printTesting(test_accuracy, model_name)

    return 0

def train(model, x_train, y_train, x_test, y_test, loss_op, optimization, epochs, model_save_path='', model_name='model', save_weights=False, save_keras_model=False):
    print('Starting training for model {}'.format(model_name))
    train_loss, train_accuracy, test_loss, test_accuracy = getTrainAndTestMetrics()
    train_summary_writer, test_summary_writer = trainLogging.getTrainAndTestSummaryWriters()
    n_batches = len(x_train)
    
    best_accuracy = 0
    best_model = None

    for epoch in range(epochs):
        n_batch = 0
        for x, y in zip(x_train, y_train):
            n_batch+=1

            trainLogging.printTrainingBatchProgress(epoch+1, epochs, n_batch, n_batches, train_loss, train_accuracy)
            trainStep(model, x, y, loss_op, optimization, train_loss, train_accuracy)

        testStep(model, x_test, y_test, loss_op, test_loss, test_accuracy)
        trainLogging.printTrainingEpochProgress(epoch+1, epochs, n_batch, n_batches, train_loss, train_accuracy, test_loss, test_accuracy)

        if test_accuracy.result() > best_accuracy:
            best_accuracy = test_accuracy.result()
            best_model = model
            if save_weights is True:
                utils.saveModelWeights(model, model_save_path, model_name)
            if save_keras_model is True:
                utils.saveKerasModel(model, model_save_path, model_name)

            print('New best model with accuracy: {}'.format(best_accuracy))

        # Reset the metrics for the next epoch
        train_loss, train_accuracy, test_loss, test_accuracy = resetMetrics(train_loss, train_accuracy, test_loss, test_accuracy)

    return best_model