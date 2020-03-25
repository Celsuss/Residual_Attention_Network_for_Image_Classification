import tensorflow as tf
import datetime

def getTrainAndTestSummaryWriters():
    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    train_log_dir = 'logs/' + current_time + '/train'
    test_log_dir = 'logs/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    return train_summary_writer, test_summary_writer

def printTrainingBatchProgress(epoch, epochs, n_batch, n_batches, train_loss, train_accuracy):
    template = '[Epoch {}/{}, Batch {}/{}] Loss: {:.3f}, Accuracy: {:.2%}'
    print(template.format(epoch, epochs, n_batch, n_batches, train_loss.result(), train_accuracy.result()), end='\r')

def printTrainingEpochProgress(epoch, epochs, n_batch, n_batches, train_loss, train_accuracy, test_loss, test_accuracy):
    template = '[Epoch {}] Loss: {:.3f}, Accuracy: {:.2%}, Test Loss: {:.3f}, Test Accuracy: {:.2%}'
    print(template.format(epoch+1, train_loss.result(), train_accuracy.result(), test_loss.result(), test_accuracy.result()))

def printTesting(test_accuracy, model_name='model'):
    template = '[Testing] Test Accuracy: {:.2%} for {}'
    print(template.format(test_accuracy.result(), model_name))