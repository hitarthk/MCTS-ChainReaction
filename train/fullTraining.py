import numpy as np
import tensorflow as tf
from configs.defaultConfigs import config
from algo.MCTS import ExperienceCollector
from models.ResnetFeatures import IntuitionPolicy

from sklearn.utils import shuffle
def loss(model, inputs, actualValues, analysisProbs):
    _, predictedValues, intuitionLogitProbs = model(inputs, training = True)
    mainLoss =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(analysisProbs, intuitionLogitProbs) + tf.keras.losses.MSE(actualValues, predictedValues))
    regLoss = tf.add_n(model.losses)
    return tf.add(mainLoss, regLoss)

def grad(model, inputs, actualValues, analysisProbs):
    with tf.GradientTape() as tape:
        lossValue = loss(model, inputs, actualValues, analysisProbs)
    return lossValue, tape.gradient(lossValue, model.trainable_variables)

def getOptimizer(datasetSize):
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        0.001,
        decay_steps = datasetSize,
        decay_rate=1,
        staircase=False)

    return tf.keras.optimizers.Adam(lr_schedule)

def trainLoop(model, inputs, actualValues, analysisProbs, numValidation = 1000, numEpochs = 10, batchSize = 32):
    optimizer = getOptimizer()

    permutation = np.random.permutation(len(inputs))
    inputs, actualValues, analysisProbs = shuffle(inputs, actualValues, analysisProbs)

    trainInputs, trainActualValues, trainAnalysisProbs = inputs[numValidation:], actualValues[numValidation:], analysisProbs[numValidation:]
    valInputs, valActualValues, valAnalysisProbs = inputs[:numValidation], actualValues[: numValidation], analysisProbs[: numValidation]

    trainDataset = tf.data.Dataset.from_tensor_slices((trainInputs, (trainActualValues, trainAnalysisProbs)))
    valDataset = tf.data.Dataset.from_tensor_slices((valInputs, (valActualValues, valAnalysisProbs)))

    trainDataset.batch(batchSize).repeat(numEpochs)
    valDataset.batch(batchSize).repeat(numEpochs)

    for epoch in range(numEpochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_val_loss_avg = tf.keras.metrics.Mean()
        for inputBatch, outBatch in trainDataset:
            actualValuesBatch = outBatch[0]
            analysisProbsBatch = outBatch[1]
            lossValue, grads = grad(model, inputBatch, actualValuesBatch, analysisProbsBatch)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss_avg(lossValue)


        for inputBatch, outBatch in valDataset:
            actualValuesBatch = outBatch[0]
            analysisProbsBatch = outBatch[1]
            lossValue = loss(model, inputBatch, actualValuesBatch, analysisProbsBatch)
            epoch_val_loss_avg(lossValue)

        print(f'Epoch: {epoch}: trainLoss: {epoch_loss_avg.result()}: valLoss: {epoch_val_loss_avg}')


def trainBrain(config):
    intuitionPolicy = IntuitionPolicy(config.totalRows, config.totalCols, config.numPlayers, config.l2Weight, config.numResnetBlocks,
    config.filters)
    intuitionPolicy.build(input_shape = (None, config.totalRows, config.totalCols, config.numPlayers))
    print(intuitionPolicy.summary())
    for i in range(config.totalLearningIterations):
        exp = ExperienceCollector()
        exp.collectExperience(config.gamesPerTraining, intuitionPolicy)
        buf = exp.buffer
        inputs, actualValues, analysisProbs = buf.states, buf.rewards, buf.analysisProbs
        trainLoop(intuitionPolicy, inputs, actualValues, analysisProbs)
        print(f'Done {i} runs of self play')

if __name__=='__main__':
    trainBrain(config)



