import numpy as np
import theano
import theano.tensor as T
import theano.printing as printing
import lasagne
import cPickle
import lasagne.layers.helper as helper
import random

#This snippet loads the text file and creates dictionaries to 
#encode characters into a vector-space representation and vice-versa. 

#Lasagne Seed for Reproducibility
lasagne.random.set_rng(np.random.RandomState(1))

vocab_size = 84 #3
# Sequence Length
SEQ_LENGTH = 96 #4

# Number of units in the two hidden (LSTM) layers
N_HIDDEN = 256

# Optimization learning rate
LEARNING_RATE = 0.1

# All gradients above this will be clipped
GRAD_CLIP = 100

# How often should we check the output?
PRINT_FREQ = 1000

# Number of epochs to train the net
NUM_EPOCHS = 50

# Batch Size
BATCH_SIZE = 128 #2

# Number of tags
NUM_TAGS = 46 #2

char_dims = 10
dim_out = 128

def get_mask(x):
    mask = np.ones_like(x)
    mask[np.where(x==0.)] = 0
    return mask 

def main(num_epochs=NUM_EPOCHS, layers=1, load_file=None):
    print "Building network ..."
    # First, we build the network, starting with an input layer
    # Recurrent layers expect input of shape
    # (batch size, SEQ_LENGTH, num_features)

    x = T.matrix('x')
    mask = T.matrix('mask')
    # Theano tensor for the targets
    target_values = T.ivector('target_output')
    
    # We now build a layer for the embeddings.
    U = np.random.randn(vocab_size, char_dims).astype('float32')
    embeddings = theano.shared(U, name='embeddings', borrow=True)
    x_embedded = embeddings[x.astype('int32')]

    l_in = lasagne.layers.InputLayer(shape=(None, None, char_dims))


    recurrent_type = lasagne.layers.LSTMLayer
    # We now build the LSTM layer which takes l_in as the input layer
    # We clip the gradients at GRAD_CLIP to prevent the problem of exploding gradients. 
    l_forward_1 = recurrent_type(
        l_in, N_HIDDEN, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh)
    l_backward_1 = recurrent_type(
        l_in, N_HIDDEN, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh,
        backwards=True)

    if layers == 2:
        l_forward_2 = recurrent_type(
            l_forward_1, N_HIDDEN, grad_clipping=GRAD_CLIP,
            nonlinearity=lasagne.nonlinearities.tanh)
        l_backward_2 = recurrent_type(
            l_backward_1, N_HIDDEN, grad_clipping=GRAD_CLIP,
            nonlinearity=lasagne.nonlinearities.tanh,
            backwards=True)
        l_forward_slice = l_forward_2.get_output_for([x_embedded, mask])[:,-1,:]
        l_backward_slice = l_backward_2.get_output_for([x_embedded, mask])[:,-1,:]
    else:
        l_forward_slice = l_forward_1.get_output_for([x_embedded, mask])[:,-1,:]
        l_backward_slice = l_backward_1.get_output_for([x_embedded, mask])[:,-1,:]

    # Now combine the LSTM layers.  
    _Wf, _Wb = np.random.randn(N_HIDDEN, dim_out).astype('float32'), np.random.randn(N_HIDDEN, dim_out).astype('float32')
    _bias = np.random.randn(dim_out).astype('float32')
    wf = theano.shared(_Wf, name='join forward weights', borrow=True)
    wb = theano.shared(_Wb, name='join backward weights', borrow=True)
    bias = theano.shared(_bias, name='join bias', borrow=True)

    joined = T.dot(l_forward_slice, wf) + T.dot(l_backward_slice, wb) + bias
    tmp = lasagne.layers.InputLayer(shape=(BATCH_SIZE, dim_out))
    l_out = lasagne.layers.DenseLayer(tmp, num_units=NUM_TAGS, W = lasagne.init.Normal(), nonlinearity=lasagne.nonlinearities.softmax)

    # lasagne.layers.get_output produces a variable for the output of the net
    network_output = l_out.get_output_for(joined)
    # The loss function is calculated as the mean of the (categorical) cross-entropy between the prediction and target.
    cost = T.nnet.categorical_crossentropy(network_output,target_values).mean()

    # Retrieve all parameters from the network
    #all_params = lasagne.layers.get_all_params(l_forward_1) + lasagne.layers.get_all_params(l_backward_1) + lasagne.layers.get_all_params(l_out) + [wf, wb, bias, embeddings] 
    all_params = helper.get_all_params(l_backward_2) + helper.get_all_params(l_forward_2) + helper.get_all_params(l_out) + [wf, wb, bias, embeddings] 
    #print len(all_params)

    grads = T.grad(cost, all_params)
    get_grads = theano.function([x, mask, target_values], grads)
    # Compute AdaGrad updates for training
    print("Computing updates ...")
    updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)

    # Theano functions for training and computing cost
    print("Compiling functions ...")
    train = theano.function([x, mask, target_values], cost, updates=updates, allow_input_downcast=True)
    compute_cost = theano.function([x, mask, target_values], cost, allow_input_downcast=True) 
    
    pred = T.argmax(network_output, axis=1)
    errors = T.sum(T.neq(pred, target_values))
    count_errors = theano.function([x, mask, target_values], errors, allow_input_downcast=True)

    def get_data(fname):
        import cPickle
        with open(fname, 'rb') as handle:
            data = cPickle.load(handle)
        return data[0], data[1]

    print 'Loading train'
    train_xs, train_ys = get_data('train')
    print 'Loading dev'
    dev_xs, dev_ys = get_data('dev')
    print 'Loading test'
    test_xs, test_ys = get_data('test')
    print 'Sizes:\tTrain: %d\tDev: %d\tTest: %d\n' % (len(train_xs) * BATCH_SIZE, len(dev_xs) * BATCH_SIZE, len(test_xs) * BATCH_SIZE)


    def get_accuracy(pXs, pYs):
        total = sum([len(batch) for batch in pXs])
        errors = sum([count_errors(tx, get_mask(tx), ty) for tx, ty in zip(pXs, pYs)])
        return float(total-errors)/total

    if not load_file is None:
        print 'Loading params...'
        with open(load_file, 'rb') as handle:
            params = cPickle.load(handle)
        print len(params)
        for ix, _ in enumerate(zip(params, all_params)):
            all_params[ix].set_value(params[ix].astype('float32'))

    print("Training ...")
    try:
        model_name = 'LSTM_%d' % layers
        best_acc = 0.0
        for it in xrange(num_epochs):
            data = zip(train_xs, train_ys)
            random.shuffle(data)
            train_xs, train_ys = zip(*data)

            avg_cost = 0;
            total = 0.
            cur = 0
            for x, y in zip(train_xs, train_ys):
                cur += 1                
                avg_cost += train(x, get_mask(x), y)
                total += 1.

            train_acc = 0.
            #train_acc = get_accuracy(train_xs, train_ys)
            dev_acc = get_accuracy(dev_xs[55:60], dev_ys[55:60])
            test_acc = get_accuracy(test_xs, test_ys)

            if dev_acc > best_acc:
                params = [np.asarray(p.eval()) for p in all_params]
                with open('%s_%f.pkl' % (model_name, dev_acc), 'wb') as handle:
                    cPickle.dump(params, handle)
                best_acc = dev_acc

            print("Epoch {} average loss = {}".format(it, avg_cost / total))
            print "Accuracies:\t train: %f\tdev: %f\ttest: %f\n" % (train_acc, dev_acc, test_acc) 
     
    except KeyboardInterrupt:
        pass

import sys
if __name__ == '__main__':
    f = sys.argv[1]
    main(10, layers=2)
