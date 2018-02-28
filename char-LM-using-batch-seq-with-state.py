"""
This script implements a simple charecter LM. It meants to try batch_sequence_with_states feature.
This is partially based on https://github.com/f90/Tensorflow-Char-LSTM example.

Problems: It seems this approach cant applied to large vocabularies (at least on my laptop) and program crash.

Amir Harati, 2018

"""

import os
import tensorflow as tf
import numpy as np
import threading
import sys


class CharLM:
    """ A wrapper class for training and testing using Model and
        InputPipeline. """
    def __init__(self, batch_size, hidden_size, num_layers,
                 num_steps, keep_prob, data_path,
                 sess, max_itr, checkpoints_dir, logs_dir):
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.keep_prob = keep_prob
        self.sess = sess
        self.checkpoints_dir = checkpoints_dir
        self.max_itr = max_itr
        num_enqueue_threads = 4

        self.input_pipeline = InputPipeline(data_path, self.sess)
        key, context, sequences = self.input_pipeline.key_context_sequences
        self.vocab_size = self.input_pipeline.vocab_size

        self.model = Model(self.batch_size,
                           self.hidden_size,
                           self.vocab_size + 1,
                           self.num_layers,
                           self.num_steps,
                           self.keep_prob,
                           key, context, sequences, num_enqueue_threads)

        self.histograms = [tf.summary.histogram(var.op.name, var)
                           for var in tf.trainable_variables()]
        self.summary_op = tf.summary.merge_all()
        # Create summary writer
        self.summary_writer = \
            tf.summary.FileWriter(logs_dir, sess.graph)

    def train(self, saver):
        """ train the model."""
        # start input pipe
        # TODO: change this
        self.input_pipeline.start()
        itr = 0
        while itr < self.max_itr:
            itr += 1
            [res_loss, _, res_global_step, summary] = \
                self.sess.run(self.model.trainOps + [self.summary_op], feed_dict={self.model.is_training: True})

            if res_global_step % 100 == 0:
                print("loss: ", res_loss)
                # print("prepL", np.exp(costs/lens))
            self.summary_writer.add_summary(summary,
                                            global_step=int(res_global_step))
            if res_global_step % 500 == 0:
                print("Saving model...")
                saver.save(self.sess, self.checkpoints_dir + "model",
                           global_step=int(res_global_step))
                self.model.sample(self.sess, self.input_pipeline.ids_to_words)

        # stop the inputpipe
        self.input_pipeline.stop()

    def test(self):
        """ test the model. """
        # CHECKPOINTING
        # # Load pretrained model to test
        #latestCheckpoint = tf.train.latest_checkpoint(self.checkpoints_dir +
        #                                              "model")
        #restorer = tf.train.Saver(tf.global_variables(),
        #                          write_version=tf.train.SaverDef.V2)
        #restorer.restore(self.sess, latestCheckpoint)
        print('Pre-trained model restored')
        self.input_pipeline.start()
        iteration = 0
        while True:
            try:
                [cost, perplexity, summary] = \
                    self.sess.run(self.model.testOps + [self.summary_op])
            except tf.errors.OutOfRangeError:
                print("Finished testing!")
                break
            self.summary_writer.add_summary(summary, global_step=int(iteration))
            iteration += 1
            print("cost:", cost)
            print("perplexity:", perplexity)

        self.input_pipeline.stop()



class InputPipeline:
    """ Define a data pipeline that fetch data using a seperate thread.
        Based on: https://github.com/f90/Tensorflow-Char-LSTM
        TODO: Can we use the QueueRunner instead of python threading?
    """
    def __init__(self, data_path, sess):
        queue_capacity = 5000
        min_queue_capacity = 2000

        # load the data
        self.dataset, self.ids_to_words = self._load_data(data_path)

        self._vocab_size = len(self.ids_to_words)

        self.sess = sess

        self.keyInput = tf.placeholder(tf.string)  # To identify each sequence
        self.lengthInput = tf.placeholder(tf.int32)  # Length of sequence
        self.seqInput = tf.placeholder(tf.int32, shape=[None])  # Input sequence
        self.seqOutput = tf.placeholder(tf.int32, shape=[None])  # Output sequence

        self.q = tf.RandomShuffleQueue(queue_capacity,
                                       min_queue_capacity,
                                       [tf.string, tf.int32,
                                        tf.int32, tf.int32])
        self.enqueue_op = self.q.enqueue([self.keyInput, self.lengthInput,
                                         self.seqInput, self.seqOutput])

        with tf.device("/cpu:0"):
            self.key, contextT, sequenceIn, sequenceOut = self.q.dequeue()
            self.context = {"length": tf.reshape(contextT, [])}
            self.sequences = {"inputs": tf.reshape(sequenceIn, [contextT]), "outputs": tf.reshape(sequenceOut, [contextT])}

    @property
    def key_context_sequences(self):
        return self.key, self.context, self.sequences

    @property
    def vocab_size(self):
        return(self._vocab_size)

    def _load_data(self, data_path_a):
        '''
        Load data and convert to proper format.
        It loads PBT data format.
        '''
        # extract unique list of words from training data
        # words = list(set([word.strip() for line in open(train_file) for word in line.split()]))
        words = list(set([word for line in open(data_path_a) for word in line]))

        # map word->ids and ids->words
        # we also want to start from 1
        # it should be sorted so it works from saved models
        words_to_ids = {w: id + 1 for id, w in enumerate(sorted(words))}
        print(words_to_ids)
        ids_to_words = {words_to_ids[x]: x for x in words_to_ids}
        # load data in form of ids.
        data_lines = [line for line in open(data_path_a)]
        data = []
        for line in data_lines:
            tmp = [words_to_ids[word] for word in line]
            data.append(tmp)

        return data, ids_to_words

    # Enqueueing method in different thread, loading sequence examples and feeding into FIFO Queue
    def _load_and_enqueue(self):
        run = True
        key = 0  # Unique key for every sample, even over multiple epochs (otherwise the queue could be filled up with two same-key examples)
        while run:
            print("input pipe is running....")
            for current_seq in self.dataset:
                try:
                    self.sess.run(self.enqueue_op, feed_dict={
                                  self.keyInput: str(key),
                                  self.lengthInput: len(current_seq) - 1,
                                  self.seqInput: current_seq[:-1],
                                  self.seqOutput: current_seq[1:]},
                                  options=tf.RunOptions(timeout_in_ms=60000))
                    # timeout so  queue get empty
                except tf.errors.DeadlineExceededError as e:
                    print("Timeout while waiting to enqueue into input queue! Stopping input queue thread!")
                    run = False
                    break
                key += 1
            print("Finished enqueueing all samples!")

    def start(self):
        """ Start feteching the input pipeline."""
        with tf.device("/cpu:0"):
            self.t = threading.Thread(target=self._load_and_enqueue)
            self.t.start()

    def stop(self):
        """ Close the input pipeline."""
        # Stop our custom input thread
        print("Stopping custom input thread")
        self.sess.run(self.q.close())  # Then close the input queue
        self.t.join(timeout=1)


class Model:
    """Define the Model for training and testing LM.
    """
    def __init__(self, batch_size, hidden_size, vocab_size, num_layers, num_steps, keep_prob,key, context, sequences, num_enqueue_threads):
        """ define the model.

        Arguments:
            batch_size {Integer} -- number of examples in the batch.
            hidden_size {Integer} -- size of the hidden layer for LSTM.
            vocab_size {Integer} -- size of vocabularies.
            num_layers {Integer} -- number of layers
            num_steps {Integer} -- number of steps for  unrolled LSTM.
            keep_prob {float} -- keep_prob for dropout.
            key:
            context
            sequeunces
            num_enqueue_threads
        """
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.keep_prob = keep_prob
        self.key = key
        self.context = context
        self.sequences = sequences
        self.num_enqueue_threads = num_enqueue_threads

        self.is_training = tf.placeholder(dtype=tf.bool, name="is_training")
        self.total_cost = tf.Variable(tf.constant(0.0), trainable=False)
        self.total_len = tf.Variable(tf.constant(0.0), trainable=False)

        # define the model graph
        self._create_model()
        #if self.is_training is True:
        self._train_op()
        self._sample_op()

    def _create_model(self):
        """define the  graph.
           This is a private method and will be called by __init__.
        """
        with tf.variable_scope("LSTM"):
            cells = list()
            self.initial_states = dict()

            for i in range(0, self.num_layers):
                # Here we can use different type of RNNs
                cell = tf.contrib.rnn.LSTMBlockCell(num_units=self.hidden_size, forget_bias=1.0)
                # apply the dropout
                # use same keep_prob for input/output
                if self.is_training is True:
                    cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                                         input_keep_prob=
                                                         self.keep_prob,
                                                         output_keep_prob=
                                                         self.keep_prob)
                cells.append(cell)
                self.initial_states["lstm_state_c_" + str(i)] = \
                    tf.zeros(cell.state_size[0], dtype=tf.float32)
                self.initial_states["lstm_state_h_" + str(i)] = \
                    tf.zeros(cell.state_size[1], dtype=tf.float32)

             # multilayer RNN.
            # TODO: test other options.
            self.cell = tf.nn.rnn_cell.MultiRNNCell(cells)

        with tf.variable_scope("batching"):
            # read batch in  differnt thread
            # This function get one example at the time; so internally create a batch?
            self.batch = tf.contrib.training.batch_sequences_with_states(
                input_key=self.key,
                input_sequences=self.sequences,
                input_context=self.context,
                input_length=tf.cast(self.context["length"], tf.int32),
                initial_states=self.initial_states,
                num_unroll=self.num_steps,
                batch_size=self.batch_size,
                num_threads=self.num_enqueue_threads,
                capacity=self.batch_size * self.num_enqueue_threads * 2)
            inputs = self.batch.sequences["inputs"]
            targets = self.batch.sequences["outputs"]

            self.mask = tf.sign(tf.abs(tf.cast(targets, dtype=tf.float32)))

            tf.summary.histogram("inputs", inputs)

        # For a charecter LM  embedding can be simple 1 hot vector.
        with tf.variable_scope("prepare_input"):
            embedding = tf.constant(np.eye(self.vocab_size), dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, inputs)

            # Reshape inputs (and targets respectively) into list of length T
            # (self.num_steps), each element is a Tensor of shape (batch_size, input_dimensionality)
            inputs_by_time = tf.split(inputs, self.num_steps, 1)

            inputs_by_time = [tf.squeeze(elem, squeeze_dims=1) for elem in inputs_by_time]
            targets_by_time = tf.split(targets, self.num_steps, 1)
            targets_by_time = [tf.squeeze(elem, squeeze_dims=1) for elem in targets_by_time]
            self.targets_by_time_packed = tf.stack(targets_by_time)

            tf.summary.histogram("inputs_by_time", inputs_by_time)

        # effectively is dynamic_rnn+ state saver
        with tf.variable_scope("RNN"):
             # TODO: can we replace this we dynamic_rnn + state_saver?
            state_name = list(self.initial_states.keys())
            self.seq_lengths = self.batch.context["length"]
            (self.outputs, state) = tf.nn.static_state_saving_rnn(self.cell, inputs_by_time, state_saver=self.batch, sequence_length=    self.seq_lengths, state_name=state_name, scope='STATE_SAVER_RNN')

        # TODO: use get_variable annd use vars instead of consts.
        tf.summary.histogram("outputs", self.outputs)
        with tf.variable_scope("softmax"):
            self.softmax_w = tf.get_variable("softmax_w",
                                        [self.hidden_size, self.vocab_size])
            self.softmax_b = tf.get_variable("softmax_b",
                                        [self.vocab_size])
            # [(batch_sizrxvocab_size)]
            logits = [tf.matmul(outputStep, self.softmax_w) + self.softmax_b for outputStep in self.outputs]

            logits = tf.stack(logits, axis=1)
            probs = tf.nn.softmax(logits)

        tf.summary.histogram("probabilities", probs)
        with tf.variable_scope("loss"):
            self.loss = tf.contrib.seq2seq.sequence_loss(
                logits,
                targets,
                self.mask,
                average_across_timesteps=False,
                average_across_batch=True)
            # Update the cost
            self.cost = tf.reduce_sum(self.loss)

            self.total_cost = self.total_cost.assign_add(self.cost)
            self.total_len = self.total_len.assign_add(self.num_steps)
            self.perplexity = tf.exp(tf.div(self.total_cost, self.total_len))

            tf.summary.scalar('sum_batch_cost', self.cost)
            tf.summary.scalar('perplexity', self.perplexity)

            self.testOps = [self.cost, self.perplexity]

    # TODO: make it private an define a property
    def _train_op(self):
        """ train the model using Gradient Decent. """
        with tf.variable_scope("train_op"):
            self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0.0))

            # self.learning_rate = tf.Variable(0.0, trainable=False)
            self.initial_learning_rate = tf.constant(0.01)
            learning_rate = tf.train.exponential_decay(self.initial_learning_rate,
                                                       self.global_step, 500, 0.9)
            tf.summary.scalar("learning_rate", learning_rate)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5)
            optimizer = tf.train.AdamOptimizer(learning_rate)

            # Visualise gradients
            vis_grads = [0 if i is None else i for i in grads]
            for g in vis_grads:
                tf.summary.histogram("gradients_" + str(g), g)

            self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

            self.trainOps = [self.cost, self.train_op, self.global_step]

    # TODO: refacotr this and train to use session.run
    def _sample_op(self):
        """ define a sampling sub-graph. """
        # TODO: using varible_scope here cause sampling to output garbage. why?
        self.current_states = list()

        for i in range(0, self.num_layers):
            self.current_states.append(tuple([tf.placeholder(tf.float32,
                                       shape=[1, self.hidden_size],
                                       name="lstm_state_c_" + str(i)),
                tf.placeholder(tf.float32, shape=[1, self.hidden_size],
                               name="lstm_state_h_" + str(i))]))

        self.current_states = tuple(self.current_states)

        self.input = tf.placeholder(dtype=tf.int32, shape=[None],
                                    name="sample_input")

        # TODO: do we need to store/restor the state?
        embedding = tf.constant(np.eye(self.vocab_size), dtype=tf.float32)
        input_ = tf.nn.embedding_lookup(embedding, self.input)    # 1 x Vocab-size
        input_by_time = [input_]  # List of 1 x Vocab-size tens
        #with tf.variable_scope("sample_op"):
        outputs, self.state = \
            tf.contrib.rnn.static_rnn(self.cell, input_by_time,
                                      initial_state=self.current_states,
                                      scope='SSRNN')
        with tf.variable_scope("softmax", reuse=tf.AUTO_REUSE):
            self.softmax_w = tf.get_variable("softmax_w",
                                             [self.hidden_size, self.vocab_size])
            self.softmax_b = tf.get_variable("softmax_b",
                                             [self.vocab_size])
            logits = [tf.matmul(outputStep, self.softmax_w) + self.softmax_b for outputStep in outputs]
            logits = tf.stack(logits, axis=1)
            self.probs = tf.nn.softmax(logits)


    def sample(self, sess, ids_to_words):
        """ use graph defined in _sample_op to actually sample the model."""
        current_seq_ind = []
        iteration = 0
        initial_states = []
        for i in range(0, self.num_layers):
            initial_states.append(tuple([np.zeros(shape=[1, self.hidden_size], dtype=np.float32), np.zeros(shape=[1, self.hidden_size], dtype=np.float32)]))

        initial_states = tuple(initial_states)
        s = initial_states
        p = (1.0 / (self.vocab_size)) * np.ones(self.vocab_size)
        #self.is_training = False
        while iteration < 1000:
            # Now p contains probability of upcoming char, as estimated by model, and s the last RNN state
            ind_sample = np.random.choice(range(0, self.vocab_size), p=p.ravel())
            if ind_sample == 0 or ids_to_words[ind_sample] == "\n": # EOS token
                print("Model decided to stop generating!")
                break

            current_seq_ind.append(ind_sample)

            # Create feed dict for states
            feed = dict()
            feed[self.is_training] = False
            for i in range(0, self.num_layers):
                for c in range(0, len(s[i])):
                    feed[self.current_states[i][c]] = s[i][c]
                    feed[self.current_states[i][c]] = s[i][c]
                    pass
            feed[self.input] = [ind_sample]  # Add new input symbol to feed
            [p, s] = sess.run([self.probs, self.state], feed_dict=feed)

            iteration += 1

        out_str = ""
        for c in current_seq_ind:
            out_str += ids_to_words[c]
        print(out_str)


    def eval(self):
        pass


    @property
    def cost(self):
        return self._cost

    @cost.setter
    def cost(self, value):
        self._cost = value


def main():
    total = len(sys.argv)
    if total < 3:
        print("usage: python char-LM-using-batch-seq-with-state.py  train/test data_path.")
        return
    # Get the arguments list
    cmdargs = str(sys.argv)
    action = sys.argv[1]
    data_path = sys.argv[2]
    with tf.Session() as sess:
        checkpoints_dir = "./checkpoints/"
        logs_dir = "./logs"

        charlm_model = CharLM(batch_size=16,
                              hidden_size=300,
                              num_layers=1,
                              num_steps=10,
                              keep_prob=.8, # for now since sampling cant be done
                              data_path=data_path,
                              sess=sess,
                              max_itr=20000,
                              checkpoints_dir=checkpoints_dir,
                              logs_dir=logs_dir)

        coord = tf.train.Coordinator()
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        tf_threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        saver = tf.train.Saver(tf.global_variables(),
                               write_version=tf.train.SaverDef.V2)

        latestCheckpoint = tf.train.latest_checkpoint(checkpoints_dir)
        if latestCheckpoint is not None:
            restorer = tf.train.Saver(tf.global_variables(),
                                      write_version=tf.train.SaverDef.V2)
            restorer.restore(sess, latestCheckpoint)
            print('Pre-trained model restored')

        # call the train
        if action == "train":
            charlm_model.train(saver)
        elif action == "test":
            charlm_model.test()

if __name__ == "__main__":
    main()





