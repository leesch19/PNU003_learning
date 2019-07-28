import tensorflow as tf
import DataHolder
import numpy as np
import modeling


def Fully_Connected(inp, output, name, activation, reuse=False):
    h = tf.contrib.layers.fully_connected(
        inputs=inp,
        num_outputs=output,
        activation_fn=activation,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=3e-7),
        biases_initializer=tf.constant_initializer(3e-7),
        scope=name,
        reuse=reuse
    )

    return h


def seq_length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.truncated_normal_initializer(stddev=initializer_range)


class LSTM_Model:
    def __init__(self):
        self.bert_path = 'C:\\Users\\USER\\Desktop\\bert_path\\bert_model.ckpt'

        self.data_holder = DataHolder.DataHolder()

        self.X = tf.placeholder(shape=[None, None], dtype=np.int32)
        self.X_mask = tf.placeholder(shape=[None, None], dtype=np.int32)

        self.Y = tf.placeholder(shape=[None, 2], dtype=np.int32)

        self.keep_prob = 0.8

    def create_model(self, input_ids, input_mask, is_training=True, reuse=False):
        bert_config = modeling.BertConfig.from_json_file('bert_config.json')

        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,  #is_traing -> true(bert model drop out 저절로 설정)
            input_ids=input_ids,
            input_mask=input_mask,
            scope='bert',
            reuse=reuse #설정 안해도 무관
        )

        bert_variables = tf.global_variables() #뉴럴 네트워크 모두 가지고 온다.

        return model, bert_variables

    def forward(self, model_output):
        model_output.set_shape(shape=[None, 768])

        """Get loss and log probs for the next sentence prediction."""

        with tf.variable_scope("pointer_net"):
            log_probs1 = Fully_Connected(model_output, output=2, name='pointer_symbol', activation=None,
                                         reuse=False)

            return log_probs1

    def training(self, epo, l2_norm=True, continue_training=False, first_training=False):
        #실험하는 컴퓨터 환경에 맞게 수정하세요
        save_path = 'C:\\Users\\USER\\Desktop\\rating_movies\\my_model'

        with tf.Session() as sess:
            model, bert_variables = self.create_model(self.X, self.X_mask)

            target_embedding = model.get_pooled_output() #get_pooled_output(맨 첫번째 vector만 return)
            prediction = self.forward(target_embedding)

            total_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=prediction)
            total_loss = tf.reduce_mean(total_loss)

            learning_rate = 0.001

            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

            sess.run(tf.initialize_all_variables())

            if first_training is True: #True면 model이 처음 생성된 것이다. False면 학습이 진행 된것을 불러온다.
                print('BERT multi-lang loaded')
                saver = tf.train.Saver(bert_variables)
                saver.restore(sess, self.bert_path)

            if continue_training is True:
                print('model restoring!')
                saver = tf.train.Saver()
                saver.restore(sess, save_path) #save_path에 새로운 폴더 저장

            for i in range(epo):
                sequence1, sequence_mask, label = self.data_holder.next_random_batch()
                training_feed_dict = {self.X: sequence1, self.X_mask: sequence_mask, self.Y: label}

                _, loss_value = sess.run([optimizer, total_loss], feed_dict=training_feed_dict)

                if i % 2 == 0:
                    print(i, loss_value)

                if i % 100 == 0:
                    saver = tf.train.Saver()
                    saver.save(sess, save_path)
                    print('saved!')
                    print('loss:', loss_value)

            saver = tf.train.Saver()
            saver.save(sess, save_path)

    def evaluation(self):
        # 실험하는 컴퓨터 환경에 맞게 수정하세요
        save_path = 'C:\\Users\\USER\\Desktop\\rating_movies\\my_model'

        with tf.Session() as sess:
            prediction = self.forward()
            prediction_idx = tf.argmax(prediction, axis=1)

            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver()
            saver.restore(sess, save_path)

            cor = 0
            cnt = 0

            for i in range(50):
            #while(True):
                check, sequence1, label = self.data_holder.next_test_batch()

                if check is False:
                    break

                training_feed_dict = {self.X: sequence1}

                result = sess.run(prediction_idx, feed_dict=training_feed_dict)

                for i in range(self.data_holder.Batch_Size):
                    if result[i] == label[i]:
                        cor += 1
                    cnt += 1

        print(cor, '/', cnt)