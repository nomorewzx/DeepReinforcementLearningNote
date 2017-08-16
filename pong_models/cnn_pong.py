import tensorflow as tf
import numpy as np
import pong_utils
import gym
import os

INPUT_FEATURE_DIM = 6400
FC_NN_HIDDEN_COUNTS = 200
ACTION_SPACE_SIZE = 3
LEARNING_RATE = 1e-3
DECAY = 0.99
ACTION_SPACE = [1,2,3]
SUMMARY_DIR = "../summary/cnn_pong"
CHECK_POINT_DIR = "../saved_model_cnn_pong/"
RENDER = True

graph = tf.Graph()

with graph.as_default():
    input_list = tf.placeholder(shape=[None, INPUT_FEATURE_DIM], dtype=tf.float32)
    discounted_rewards = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    labels = tf.placeholder(shape=[None, ACTION_SPACE_SIZE], dtype=tf.float32)
    rewards = tf.placeholder(shape=[None], dtype=tf.float32)

    global_step = tf.Variable(0, dtype=tf.int32, trainable= False, name="global_step")

    with tf.variable_scope("cnn_layer1") as scope:
        images = tf.reshape(input_list, shape=[-1, 80, 80, 1])
        filters = tf.get_variable("weights", shape=[5,5,1,10], initializer= tf.contrib.layers.xavier_initializer_conv2d())

        biases = tf.get_variable("biases", shape= [10], initializer= tf.random_normal_initializer())

        logits = tf.nn.conv2d(images, filters, strides=[1,1,1,1],
                              padding='SAME', name="cnn_layer_1_conv")

        activated_logits = tf.nn.relu(logits + biases, name= scope.name)

        conv1_output = tf.nn.max_pool(activated_logits, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    with tf.variable_scope("cnn_layer2") as scope_layer_2:
        weights = tf.get_variable("weights", shape=[5,5,10,10], dtype=tf.float32,
                                  initializer= tf.contrib.layers.xavier_initializer_conv2d())

        biases = tf.get_variable("biases", shape= [10], initializer= tf.random_normal_initializer())

        logits = tf.nn.conv2d(conv1_output, weights, strides=[1,1,1,1],
                              padding="SAME", name="cnn_layer2_conv")

        activated_logits = tf.nn.relu(logits + biases)

        conv2_output = tf.nn.max_pool(activated_logits, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

    with tf.variable_scope("fc_nn_hidden_layer") as fc_nn_hidden:
        input_features_dim = 20 * 20 * 10

        weights = tf.get_variable("weights", shape=[input_features_dim, FC_NN_HIDDEN_COUNTS],
                                  initializer= tf.truncated_normal_initializer(mean= 0.0, stddev= 1/np.sqrt(input_features_dim)))

        biases = tf.get_variable("biases", shape=[FC_NN_HIDDEN_COUNTS],
                                 initializer= tf.zeros_initializer())

        flatten_conv2_output = tf.reshape(conv2_output, shape=[-1, input_features_dim])

        hidden_layer_relu = tf.nn.relu(tf.matmul(flatten_conv2_output, weights) + biases)

        hidden_layer_output = tf.nn.dropout(hidden_layer_relu, keep_prob= 0.75, name="drop_out")

    with tf.variable_scope("fc_nn_output_layer") as fc_nn_output_layer:
        weights = tf.get_variable("weights", shape=[FC_NN_HIDDEN_COUNTS, ACTION_SPACE_SIZE],
                                  initializer= tf.truncated_normal_initializer(mean= 0.0, stddev= 1/np.sqrt(FC_NN_HIDDEN_COUNTS)))

        biases = tf.get_variable("biases", shape=[ACTION_SPACE_SIZE],
                                 initializer= tf.zeros_initializer())

        logits = tf.matmul(hidden_layer_output, weights) + biases

        action_probs = tf.nn.softmax(logits)

    with tf.variable_scope("loss") as loss_scope:
        loss = tf.reduce_mean(-discounted_rewards *
                             labels *
                             tf.log(tf.clip_by_value(action_probs, clip_value_min=1e-40, clip_value_max= 1.0)))

        optimizer = tf.train.RMSPropOptimizer(learning_rate= LEARNING_RATE, decay= DECAY).minimize(loss, global_step= global_step)

    with tf.variable_scope("summary") as scope:
        loss_summary = tf.summary.scalar("loss", loss)
        reward_summary = tf.summary.scalar("reward", tf.reduce_sum(rewards))

        summary_op = tf.summary.merge_all()

with tf.Session(graph= graph) as session:

    tf.global_variables_initializer().run()
    writer = tf.summary.FileWriter(SUMMARY_DIR, graph=session.graph)
    saver = tf.train.Saver(keep_checkpoint_every_n_hours= 1)

    check_point = tf.train.get_checkpoint_state(CHECK_POINT_DIR)
    if check_point and check_point.model_checkpoint_path:
        saver.restore(session, check_point.model_checkpoint_path)

    pong_env = gym.make("Pong-v0")

    total_reward = 0
    observation = pong_env.reset()
    prev_status = None
    one_hot_action_list = list()
    reward_list = list()
    status_list = list()
    game_numer = 0

    while True:
        if RENDER:
            pong_env.render()
        current_status = pong_utils.preprocessing_and_flatten_observation(observation)

        if prev_status is not None:
            status = current_status - prev_status
        else:
            status = np.zeros_like(current_status)

        prev_status = current_status

        feed_dict = {input_list: [status]}

        action_probs_train = session.run(action_probs, feed_dict=feed_dict)

        action_index = pong_utils.poll_action_index_from_action_probs(action_probs_train[0])

        action = ACTION_SPACE[action_index]

        one_hot_action = np.zeros_like(action_probs_train[0])

        one_hot_action[action_index] = 1
        one_hot_action_list.append(one_hot_action)

        observation, reward, done, info = pong_env.step(action)
        total_reward += reward

        reward_list.append(reward)
        status_list.append(status)

        if done:
            discounted_reward_list = pong_utils.discounted_rewards(reward_list, 0.99)

            ep_discounted_reward_list = np.vstack(discounted_reward_list)
            ep_status_list = np.vstack(status_list)
            ep_one_hot_action_list = np.vstack(one_hot_action_list)

            feed_dict = { input_list: ep_status_list, labels: ep_one_hot_action_list,
                     discounted_rewards: ep_discounted_reward_list, rewards: reward_list }

            _, loss_train, global_step_train, summary_op_train = session.run([optimizer, loss, global_step, summary_op], feed_dict=feed_dict)

            print("Episode %d finished, reward: %d" % (global_step_train, total_reward))
            saver.save(session, CHECK_POINT_DIR + 'pong_model', global_step=global_step_train)
            writer.add_summary(summary_op_train, global_step=global_step_train)

            status_list = []
            one_hot_action_list = []
            reward_list = []
            observation = pong_env.reset()
            prev_status = None
            total_reward = 0
            game_numer = 0

        if reward != 0:
            print("Episode %d Game %d finished, reward: %f" % (global_step.eval(), game_numer, reward) + (" " if reward == -1 else "!!!!!!!!!!"))
            game_numer += 1