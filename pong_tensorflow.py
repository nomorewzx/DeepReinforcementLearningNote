"""
This model will take around 5300 episodes to compete with its opponent. And I trained this model on Mac Pro 15 Inch (16G RAM, Core i7) about 12 hours.
"""
import numpy as np
import tensorflow as tf
import cPickle as pickle
import pong_utils
import gym
import os

HIDDEN_LAYER_COUNTS = 200
LEARNING_RATE = 1e-3

REWARD_DISCOUNT_FACTOR = 0.99

RESUME = False
RENDER = True

INPUT_DIM = 80 * 80
ACTION_SPACE_SIZE = 3

STILL = 1
MOVE_UP = 2
MOVE_DOWN = 3
DECAY = 0.99

ACTION_SPACE = [STILL, MOVE_UP, MOVE_DOWN]

graph = tf.Graph()
with graph.as_default():

    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    input_statuses = tf.placeholder(shape=[None, INPUT_DIM], dtype=tf.float32, name="input_status_by_episode")
    labels = tf.placeholder(shape=[None, ACTION_SPACE_SIZE], dtype=tf.float32, name="labels")

    hidden_layer_weights = tf.Variable(tf.truncated_normal([INPUT_DIM, HIDDEN_LAYER_COUNTS],mean=0.0, stddev= 1. / np.sqrt(INPUT_DIM), dtype=tf.float32), name="hidden_layer_weights")
    output_layer_weights = tf.Variable(tf.truncated_normal([HIDDEN_LAYER_COUNTS, ACTION_SPACE_SIZE], mean=0.0, stddev= 1. / np.sqrt(HIDDEN_LAYER_COUNTS), dtype=tf.float32), name="output_layer_weights")

    hidden_layer_output = tf.nn.relu(tf.matmul(input_statuses, hidden_layer_weights))

    sampled_policy_logit = tf.matmul(hidden_layer_output, output_layer_weights)

    action_probs = tf.nn.softmax(sampled_policy_logit)

    rewards = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="rewards_by_episode")

    hidden_layer_outputs = tf.nn.relu(tf.matmul(input_statuses, hidden_layer_weights))

    logits = tf.matmul(hidden_layer_outputs, output_layer_weights)
    softmax_logits = tf.nn.softmax(logits= logits)

    loss = tf.nn.l2_loss(labels - softmax_logits)

    optimizer = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE, decay= DECAY)

    gradients = optimizer.compute_gradients(loss= loss, grad_loss= rewards)
    train_op = optimizer.apply_gradients(gradients, global_step= global_step)

with tf.Session(graph=graph) as session:

    saver = tf.train.Saver(max_to_keep= 4)
    tf.global_variables_initializer().run()
    check_point = tf.train.get_checkpoint_state(os.path.dirname('saved_model/checkpoint'))
    if check_point and check_point.model_checkpoint_path:
        saver.restore(session, check_point.model_checkpoint_path)
    pong_env = gym.make("Pong-v0")
    observation = pong_env.reset()
    prev_status = None
    running_reward = None
    reward_sum = 0
    episode_numer = 0
    game_numer = 0

    input_status_list = list()
    one_hot_action_list = list()
    reward_list = list()

    while True:
        if RENDER:
            pong_env.render()
        current_status = pong_utils.preprocessing_observation_of_pong(observation)

        if prev_status is not None:
            status = current_status - prev_status
        else:
            status = np.zeros(INPUT_DIM)

        prev_status = current_status
        input_status_list.append(status)

        action_probs_train = session.run(action_probs, feed_dict={input_statuses: [status]})
        action_probs_train = action_probs_train[0, :]
        action_index = np.random.choice(ACTION_SPACE_SIZE, p= action_probs_train)

        one_hot_action = np.zeros_like(action_probs_train)
        one_hot_action[action_index] = 1

        one_hot_action_list.append(one_hot_action)

        observation, reward, done, info = pong_env.step(ACTION_SPACE[action_index])

        reward_sum += reward

        reward_list.append(reward)

        discounted_reward_list = pong_utils.discounted_rewards(reward_list=reward_list, gamma= REWARD_DISCOUNT_FACTOR)

        if done:

            episode_numer += 1
            game_numer = 0

            episode_rewards = np.vstack(discounted_reward_list)
            episode_labels = np.array(one_hot_action_list, dtype=np.float32)
            episode_statuses = np.array(input_status_list, dtype=np.float32)

            feed_dict = {input_statuses: input_status_list, rewards: episode_rewards, labels: episode_labels}

            _, loss_train, logits_train, global_step_train = session.run([train_op, loss, logits, global_step], feed_dict=feed_dict)
            print("Resetting env. episode reward total was %f . Loss is %f" % (reward_sum, loss_train))
            reward_sum = 0
            saver.save(session, 'saved_model/pong_model', global_step= global_step_train)
            observation = pong_env.reset()  # reset env
            prev_status = None

            input_status_list = []
            reward_list = []
            one_hot_action_list = []

        if reward != 0:
            game_numer +=1
            print("Episode %d game %d finished, reward: %f" % (global_step.eval(), game_numer, reward) + (" " if reward == -1 else "!!!!!!!!!!"))