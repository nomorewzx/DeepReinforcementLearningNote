import numpy as np
import tensorflow as tf
import cPickle as pickle
import pong_utils
import gym
import os

HIDDEN_LAYER_COUNTS = 200
LEARNING_RATE = 1e-4

REWARD_DISCOUNT_FACTOR = 0.99

RESUME = False
RENDER = True

INPUT_DIM = 80 * 80
ACTION_SPACE_SIZE = 2

MOVE_UP = 2
MOVE_DOWN = 3

graph = tf.Graph()
with graph.as_default():

    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    input_statuses = tf.placeholder(shape=[None, INPUT_DIM], dtype=tf.float32, name="input_status_by_episode")

    hidden_layer_weights = tf.Variable(tf.random_uniform([INPUT_DIM, HIDDEN_LAYER_COUNTS], dtype=tf.float32), name="hidden_layer_weights")
    output_layer_weights = tf.Variable(tf.random_uniform([HIDDEN_LAYER_COUNTS, ACTION_SPACE_SIZE], dtype=tf.float32), name="output_layer_weights")

    hidden_layer_output = tf.nn.relu(tf.matmul(input_statuses, hidden_layer_weights))

    sampled_policy_logit = tf.matmul(hidden_layer_output, output_layer_weights)

    action_probs = tf.nn.softmax(sampled_policy_logit)
    random_prob = tf.random_uniform(shape=[], maxval=1.0, minval=0.1)

    sampled_action = tf.cond(tf.less(action_probs[0][0], random_prob), lambda: tf.constant(shape=[2], value=[1, 0]),
                             lambda: tf.constant(shape=[2], value=[0, 1]))


    rewards = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="rewards_by_episode")
    input_actions = tf.placeholder(shape=[None, 2], dtype=tf.int32, name="input_actions_by_episode")

    hidden_layer_outputs = tf.nn.relu(tf.matmul(input_statuses, hidden_layer_weights))

    policy_logits = tf.matmul(hidden_layer_outputs, output_layer_weights)

    loss = tf.reduce_mean(
        tf.multiply(rewards, tf.nn.softmax_cross_entropy_with_logits(logits=policy_logits, labels=input_actions)))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(loss, global_step= global_step)

with tf.Session(graph=graph) as session:

    saver = tf.train.Saver(max_to_keep= 4)
    tf.global_variables_initializer().run()
    check_point = tf.train.get_checkpoint_state(os.path.dirname('saved_model/checkpoint'))
    print(check_point)
    if check_point and check_point.model_checkpoint_path:
        print("Find Checkpoint")
        saver.restore(session, check_point.model_checkpoint_path)
    pong_env = gym.make("Pong-v0")
    observation = pong_env.reset()
    prev_status = None
    running_reward = None
    reward_sum = 0
    episode_numer = 0
    game_numer = 0

    input_status_list = list()
    reward_list = list()
    sampled_action_list = list()

    while True:
        if RENDER:
            pong_env.render()
        current_status = pong_utils.preprocessing_observation_of_pong(observation)

        if prev_status is not None:
            status = current_status - prev_status
        else:
            status = np.zeros(INPUT_DIM)

        prev_status = current_status

        sampled_action_train, action_probs_train, random_prob_train = session.run([sampled_action, action_probs, random_prob], feed_dict={input_statuses: [status]})

        action = np.argmax(sampled_action_train) + 2

        observation, reward, done, info = pong_env.step(action)
        reward_sum += reward

        input_status_list.append(status)
        reward_list.append(reward)
        sampled_action_list.append(sampled_action_train)

        discounted_reward_list = pong_utils.discounted_rewards(reward_list=reward_list, gamma= REWARD_DISCOUNT_FACTOR).reshape([-1,1])
        if done:
            episode_numer += 1
            game_numer = 0
            feed_dict = {input_statuses: input_status_list, rewards: discounted_reward_list, input_actions: sampled_action_list}
            _, loss_train, global_step_train = session.run([optimizer, loss, global_step], feed_dict=feed_dict)
            print("Resetting env. episode reward total was %f . Loss is %f" % (reward_sum, loss_train))
            reward_sum = 0
            saver.save(session, 'saved_model/pong_model', global_step= global_step_train)
            observation = pong_env.reset()  # reset env
            prev_status = None

        if reward != 0:
            game_numer +=1
            print("Episode %d game %d finished, reward: %f" % (global_step.eval(), game_numer, reward) + (" " if reward == -1 else "!!!!!!!!!!"))