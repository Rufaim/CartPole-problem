import datetime
import numpy as np
import tensorflow as tf
import gym
from ddpg import DeepDeterministicPolicyGradients
from replay_buffer import ReplayBuffer
from neural_nets import ActorNet, CriticNet
from exploration import OrnsteinUhlenbeckActionNoise

MINIBATCH_SIZE = 200
MAX_EPISODES = 1000
MAX_EP_STEPS = 1000

TAU = 0.001
ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.001
GRADIENT_MAX_NORM = 5
BUFFER_SIZE = 100000
DISCOUNT_FACTOR = 0.99
P_RAND_ACTION = 0.05
SEED = 42


kernel_init = tf.keras.initializers.glorot_normal(SEED)
environment = gym.make('Pendulum-v0')
state_size = environment.observation_space.shape[0]
action_size = environment.action_space.shape[0]
action_bound = environment.action_space.high[0]

CRITIC_NET_STRUCTURE = [tf.keras.layers.Dense(300,kernel_initializer=kernel_init),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.ReLU(),
                        tf.keras.layers.Dense(400,kernel_initializer=kernel_init,activation=tf.nn.relu),
                        tf.keras.layers.Dense(1,kernel_initializer=kernel_init)
                        ]
ACTOR_NET_STRUCTURE = [tf.keras.layers.Dense(300,kernel_initializer=kernel_init),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.ReLU(),
                        tf.keras.layers.Dense(400,kernel_initializer=kernel_init,activation=tf.nn.relu),
                        tf.keras.layers.Dense(action_size,kernel_initializer=kernel_init,activation=tf.nn.tanh)
                        ]


actor_net = ActorNet(ACTOR_NET_STRUCTURE,[action_bound],TAU,ACTOR_LEARNING_RATE)
critic_net = CriticNet(CRITIC_NET_STRUCTURE,TAU,CRITIC_LEARNING_RATE,GRADIENT_MAX_NORM)

action_noise = OrnsteinUhlenbeckActionNoise(np.zeros((action_size,)),0.2)

replay_buffer = ReplayBuffer(BUFFER_SIZE,SEED)
model = DeepDeterministicPolicyGradients(actor_net,critic_net,action_noise,\
                                         replay_buffer,action_size,state_size,DISCOUNT_FACTOR,MINIBATCH_SIZE)

logdir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(logdir)
file_writer.set_as_default()

for i in range(MAX_EPISODES):
    state = environment.reset()
    ep_reward = 0

    for j in range(MAX_EP_STEPS):
        a = model.actor_predict(state)
        a = a.reshape((-1,))
        next_state, r, t, _ = environment.step(a)
        model.add_to_buffer(np.squeeze(state), a, r, t, np.squeeze(next_state))

        model.update()

        state = next_state.copy()
        ep_reward += r
        if t:
            break
    tf.summary.scalar('Episode reward', data=ep_reward, step=i)
    file_writer.flush()
    print('Episode: {:d} | Reward: {:.2f} |'.format(i, ep_reward, i))
environment.close()