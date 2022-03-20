"""
Solving FrozenLake environment using Policy-Iteration.
"""
import numpy as np
import gym
from gym import wrappers
from gym.envs.registration import register

class PolicyIteration(object):
    def __init__(self, env, gamma) -> None:
        self.env = env
        self.gamma = gamma
        self.eps = 1e-10

        self.max_iterations = 200000
    
    def policy_iteration(self):
        # step1: 选取一个策略
        policy = np.random.choice(self.env.env.nA, size=(self.env.env.nS))
        for i in range(self.max_iterations):
            # step2: 根据选择的策略计算状态价值
            old_policy_value = self.compute_policy_value(policy=policy)
            # step3: 根据状态价值更新策略
            new_policy = self.update_policy(old_policy_value)
            if np.all(policy == new_policy):
                print("end at step " + str(i))
                break
            policy = new_policy
            print("iter " + str(i))
        # 返回结果 policy
        return policy
    
    def run_episode(self, env, policy, render=False):
        # reset()返回初始状态
        state = env.reset()
        total_reward = 0
        step_idx = 0
        while True:
            if render:
                env.render()
            state, reward, done, _ = env.step(policy[state])
            total_reward += (self.gamma ** step_idx * reward)
            step_idx += 1
            if done:
                break
        return total_reward
    
    # 计算策略的状态价值
    def compute_policy_value(self, policy):
        state_value = np.zeros(shape=self.env.env.nS)
        while True:
            old_value = np.copy(state_value)
            # 更新一次所有状态的价值
            for s in range(self.env.env.nS):
                action = policy[s]
                state_value[s] = sum([p * (r + self.gamma * old_value[s_next]) for p, s_next, r, _ in self.env.env.P[s][action]])
            if (np.sum((np.abs(old_value - state_value)))) <= self.eps:
                break
        return state_value
    
    def update_policy(self, old_policy_value):
        new_policy = np.zeros(self.env.env.nS)
        for s in range(self.env.env.nS):
            state_value = old_policy_value[s]

            # 计算每个状态下，每个动作的价值
            # 每个动作后的下一个状态是一个概率分布, 不是确定的
            q_sa = np.zeros(self.env.env.nA)
            for state_action in range(self.env.env.nA):
                q_sa[state_action] = sum(
                    [p * (r + self.gamma * old_policy_value[next_state]) for p, next_state, r, _ in self.env.env.P[s][state_action]]
                )
            # 新策略: 每个状态选择价值最大的动作
            new_policy[s] = np.argmax(q_sa)
        return new_policy
    
    def evaluate_policy(self, policy, n = 100):
        total_reward = [self.run_episode(self.env, policy) for _ in range(n)]
        return np.mean(total_reward)

if __name__ == '__main__':

    env_name  = 'FrozenLake-v0' # 'FrozenLake8x8-v0'
    env = gym.make(env_name)
    
    policy_iter = PolicyIteration(env, gamma=1.0)
    policy = policy_iter.policy_iteration()
    print(policy)
    scores = policy_iter.evaluate_policy(policy)
    print('Average scores = ', np.mean(scores))