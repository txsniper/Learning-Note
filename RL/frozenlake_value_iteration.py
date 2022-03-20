"""
Solving FrozenLake environment using Policy-Iteration.
"""
from functools import total_ordering
import numpy as np
import gym
from gym import wrappers
from gym.envs.registration import register

class ValueIteration(object):
    def __init__(self, env, gamma) -> None:
        self.env = env
        self.gamma = gamma
        pass
    
    def extract_policy(self, state_value):
        policy = np.zeros(self.env.env.nS)
        for s in range(self.env.env.nS):
            # 计算每个状态的动作价值
            q_sa = np.zeros(self.env.nA)
            for action in range(self.env.env.nA):
                # 每个动作后的下一个状态是一个概率分布
                for next_sr in self.env.env.P[s][action]:
                    p, next_state, reward, _ = next_sr
                    q_sa[action] += (p * (reward + self.gamma * state_value[next_state]))
            # 获取当前状态下的最优动作
            policy[s] = np.argmax(q_sa)
        return policy

    def value_iteration(self):
        # 初始化状态价值
        state_value = np.zeros(env.env.nS)
        max_iterations = 100000
        eps = 1e-20

        for i in range(max_iterations):
            old_state_value = np.copy(state_value)
            # 更新每个状态的价值
            for s in range(self.env.env.nS):
                # 计算每个状态的所有动作价值
                q_sa = [
                    sum(
                        [p * (r + old_state_value[next_state] * self.gamma) for p, next_state, r, _ in self.env.env.P[s][action]]
                    ) for action in range(self.env.env.nA)
                ]
                # 选取最大的动作价值作为状态价值
                # 在这里应该已经知道当前最优的动作了
                state_value[s] = max(q_sa)
            if (np.sum(np.fabs(old_state_value - state_value)) <= eps):
                print('value iter at ' + str(i))
                break
        return state_value

    def run_episode(self, env, policy, render=False):
        state = env.reset()
        total_reward = 0
        step_idx = 0
        while True:
            if render:
                env.render()
            state, reward, done, _ = env.step(int(policy[state]))
            total_reward += (self.gamma ** step_idx * reward)
            step_idx += 1
            if done:
                break
        return total_reward

    def evaluate_policy(self, policy, n = 100):
        scores = [self.run_episode(self.env, policy) for _ in range(n)]
        return np.mean(scores)

if __name__ == '__main__':

    env_name  = 'FrozenLake-v0' # 'FrozenLake8x8-v0'
    env = gym.make(env_name)
    gamma = 1.0
    value_iter = ValueIteration(env, gamma=gamma)
    state_value = value_iter.value_iteration()
    policy = value_iter.extract_policy(state_value=state_value)
    policy_score = value_iter.evaluate_policy(policy)
    # optimal_v = value_iteration(env, gamma);
    # policy = extract_policy(optimal_v, gamma)
    # policy_score = evaluate_policy(env, policy, gamma, n=1000)
    print('Policy average score = ', policy_score)