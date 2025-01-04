import numpy as np
from environment import Environment
from agent import Agent
import matplotlib.pyplot as plt

def train_agent():
    env = Environment()
    state_size = env.get_state().shape[0]
    action_size = env.num_actions
    agent = Agent(state_size, action_size)
    
    episodes = 1000
    target_update_frequency = 10
    rewards_history = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            
            state = next_state
            total_reward += reward
            steps += 1
            
        if episode % target_update_frequency == 0:
            agent.update_target_network()
            
        if episode % 100 == 0:
            agent.save(f"dqn_agent_episode_{episode}.pth")
            
        rewards_history.append(total_reward)
        
        print(f"Episode: {episode + 1}/{episodes}")
        print(f"Steps: {steps}")
        print(f"Total reward: {total_reward}")
        print(f"Epsilon: {agent.epsilon}")
        print("--------------------")
        
    plt.plot(rewards_history)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.savefig('training_progress.png')
    plt.show()
    
    return agent

if __name__ == "__main__":
    trained_agent = train_agent()
    trained_agent.save("final_dqn_agent.pth")