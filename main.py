from ddpg import DDPG as Agent
import gym

# Register gym environment. By specifying kwargs,
# you are able to choose which patient to simulate.
# patient_name must be 'adolescent#001' to 'adolescent#010',
# or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
from gym.envs.registration import register
register(
    id='simglucose-adolescent2-v0',
    #entry_point='simglucose.envs:T1DSimEnv',
    entry_point='simglucose.envs:T1DSimEnv',
    kwargs={'patient_name': 'adolescent#002'}
)

env = gym.make('simglucose-adolescent2-v0')


def loop(agent, env):
    observation = env.reset()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    agent.remember(observation[0], action[0], reward[0], done, next_state)
    agent.train()


# if __name__ == "__main__"
def main():
    agent = Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.shape[0])
    for patient_num in range(3):
        patient_id = 'adullt' + (patient_num + 1)
        patient_name = 'adult#' + '{:03}'.format(patient_num + 1)
        print(patient_name, patient_id)
        register(
            id='simglucose-' + patient_id + '-v0',
            #entry_point='simglucose.envs:T1DSimEnv',
            entry_point='simglucose.envs:T1DSimEnv',
            kwargs={'patient_name': patient_name}
        )
        env = gym.make('simglcose-' + patient_id + '-v0')
        print(patient_id)
        while True:
            loop(agent, env)
