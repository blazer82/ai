require 'rltorch'
--require 'alewrap'

env = rltorch.MountainCar_v0()

policy = rltorch.RandomPolicy(env.observation_space,env.action_space)

local initial_observation = env:reset()
policy:new_episode(env:reset())

local total_reward=0 
while(true) do
  --env:render{mode="console"}      
  env:render{mode="qt", fps=30}      

  local action=policy:sample()  -- sample one action based on the policy
  local observation,reward,done,info=unpack(env:step(action))  -- apply the action
  policy:feedback(reward) -- transmit the (immediate) reward to the policy
  policy:observe(observation)      
  total_reward=total_reward+reward -- update the total reward for this trajectory

  --- Leave the loop if the environment is in a final state
  if (done) then
    break
  end
end

policy:end_episode(total_reward) -- Transmit the total trajectory reward to the policy
env:close()