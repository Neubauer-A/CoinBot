def compute_avg_return(environment, policy, num_episodes=10, p=False):
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return
        if p:
            print(episode_return.numpy()[0])

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]