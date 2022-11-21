from Myworld import Environment
from ARG_FILE import args
from Agent import Agent

def main():
    env = Environment(args.node_info, args.line_info, args.path_info, args.send_path, 5)
    agent = Agent(env)
    agent.train(max_episodes=1000)

if __name__ == "__main__":
    main()