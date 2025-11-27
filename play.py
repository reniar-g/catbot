import argparse
from utility import play_game
from cat_env import make_env

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Play CatBot environment in freeplay mode.')
    parser.add_argument('--cat', 
                      choices=['mittens', 'batmeow', 'paotsin', 'peekaboo', 'squiddyboi', 'trainer',
                               'patrol', 'diagonal', 'adaptive', 'phases', 'chaos', 'hybrid'],
                      default='batmeow',
                      help='Type of cat to play against (default: batmeow)')
    
    args = parser.parse_args()
    
    env = make_env(cat_type=args.cat)
    
    # Start the game
    print(f"\nStarting game with {args.cat}!")
    print("Use arrow keys to move. Press Q to quit.")
    play_game(env)