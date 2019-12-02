DATA_COLUMNS = ['player_name', 'game_count', 'monetary_gain', 'num_wins']
DATA_FILE_PREFIX = 'pdb.'
HOLDEM_MONTHS = '01 02 03 04 05 06 07 08 09 10 11 12'.split() 
HOLDEM_YEARS = '1995 1996 1997 1998 1999 2000 2001'.split()
PLAYER_COLUMNS = ['player_name', 'game_id', 'num_players', 'position_played',
                  'pre_flop_actions', 'pre_turn_actions', 'pre_river_actions',
                  'showdown_actions', 'intial_stack', 'pot_input_amount',
                  'pot_winnings_amount', 'card1', 'card2']
HOLDEM_FOLDER = ''
TEST_HOLDEM_FOLDER = ''

try:
    from settings_local import *
except ImportError:
    pass
