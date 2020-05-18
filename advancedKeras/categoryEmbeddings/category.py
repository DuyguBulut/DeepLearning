# Imports
from keras.layers import Embedding
from numpy import unique

# Count the unique number of teams
n_teams = unique(games_season['team_1']).shape[0]

# Create an embedding layer
team_lookup = Embedding(input_dim=n_teams,
                        output_dim=1,
                        input_length=1,
                        name='Team-Strength')
                        
                        
 # Imports
from keras.layers import Input, Embedding, Flatten
from keras.models import Model
# Create an input layer for the team ID
teamid_in = Input(shape=(1,))
# Lookup the input in the team strength embedding layer
strength_lookup = team_lookup(teamid_in)
# Flatten the output
strength_lookup_flat = Flatten()(strength_lookup)
# Combine the operations into a single, re-usable model
team_strength_model = Model(teamid_in, strength_lookup_flat, name='Team-Strength-Model')


