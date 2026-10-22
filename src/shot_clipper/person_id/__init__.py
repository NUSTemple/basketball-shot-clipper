"""Who scored each goal - see docs/PLAYER_IDENTIFICATION.md.

shooter.py finds the shooter in a goal clip, reid.py embeds them (person
re-identification), grouping.py groups a session's goals by shooter and
suggests a name per group, scan.py (shot-clipper-players) runs it over a
clips folder. A person confirms each group's name once in the label UI's
Players tab; that's stored as the clips' `scorer` like a hand-picked one.

Kept import-light: the label UI only needs PLAYERS_FILENAME, and its hosted
image has no torch.
"""
PLAYERS_FILENAME = "_players.json"
