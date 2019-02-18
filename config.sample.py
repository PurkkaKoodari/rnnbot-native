# telegram bot api token, from @botfather
TOKEN = ""

# unix timestamp to start counting commits from, in utc
COMMIT_EPOCH = 0
# how often to commit, in seconds
COMMIT_INTERVAL = 24 * 60 * 60

# how many recent messages (at most) to put in a commit
COMMIT_MESSAGES = 4000

# how many chars a commit must contain in order to not be skipped
# must be set to at least sequence_length + 1 (from rnn.c) to avoid crashes
COMMIT_MIN_LENGTH = 200

# how often to include the message sender's name in the message committed
# too high values just get the bot to spam variations of the names,
# which can get boring quickly
NAME_CHANCE = 1/3

# how long a message must be in chars in order to be committed
MESSAGE_MIN_LENGTH = 12
# how many words a message must have in order to be committed
MESSAGE_MIN_WORDS = 3

# how many neurons to have in the hidden layer
HIDDEN_SIZE = 200

# the group/user ids the bot can be used in
# all messages from these are collected and /rnn commands work in these
GROUPS = {-12345678, 12345678}
# the user ids that can use /rnncommit
ADMINS = {12345678}

# the format to output logs in
LOG_FORMAT = "%(asctime)s [%(name)s] [%(levelname)s] %(message)s"

# the message to use for /rnn when no data is available
NO_DATA = "No messages in the dataset."

# the message to use for /rnniter when data is available
ITER_MESSAGE = """<b>Iteration:</b> {iteration}
<b>Loss:</b> {loss}
<b>Last commit:</b> {from_last} ago
<b>Messages committed:</b> {in_last}
<b>Next commit:</b> in {to_next}
<b>Messages collected:</b> {in_next}"""

# the message to use for /rnniter when no data is available
ITER_NO_DATA = """No messages in the dataset.
<b>Next commit:</b> in {to_next}
<b>Messages collected:</b> {in_next}"""
