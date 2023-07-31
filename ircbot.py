# IRC-GPT2-Chatbot
# by FlyingFathead & ChaosWhisperer | v0.12 | 08/2023
# https://github.com/FlyingFathead/IRC-GPT2-Chatbot/

# time & logging
import time
import logging

# irc bot libraries
import irc.client
import irc.events

# configparser
import configparser

# for fixing non-unicode inputs
from jaraco.stream import buffer

import json, os, string, sys, threading, random, model, sample, encoder, logging, time
import numpy as np
import tensorflow as tf
import re
import os
import random

# Read configuration file
config = configparser.ConfigParser()
config.read('config.txt')

# Assign config variables
new = config.getboolean('DEFAULT', 'new')
input_prefix = config.get('DEFAULT', 'input_prefix').strip('"')
output_prefix = config.get('DEFAULT', 'output_prefix').strip('"')
starting_context = config.get('DEFAULT', 'starting_context').strip('"')
debug = config.getboolean('DEFAULT', 'debug')
timeout = config.getint('DEFAULT', 'timeout')
top = config.getfloat('DEFAULT', 'top')
degree = config.getfloat('DEFAULT', 'degree')
mx = config.getfloat('DEFAULT', 'mx')
tok = config.getint('DEFAULT', 'tok')
learning = config.get('DEFAULT', 'learning').strip('"')
min_num_answers = config.getint('DEFAULT', 'min_num_answers')
max_num_answers = config.getint('DEFAULT', 'max_num_answers')
server = config.get('DEFAULT', 'SERVER').strip('"')
channel = config.get('DEFAULT', 'CHANNEL').strip('"')
nickname = config.get('DEFAULT', 'NICKNAME').strip('"')        
port = config.getint('DEFAULT', 'PORT')
realname = config.get('DEFAULT', 'REALNAME').strip('"')
username = config.get('DEFAULT', 'USERNAME').strip('"')

# Read the filter_list from the configuration file and convert it to a Python list
filter_list_str = config.get('DEFAULT', 'filter_list').strip()
filter_list = [word.strip() for word in filter_list_str.split(',')]

# (for multi-user mode) Initialize an empty dictionary for each user's context
user_contexts = {}
user_temperatures = {}

# Convert degree to a string
temps = str(degree)

# End settings
mode = True
learn = True
user = ""
cache = ""
running = False
temps = str(degree)
tpstring = str(top)

# turns
global turns
turns = []

# Global variable to store the current time
now = ""

# Define a custom logging formatter
class CustomFormatter(logging.Formatter):
    def format(self, record):
        now = time.strftime('%Y-%m-%d %H:%M:%S')  # Get the current time
        record.now = now  # Add the 'now' attribute to the log record
        return super().format(record)

# Configure logging with the custom formatter
logging_format = '[{now}][{levelname}] {message}'
logging.basicConfig(format=logging_format, style='{', level=logging.INFO)
logging.getLogger().handlers[0].formatter = CustomFormatter(logging_format, style='{')

# split messages that are too long
def split_message(message, max_length):
    return [message[i:i+max_length] for i in range(0, len(message), max_length)]

class Bot:
    def __init__(self, server, channel, nickname):
        self.reactor = irc.client.Reactor()
        # self.reactor.server().errors = 'ignore'  # Ignore encoding errors; treat inbound text as-is
        self.server = server
        self.channel = channel
        self.nickname = nickname

        # UTF-8 fixes
        irc.client.ServerConnection.buffer_class = buffer.LenientDecodingLineBuffer
        irc.client.ServerConnection.buffer_class.errors = 'replace'  # replace invalid bytes

    def connect(self):        
        global now  # Indicate that we want to use the global 'now' variable

        # Get the current time
        now = time.strftime('%Y-%m-%d %H:%M:%S')

        # UTF-8 fixes
        irc.client.ServerConnection.buffer_class = buffer.LenientDecodingLineBuffer
        irc.client.ServerConnection.buffer_class.errors = 'replace'  # replace invalid bytes

        # Print connection information
        logging.info(f"Connecting to IRC network: {self.server}, port {port}")
        logging.info(f"Nickname: {self.nickname}")
        logging.info(f"Username: {username}")
        logging.info(f"Realname: {realname}")

        try:
            self.connection = self.reactor.server().connect(self.server, port, self.nickname)
        except irc.client.ServerConnectionError as x:
            logging.error(f"Failed to connect to {self.server}: {x}")
            sys.exit(1)

        # Print successful connection
        logging.info(f"Connected to: {self.server}")

        self.connection.join(self.channel)

        # Print channel join information
        logging.info(f"Joining channel: {self.channel}")

        self.connection.add_global_handler("welcome", self.on_connect)
        self.connection.add_global_handler("pubmsg", self.on_pubmsg)
    
    def on_connect(self, connection, event):
        print("[INFO] Connected to server.")

    def on_pubmsg(self, connection, event):
        try:
            input_text = event.arguments[0]
            sender_username = event.source.nick  # Get the sender's username from the IRC event

            response = generate_response(input_text, sender_username)  # Pass the sender's username to the generate_response function

            username_replacer = config.getboolean('DEFAULT', 'username_replacer')

            if username_replacer:
                # Replace 'USERNAME' with the sender's username
                # Use a regular expression to replace 'USERNAME' followed by anything
                response = re.sub(r'USERNAME\b', sender_username, response)

            # Split the response into parts that do not exceed the maximum length
            response_parts = split_message(response, 400)  # 400 to leave some room for other parts of the IRC message

            # Send each part of the response separately
            for part in response_parts:
                self.connection.privmsg(self.channel, part)
        except UnicodeDecodeError:
            print("[WARN/ERROR] A message was received that could not be decoded. Skipping.")

    def start(self):
        self.connect()
        self.reactor.process_forever()

# model interaction
def interact_model(bot, input_text, new):

    # initialize response
    response = ""

    # Read the model name from the configuration file
    model_name = config.get('DEFAULT', 'model_name').strip('"')

    # seed = random.randint(1431655765, 2863311530)
    seed = int(time.time())

    nsamples = 1
    batch_size = 1
    top_k = tok
    topp = top
    models_dir = 'models'
    tex = str(input_text)

    global learning
    global learn
    global mode
    global cache
    global turns  # Keep track of conversation turns

    # Add the starting context
    if new:  # If this is a new conversation, reset the list of turns
        turns = [starting_context]

    num_answers = random.randint(min_num_answers, max_num_answers)  # Randomize the number of answers

    enc = encoder.get_encoder(model_name, models_dir)

    if mode:
        if new:  # If this is a new conversation, reset the list of turns
            turns = []

        # Check total token count of the history plus the new user input
        potential_context = ''.join(turns) + input_prefix + tex + '\n' + output_prefix
        total_tokens = len(enc.encode(potential_context))

        # If too many tokens, remove turns from the start
        while total_tokens > 800:
            if len(turns) == 1:
                print("Cannot reduce the text further!")
                return
            turns.pop(0)
            potential_context = ''.join(turns) + input_prefix + tex + '\n' + output_prefix
            total_tokens = len(enc.encode(potential_context))

        # Add the user's input to the context after it's guaranteed to fit
        turns.append(input_prefix + tex + '\n' + output_prefix)

        raw_text = potential_context
        context_tokens = enc.encode(raw_text)
        length = 300  # Set the default length

    toppf = float(topp)
    lengthm = float(len(enc.encode(raw_text)))
    multf = float(mx)
    lxm = float(lengthm * multf)
    top_p = lxm + toppf

    # The max here is 0.84 and minimum 0.005
    if top_p > 0.84:
        top_p = 0.84
    if top_p < 0.010:
        top_p = 0.010

    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))
    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        context = tf.compat.v1.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.compat.v1.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=degree, top_k=top_k, top_p=top_p
        )

        saver = tf.compat.v1.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        generated = 0
        while generated < num_answers:
            out = sess.run(output, feed_dict={
                context: [context_tokens for _ in range(batch_size)]
            })[:, len(context_tokens):]
            
            for i in range(batch_size):
                generated += 1
                text = enc.decode(out[i])
                # Split the generated text on newline characters and only keep the first part
                text = text.split('\n')[0]

                # Remove carriage return characters from the text
                text = text.replace("\r", "")

                # Append the first part of the generated text to the response
                response += text

                # Check if a newline character is present in the generated text
                if '\n' in text:
                    break  # Stop generating the response if a newline character is encountered

                # Rest of the code                
                if debug:
                    print("==========")
                    print("Raw output: " + text)
                    print("==========")

                lines = text.splitlines()
                splitted = lines[0] if lines else ""

                turns.append(splitted + '\n')  # Append the bot's response to the turns list
                encodedstr = splitted.encode(encoding=sys.stdout.encoding, errors='ignore')
                decodedstr = encodedstr.decode("utf-8")
                final = str(decodedstr)
                finalsan = final
                finalsan = finalsan.lstrip()
                
                learning = raw_text + finalsan + " "
                
                modes = str(mode)
                print("Chatbot mode: " + modes)
                learns = str(learn)
                print("Learning mode: " + learns)
                lengths = str(length)
                print("Length: " + lengths)
                print("==========")
                splits = str(splitted)
                print("Before regex: " + splits)
                print("==========")
                print("Output: " + finalsan)
                print("==========")
                print("Raw_text or Original: " + raw_text)
                print("==========")
                print("Learning text or Next: " + learning)
                print("==========")
                tps = str(top_p)
                print("Final top_p: " + tps)
                print("==========")
                print("top_p in: " + tpstring)
                print("==========")

                return finalsan

# Response generation
def generate_response(input_text, sender_username):
    global new  # Indicate that we are using the global 'new' variable

    # Read the mention probability from the configuration file
    mention_prob = config.getfloat('DEFAULT', 'mention_prob')

    # Read the force lowercase option from the configuration file
    force_lowercase = config.getboolean('DEFAULT', 'force_lowercase')

    # Generate the response from the model
    response = interact_model(bot, input_text, new)

    # Check if the bot's nickname is present at the beginning of the response
    bot_nickname_colon = nickname + ':'
    if response.startswith(bot_nickname_colon):
        # If the bot's nickname is found at the beginning, remove it before mentioning the user
        response = response[len(bot_nickname_colon):]

    # Mention the user with the given probability
    mention_user = random.random() < mention_prob

    if mention_user:
        # Check if the response is empty or starts with a newline, in which case, do not add extra newline
        if response and not response.startswith('\n'):
            response = '\n' + response

        # Format the usermention reply appropriately based on force_lowercase
        if force_lowercase:
            # Convert the sender_username to lowercase
            # sender_username = sender_username.lower()
            response = response.lower()
            response = f"{sender_username}: {response.lstrip()}"
    else:
        # If the response does not start with the bot's nickname or if mention_user is False,
        # it is considered as generated text and should not be modified.
        # In this case, if force_lowercase is True, convert the response to lowercase.
        if force_lowercase:
            response = response.lower()

    # Remove newline characters from the response, convert to space
    response = response.replace("\n", " ")

    # Check if the response contains any word from the filter list
    contains_disallowed_word = any(word in response.lower() for word in filter_list)

    if contains_disallowed_word:
        # Regenerate the response until it doesn't contain any word from the filter list
        while contains_disallowed_word:
            response = interact_model(bot, input_text, new)
            contains_disallowed_word = any(word in response.lower() for word in filter_list)

    new = False  # Set 'new' to False after the first call
    return response

if __name__ == "__main__":
    bot = Bot(server, channel, nickname)
    bot.start()
