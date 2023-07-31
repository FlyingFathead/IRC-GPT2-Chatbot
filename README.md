# IRC-GPT2-Chatbot
An extremely simple Python-based Internet Relay Chat bot client for local GPT-2 TensorFlow models

## Setup

1. Clone this repository to your local machine.

2. Use pip to install the required packages:
    ```
    pip install -r requirements.txt
    ```

3. Get your local GPT-2 TensorFlow model files and

4. Edit the `config.txt`

    Set your model name to point out to a directory that holds them, i.e. put them inside a directory tree that corresponds to `./models/<yourmodelname>` in the `config.txt`

    You might want to configure the `input_prefix`, the `output_prefix` and `starting_context` to suit your needs.
    
    Set up your bot's server, port, nickname, realname, username and channel from the `config.txt` as well.    

5. Run the bot:
    ```
    python ircbot.py
    ```
