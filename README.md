# IRC-GPT2-Chatbot
An extremely simple Python-based Internet Relay Chat bot client for local GPT-2 TensorFlow models

## Setup

1. Clone this repository to your local machine:
    ```
    git clone https://github.com/FlyingFathead/IRC-GPT2-Chatbot/
    cd IRC-GPT2-Chatbot/
    ```

4. Use pip to install the required packages:
    ```
    pip install -r requirements.txt
    ```

5. Get your local GPT-2 TensorFlow model files and place them underneat the main program folder 

6. Edit the `config.txt`

    Set your model name to point out to a directory that holds them, i.e. put them inside a directory tree that corresponds to `./models/<yourmodelname>` in the `config.txt`

    You might want to configure the `input_prefix`, the `output_prefix` and `starting_context` to suit your needs.
    
    Set up your bot's server, port, nickname, realname, username and channel from the `config.txt` as well.    

7. Run the bot:
    ```
    python ircbot.py
    ```
