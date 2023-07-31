# IRC-GPT2-Chatbot
An extremely simple Python-based Internet Relay Chat (IRC) bot client for local GPT-2 TensorFlow models.

## Setup

1. Clone this repository to your local machine:
    ```
    git clone https://github.com/FlyingFathead/IRC-GPT2-Chatbot/
    cd IRC-GPT2-Chatbot/
    ```

4. Use `pip` to install the required packages:
    ```
    pip install -r requirements.txt
    ```

5. Get your local GPT-2 TensorFlow model files and place them underneath the main program folder (i.e. `./models/<yourmodelname>`)

    To get the model files, you can use i.e. the original OpenAI repo at: https://github.com/openai/gpt-2/

7. Edit the `config.txt`

    Set your model name to point out to a directory that holds them, i.e. put them inside a directory tree that corresponds to `./models/<yourmodelname>` in the `config.txt`, i.e. if your model name is `124M`, make sure its files are in `.models/124M/`

    You might want to configure the `input_prefix`, the `output_prefix` and `starting_context` to suit your needs.
    
    Set up your bot's IRC server and other variables from `config.txt`: `SERVER`, `PORT`, `NICKNAME`, `REALNAME`, `USERNAME`, `CHANNEL`

8. Run the bot:
    ```
    python ircbot.py
    ```

---

Enjoy!

Brought to you by [FlyingFathead](https://github.com/FlyingFathead/) with the kind whispering helps of ChaosWhisperer.
