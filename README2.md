# run

1. install [Ollama](https://ollama.com/)
2. install LLM

```bash
ollama pull hf.co/second-state/E5-Mistral-7B-Instruct-Embedding-GGUF:Q8_0
```

3. install [uv](https://docs.astral.sh/uv/getting-started/installation/#installation-methods)

4. initialize and enter virtual environment 

```bash
uv venv .venv
source ~/.venv/bin/activate
```

5. run the program

```bash
./run.sh
```

# dataset
If you want to use your own data to build KG, create a new file and store your data (let say `example3.txt`). After that, change the variable `DATASET=example` in `run.sh`

# change to chatgpt api
Now, it is default to use deepseek-chat. If you want to use chatgpt, change the variable to `MODEL` to the model name you want. For example,

```
MODEL=gpt-3.5-turbo
```
