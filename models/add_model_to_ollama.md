### Run GGUF on Ollama

1. Create Modelfile
```
nano Modelfile
```

2. Put correct template in it (depends on chat or completion model)
Tip look on Ollama website for templates. 

Ollama Modelfile docs:
https://github.com/ollama/ollama/blob/main/docs/modelfile.md

Meta llama3 template:
https://ollama.com/library/llama3/blobs/8ab4849b038c

Example modelfile for llama-3.0:
```
FROM pg_chat.Q4_0.gguf

TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>
{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}
<|start_header_id|>user<|end_header_id|>
{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>
{{ .Response }}<|eot_id|>
"""

PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|reserved_special_token"
```


3. Create model
``` 
ollama create my-model -f Modelfile
```
