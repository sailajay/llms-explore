To caluclate PPL with GPT2 and WikiText2 
1) pip install -r requirements.txt 

2) Start the local inference server
 /usr/bin/python3 /usr/local/bin/uvicorn server:app --host 0.0.0.0 --port 8000 --reload &

3)python3 perplexity_calculate.py 
