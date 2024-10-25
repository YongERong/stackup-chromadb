# Passion Weaver
A ChromaDB-powered chatbot which helps youth find events related to their interests.

## Technical Structure of Project
![Technical Structure Diagram](https://github.com/YongERong/stackup-chromadb/blob/main/Stackup-ChromaDB-Technical.png?raw=True)

Event tags with their descriptions were converted into embeddings and stored into a collection. 
A scraper module created using ScrapeGraphAI was used to traverse a list of links to event pages and deliver a Pydantic object with the event details.
These event details were converted into embeddings and compared with the tag embeddings to tag them. They were then saved into another collection.
The user converses with a chatbot with conversational memory which retrives events from the collection based on the conversation.

## Project WireFrame
![Wireframe](https://github.com/YongERong/stackup-chromadb/blob/main/assets/Stackup-ChromaDB-Wireframe.png?raw=True)

## Project Icon Design
![Icon Design](https://github.com/YongERong/stackup-chromadb/blob/main/assets/Stackup-ChromaDB-Icon.png?raw=True)

Attempted to make a play on a compass mixed with the colour palette of a passion flower.


## Things to Check Out
- [ ] Mockups and visualisations of vector database at mockups/main.ipynb
![Tags Vectors Visualisations](https://github.com/YongERong/stackup-chromadb/blob/main/assets/Tags_PCA.gif?raw=True)
- [ ] Try out the chatbot:
```
# 1. Install dependencies
pip3 install python-dotenv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Insert Gemini API Key
cp .env.example .env
# Insert Gemini API key into .env file

# 3. Run main.py file and open gradio ui in browser
python main.py
```


## Future Improvements
- [ ] Attach to Telegram API
- [ ] Better scraper module (current module has issues extracting descriptions of the events)
