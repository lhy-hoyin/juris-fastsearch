# Deisgn

To ensure the integrity of our results, we build our results ground-up. We start from the most basic unit, the legal cases. By startiing bottom-up from the legal cases, we can make sure that every single results that we show is supported by a legal case.

We source for past legal cases and add them to our files. When the application starts, the files are added to ChromaDB, our vector database. When the user makes a search, we would query out database and return the most relevent results. The vector database allows quick and revelent retrival. The database is able to do this by converting the query into a vector itself, then find other vectors nearby. These other vectors would be legal cases of high relevance.

We used Gradio to quickly come up with a simple interface for the user.

## Optimizations

### Implemented optimizations

As case files can be long, we summarise the contents of the case file before adding into the database. Through summarizing, we are able to better focus onto the key and important points brought up during the case. 

### Future optimizations

Rather than creating a new database collection each time the application loads, the collection can be persisted to reduce the loading time required to initialise.

Instead of using an algorithm-based summarizer, we can migrate to an LLM-based summariser to better capture the essense of the case file.

## Tech Stack
- [ChromaDB](https://github.com/chroma-core/chroma) - Vector Database
- [Sumy](https://github.com/miso-belica/sumy) - Text Summarizer
- [Gradio](https://www.gradio.app/) - User Interface