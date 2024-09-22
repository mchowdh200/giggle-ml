import faiss                   # make faiss available
import numpy as np

dim = 4                           # dimension
db_size = 100000                      # database size
query_amnt = 50                       # nb of queries
np.random.seed(1234)             # make reproducible

index = faiss.IndexFlatL2(dim)   # build the index
contents = np.random.random((db_size, dim)).astype('float32')
index.add(contents)              # add vectors to the index


def query_elements():
    # randomly sample from the database
    qidx = np.random.choice(db_size, query_amnt)
    # perform the search
    dist, idx = index.search(contents[qidx], 1)
    print('Sum dist', sum(dist))

    for i in range(query_amnt):
        if dist[i] != 0:
            print(qidx[i], '-->', dist[i], idx[i])


query_elements()
