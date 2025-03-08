<<<<<<< HEAD
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import sqlite3
import pandas as pd

conn = sqlite3.connect("schemes.db")
df = pd.read_sql_query("SELECT * FROM schemes", conn)

=======
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import sqlite3
import pandas as pd

conn = sqlite3.connect("schemes.db")
df = pd.read_sql_query("SELECT * FROM schemes", conn)

>>>>>>> d194048 (Initial commit)
print(df.shape)