import os
from vanna.chromadb import ChromaDB_VectorStore
from vanna.google import GoogleGeminiChat
from vanna.flask import VannaFlaskApp

# Leer variables de entorno
ODBC_CONN_STR = os.getenv("ODBC_CONN_STR")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")

class MyVanna(ChromaDB_VectorStore, GoogleGeminiChat):
    def __init__(self):
        # Persistencia de ChromaDB en carpeta local
        ChromaDB_VectorStore.__init__(self, config={
            "persist_directory": "./chromadb_data",
            "chroma_server_api": "http://chromadb:8000"
        })
        # Configuración de Gemini
        GoogleGeminiChat.__init__(self, config={
            "api_key": GEMINI_API_KEY,
            "model": GEMINI_MODEL
        })

# Instanciar y conectar
vn = MyVanna()
vn.connect_to_mssql(odbc_conn_str=ODBC_CONN_STR)

# The information schema query may need some tweaking depending on your database. This is a good starting point.
df_information_schema = vn.run_sql("SELECT * FROM INFORMATION_SCHEMA.COLUMNS")

# This will break up the information schema into bite-sized chunks that can be referenced by the LLM
plan = vn.get_training_plan_generic(df_information_schema)
#plan

# If you like the plan, then uncomment this and run it to train
vn.train(plan=plan)

# Crear y ejecutar la app Flask
app = VannaFlaskApp(vn)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
