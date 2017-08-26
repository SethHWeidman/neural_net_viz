from pymongo import MongoClient

def find_all_docs():
    """Find all the documents in the Mongo database."""
    client = MongoClient()
    db = client.viz_db
    collection_name = "viz_coll"
    for document in db[collection_name].find():
        print(document)
    client.close()
    return None

def find_key(key):
    """Find a document with a specific key in the Mongo database."""
    client = MongoClient()
    db = client.viz_db
    collection_name = "viz_coll"
    output = None
    for document in db[collection_name].find({"viz_key": key}):
        output = document['value']
    client.close()
    return output

def add_key(key, value):
    """Add a new key-value pair to the Mongo database."""
    client = MongoClient()
    db = client.viz_db
    collection_name = "viz_coll"

    new_json = {"viz_key": key, "value": value}
    db.command("insert", collection_name,
               documents=[new_json])

    client.close()
    return None

def update_key(key, value):
    """Update the value for a key in the Mongo database."""
    client = MongoClient()
    db = client.viz_db
    collection_name = "viz_coll"

    new_json = {"viz_key": key, "value": value}
    db.command("update", collection_name,
             updates=[{'q': {'viz_key': key},
                       'u': new_json}])

    client.close()
    return True

def delete_key(key):
    """Delete the item associated with a key in a Mongo database."""
    client = MongoClient()
    db = client.viz_db
    collection_name = "viz_coll"

    db.command("delete", collection_name,
               deletes=[{'q': {'viz_key': key}}],
               limit=1)

    client.close()
    return None

def delete_all_docs():
    """Delete everything from the database."""
    client = MongoClient()
    db = client.viz_db
    db.command("dropDatabase")
    client.close()
    return None
